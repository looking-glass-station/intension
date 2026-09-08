"""
NeMo diarization worker (multi-backend).

Runs in this directory's isolated .venv and is invoked as a subprocess by
``benchmarks/bench_diarize.py`` (and, eventually, ``src/diarize.py``).

    python nemo_diarize.py --mode MODE --audio A.wav [B.wav ...] --out-dir DIR
                           [--json-out summary.json] [--device auto|cuda|cpu]

Modes:
  sortformer-offline    nvidia/diar_sortformer_4spk-v1 - end-to-end, <=4 speakers.
                        Processes the whole file at once: OOMs on long audio.
  sortformer-streaming  nvidia/diar_streaming_sortformer_4spk-v2.1 - end-to-end,
                        <=4 speakers, streaming state so it handles long audio.
  clustering-general    NeMo ClusteringDiarizer, diar_infer_general.yaml
  clustering-meeting    NeMo ClusteringDiarizer, diar_infer_meeting.yaml
                        (VAD MarbleNet + TitaNet-L embeddings + NME-SC clustering;
                        arbitrary length and speaker count)

For each input it writes ``<out-dir>/<stem>.rttm`` (pipeline format) and, with
``--json-out``, a JSON summary (per-file timings + speaker stats). NeMo spews to
stdout/stderr, so machine-readable output goes to the file only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
CONF_DIR = HERE / "conf"
MODEL_DIR = HERE / "models"  # local .nemo checkpoints (gitignored; downloads are flaky here)

MODELS = {  # mode -> (hf repo id, local .nemo filename)
    "sortformer-offline": ("nvidia/diar_sortformer_4spk-v1", "diar_sortformer_4spk-v1.nemo"),
    "sortformer-streaming": ("nvidia/diar_streaming_sortformer_4spk-v2.1",
                             "diar_streaming_sortformer_4spk-v2.1.nemo"),
}
CLUSTER_CONF = {
    "clustering-general": "diar_infer_general.yaml",
    "clustering-meeting": "diar_infer_meeting.yaml",
}
# local .nemo for the ClusteringDiarizer sub-models, if present
VAD_NEMO = MODEL_DIR / "vad_multilingual_marblenet.nemo"
SPK_NEMO = MODEL_DIR / "titanet-l.nemo"

# streaming Sortformer chunking, in 80 ms frames (from the v2.1 model card).
# "high" = the 30.4 s-latency preset: for offline batch it is both faster (RTF
# ~0.002) and 1-2 DER points better than the low-latency preset.
SF_STREAM_PRESETS = {
    "high": dict(chunk_len=340, chunk_right_context=40, fifo_len=40,
                 spkcache_update_period=300, spkcache_len=188),
    "low": dict(chunk_len=6, chunk_right_context=7, fifo_len=188,
                spkcache_update_period=144, spkcache_len=188),
}
POST_PROC = {  # name -> yaml under conf/post_processing/
    "callhome": "diar_streaming_sortformer_4spk-v2_callhome-part1.yaml",
    "dihard3": "diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml",
}


def load_sortformer(mode: str):
    """Prefer a local .nemo; fall back to HF (which is unreliable on this box)."""
    from nemo.collections.asr.models import SortformerEncLabelModel

    repo, fname = MODELS[mode]
    local = MODEL_DIR / fname
    if local.exists():
        return SortformerEncLabelModel.restore_from(str(local), map_location="cpu"), str(local)
    return SortformerEncLabelModel.from_pretrained(repo), repo


def ensure_hf_token() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    token_file = Path(__file__).resolve().parents[2] / "tokens" / "huggingface"
    if token_file.exists():
        tok = token_file.read_text(encoding="utf-8").strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok


def write_rttm(path: Path, wav_name: str, segs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = wav_name.replace(" ", "_")
    path.write_text(
        "".join(
            f"SPEAKER {name} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} "
            f"<NA> <NA> {s['label']} <NA> <NA>\n"
            for s in segs
        ),
        encoding="utf-8",
    )


def characterise(segs: list[dict]) -> dict:
    per: dict[str, float] = {}
    for s in segs:
        per[s["label"]] = per.get(s["label"], 0.0) + (s["end"] - s["start"])
    return {
        "n_speakers": len(per),
        "n_segments": len(segs),
        "speech_sec": round(sum(per.values()), 2),
        "speaker_speech": {k: round(v, 1) for k, v in sorted(per.items())},
    }


def parse_nemo_segments(raw: list[str]) -> list[dict]:
    """Sortformer returns ['<start> <end> <speaker>', ...] per audio file."""
    out = []
    for line in raw:
        p = line.split()
        if len(p) >= 3:
            out.append({"start": float(p[0]), "end": float(p[1]), "label": p[2]})
    return out


def read_rttm(path: Path) -> list[dict]:
    """Parse from the right - NeMo writes unescaped spaces in the file id."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 10 and p[0] == "SPEAKER":
            start, dur, label = float(p[-7]), float(p[-6]), p[-3]
            out.append({"start": start, "end": start + dur, "label": label})
    return out


# --------------------------------------------------------------------------- #
def run_sortformer(mode, audio, out_dir, device, batch_size, *,
                   sf_latency="high", postprocessing="none"):
    import torch

    t0 = time.perf_counter()
    model, model_src = load_sortformer(mode)
    model.eval().to(device)

    streaming = mode == "sortformer-streaming"
    if streaming:
        for k, v in SF_STREAM_PRESETS[sf_latency].items():
            setattr(model.sortformer_modules, k, v)
        model.sortformer_modules._check_streaming_parameters()
    pp_yaml = str(CONF_DIR / "post_processing" / POST_PROC[postprocessing]) \
        if postprocessing != "none" else None
    load_sec = time.perf_counter() - t0

    rows = []
    for wav in audio:
        info = sf.info(str(wav))
        audio_sec = info.frames / float(info.samplerate)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            preds = model.diarize(audio=str(wav), batch_size=batch_size,
                                  postprocessing_yaml=pp_yaml, verbose=False)
            segs = parse_nemo_segments(preds[0] if preds else [])
            err = None
        except Exception as e:  # noqa: BLE001  record and continue
            segs, err = [], f"{type(e).__name__}: {str(e)[:200]}"
        wall = time.perf_counter() - t0
        gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0
        write_rttm(out_dir / f"{wav.stem}.rttm", wav.name, segs)
        rows.append(_row(wav, info, audio_sec, wall, gpu_mb, segs, err))
        print(f"[nemo:{mode}] {wav.name}: {rows[-1]['wall_sec']}s "
              f"{rows[-1]['n_speakers']}spk {rows[-1]['n_segments']}seg"
              + (f"  ERROR {err}" if err else ""), flush=True)
    return {
        "model": model_src, "model_load_sec": round(load_sec, 2),
        "sf_latency": sf_latency if streaming else None,
        "postprocessing": postprocessing,
    }, rows


def run_clustering(mode: str, audio: list[Path], out_dir: Path, device: str, batch_size: int):
    import torch
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer

    conf_path = CONF_DIR / CLUSTER_CONF[mode]
    cfg = OmegaConf.load(str(conf_path))
    work = Path(tempfile.mkdtemp(prefix="nemo_clust_"))
    manifest = work / "manifest.json"
    with manifest.open("w", encoding="utf-8") as fh:
        for wav in audio:
            fh.write(json.dumps({
                "audio_filepath": str(wav), "offset": 0, "duration": None,
                "label": "infer", "text": "-", "num_speakers": None,
                "rttm_filepath": None, "uem_filepath": None,
            }) + "\n")

    cfg.diarizer.manifest_filepath = str(manifest)
    cfg.diarizer.out_dir = str(work / "out")
    cfg.device = device
    cfg.batch_size = batch_size
    cfg.num_workers = 0  # Windows: spawned dataloader workers can't pickle NeMo collections
    cfg.verbose = False
    # prefer local .nemo checkpoints over NGC/HF (both flaky on this box)
    if VAD_NEMO.exists():
        cfg.diarizer.vad.model_path = str(VAD_NEMO)
    if SPK_NEMO.exists():
        cfg.diarizer.speaker_embeddings.model_path = str(SPK_NEMO)

    t0 = time.perf_counter()
    diar = ClusteringDiarizer(cfg=cfg)
    load_sec = time.perf_counter() - t0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    diar.diarize()
    total_wall = time.perf_counter() - t0
    gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0

    pred_dir = Path(cfg.diarizer.out_dir) / "pred_rttms"
    per_file_wall = total_wall / max(len(audio), 1)  # ClusteringDiarizer batches; no per-file split
    rows = []
    for wav in audio:
        info = sf.info(str(wav))
        audio_sec = info.frames / float(info.samplerate)
        src = pred_dir / f"{wav.stem}.rttm"
        segs = read_rttm(src) if src.exists() else []
        err = None if src.exists() else "no pred rttm"
        write_rttm(out_dir / f"{wav.stem}.rttm", wav.name, segs)
        rows.append(_row(wav, info, audio_sec, per_file_wall, gpu_mb, segs, err))
        print(f"[nemo:{mode}] {wav.name}: {rows[-1]['n_speakers']}spk "
              f"{rows[-1]['n_segments']}seg" + (f"  {err}" if err else ""), flush=True)

    shutil.rmtree(work, ignore_errors=True)
    return {
        "model": f"ClusteringDiarizer/{CLUSTER_CONF[mode]}",
        "model_load_sec": round(load_sec, 2),
        "batched_total_wall_sec": round(total_wall, 2),
    }, rows


def _row(wav: Path, info, audio_sec, wall, gpu_mb, segs, err) -> dict:
    return {
        "stem": wav.stem, "input": str(wav),
        "audio_sec": round(audio_sec, 2), "sr": info.samplerate, "subtype": info.subtype,
        "wall_sec": round(wall, 2),
        "rtx": round(audio_sec / wall, 2) if wall else 0.0,
        "torch_gpu_peak_mb": round(gpu_mb, 1),
        "error": err,
        **characterise(segs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=[*MODELS, *CLUSTER_CONF])
    ap.add_argument("--audio", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--sf-latency", default="high", choices=sorted(SF_STREAM_PRESETS),
                    help="streaming Sortformer chunking preset (offline: 'high' is best)")
    ap.add_argument("--postprocessing", default="none", choices=["none", *POST_PROC],
                    help="Sortformer onset/offset/min-duration tuning")
    args = ap.parse_args()

    ensure_hf_token()
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audio = [p.resolve() for p in args.audio]

    if args.mode in MODELS:
        meta, rows = run_sortformer(args.mode, audio, args.out_dir, device, args.batch_size,
                                    sf_latency=args.sf_latency, postprocessing=args.postprocessing)
    else:
        meta, rows = run_clustering(args.mode, audio, args.out_dir, device, args.batch_size)

    summary = {"mode": args.mode, "device": device, "torch": torch.__version__, **meta, "files": rows}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[nemo:{args.mode}] done", flush=True)


if __name__ == "__main__":
    main()
