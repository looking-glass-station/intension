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

MODELS = {
    "sortformer-offline": "nvidia/diar_sortformer_4spk-v1",
    "sortformer-streaming": "nvidia/diar_streaming_sortformer_4spk-v2.1",
}
CLUSTER_CONF = {
    "clustering-general": "diar_infer_general.yaml",
    "clustering-meeting": "diar_infer_meeting.yaml",
}
CONF_DIR = Path(__file__).resolve().parent / "conf"


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
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            out.append({"start": start, "end": start + dur, "label": p[7]})
    return out


# --------------------------------------------------------------------------- #
def run_sortformer(mode: str, audio: list[Path], out_dir: Path, device: str, batch_size: int):
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    t0 = time.perf_counter()
    model = SortformerEncLabelModel.from_pretrained(MODELS[mode])
    model.eval().to(device)
    load_sec = time.perf_counter() - t0

    rows = []
    for wav in audio:
        info = sf.info(str(wav))
        audio_sec = info.frames / float(info.samplerate)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            preds = model.diarize(audio=str(wav), batch_size=batch_size, verbose=False)
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
    return {"model": MODELS[mode], "model_load_sec": round(load_sec, 2)}, rows


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
    cfg.verbose = False

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
    args = ap.parse_args()

    ensure_hf_token()
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audio = [p.resolve() for p in args.audio]

    runner = run_sortformer if args.mode in MODELS else run_clustering
    meta, rows = runner(args.mode, audio, args.out_dir, device, args.batch_size)

    summary = {"mode": args.mode, "device": device, "torch": torch.__version__, **meta, "files": rows}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[nemo:{args.mode}] done", flush=True)


if __name__ == "__main__":
    main()
