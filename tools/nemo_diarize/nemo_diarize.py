"""
NeMo Sortformer diarization worker.

Runs in this directory's isolated .venv (NeMo pins torch/transformers/numpy in ways
that are incompatible with the main intension env). Invoked as a subprocess by
``src/diarize.py`` and ``benchmarks/bench_diarize.py``.

    python nemo_diarize.py --audio A.wav [B.wav ...] --out-dir DIR
                           [--model nvidia/diar_sortformer_4spk-v1]
                           [--device auto|cuda|cpu] [--json-out summary.json]

For each input it writes ``<out-dir>/<stem>.rttm`` (same format the pipeline uses)
and, if ``--json-out`` is given, a JSON summary there (timings + per-file speaker
stats). NeMo spews to stdout/stderr, so machine-readable output goes to the file,
never stdout.

Note: diar_sortformer_4spk-v1 is capped at 4 speakers. For content with more
(streams, call-ins) use --model with a clustering config instead, or expect the
extra speakers to be merged.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import soundfile as sf


def ensure_hf_token() -> None:
    """
    Sortformer is a gated-ish HF download; unauthenticated pulls get rate-limited
    to a stall. Reuse the pipeline's token if the caller didn't pass one in env.
    """
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    token_file = Path(__file__).resolve().parents[2] / "tokens" / "huggingface"
    if token_file.exists():
        tok = token_file.read_text(encoding="utf-8").strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok


def parse_segments(raw: list[str]) -> list[dict]:
    """NeMo returns ['<start> <end> <speaker>', ...] per audio file."""
    segs: list[dict] = []
    for line in raw:
        parts = line.split()
        if len(parts) < 3:
            continue
        start, end = float(parts[0]), float(parts[1])
        segs.append({"start": start, "end": end, "label": parts[2]})
    return segs


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="nvidia/diar_sortformer_4spk-v1")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    ensure_hf_token()

    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    model = SortformerEncLabelModel.from_pretrained(args.model)
    model.eval()
    model.to(device)
    load_sec = time.perf_counter() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for wav in args.audio:
        wav = wav.resolve()
        info = sf.info(str(wav))
        audio_sec = info.frames / float(info.samplerate)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        preds = model.diarize(audio=str(wav), batch_size=args.batch_size, verbose=False)
        wall_sec = time.perf_counter() - t0
        gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0

        segs = parse_segments(preds[0] if preds else [])
        rttm_path = args.out_dir / f"{wav.stem}.rttm"
        write_rttm(rttm_path, wav.name, segs)

        row = {
            "stem": wav.stem,
            "input": str(wav),
            "rttm": str(rttm_path),
            "audio_sec": round(audio_sec, 2),
            "sr": info.samplerate,
            "subtype": info.subtype,
            "wall_sec": round(wall_sec, 2),
            "rtx": round(audio_sec / wall_sec, 2) if wall_sec else 0.0,
            "torch_gpu_peak_mb": round(gpu_mb, 1),
            **characterise(segs),
        }
        results.append(row)
        print(f"[nemo] {wav.name}: {row['wall_sec']}s {row['rtx']}xRT "
              f"{row['n_speakers']}spk {row['n_segments']}seg", flush=True)

    summary = {
        "model": args.model,
        "device": device,
        "torch": torch.__version__,
        "model_load_sec": round(load_sec, 2),
        "files": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[nemo] done", flush=True)


if __name__ == "__main__":
    main()
