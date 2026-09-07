"""
Diarization benchmark harness.

Phase 1 baselines the project's current diarizer (pyannote/speaker-diarization-3.1
via whisperx). Phase 2 will add ``--backend nemo`` once the isolated NeMo env
exists. Both write the same shape of output so they can be diffed.

Per invocation it:
  * loads the backend model once (timed),
  * runs a fixed warmup clip (timed, discarded),
  * for each WAV in the sample: times the raw ``model(wav)`` call, records
    realtime factor + torch GPU peak, characterises the speaker segmentation,
    and writes a hypothesis RTTM.

Outputs (under benchmarks/diarization/, all git-tracked):
  results_<backend>.csv        one row per file, appended; existing files skipped
  runs_<backend>.jsonl         one row per invocation (env, device, git, totals)
  hyp_<backend>/<stem>.rttm    hypothesis segments

It never reads or writes anything under data/.

    uv run python benchmarks/bench_diarize.py
    uv run python benchmarks/bench_diarize.py --limit 2
    uv run python benchmarks/bench_diarize.py --fresh
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BENCH_DIR = ROOT / "benchmarks" / "diarization"

print = functools.partial(print, flush=True)  # noqa: A001  progress must survive piping

CSV_FIELDS = [
    "run_ts", "backend", "channel", "file",
    "audio_sec", "sr", "subtype", "wall_sec", "rtx", "torch_gpu_peak_mb",
    "n_speakers", "n_segments",
    "speech_sec", "speech_ratio",
    "mean_seg_sec", "median_seg_sec", "max_seg_sec",
    "speaker_speech_json",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_sample(sample_file: Path) -> List[Path]:
    paths: List[Path] = []
    for line in sample_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = (ROOT / line).resolve()
        if not p.exists():
            raise FileNotFoundError(f"sample entry not found: {line}")
        paths.append(p)
    return paths


def channel_of(wav_path: Path) -> str:
    """data/<channel>/<source>/<name>/wav/<file>.wav -> '<channel>/<name>'."""
    try:
        rel = wav_path.relative_to(ROOT / "data").parts
        return f"{rel[0]}/{rel[2]}" if len(rel) >= 4 else rel[0]
    except ValueError:
        return "?"


def wav_probe(wav_path: Path) -> Dict[str, object]:
    import soundfile as sf

    info = sf.info(str(wav_path))
    return {
        "audio_sec": info.frames / float(info.samplerate),
        "sr": info.samplerate,
        "subtype": info.subtype,
    }


def make_warmup_clip(dest: Path, seconds: float = 6.0, sr: int = 16000) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(int(seconds * sr)):
            # two alternating "speakers": tone A for 2s, silence 0.5s, tone B ...
            block = int(i / sr / 2.5) % 2
            freq = 180 if block == 0 else 320
            amp = 0 if (i // sr) % 5 == 4 else 6000
            frames += struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sr)))
        w.writeframes(bytes(frames))
    return dest


def segments_to_rttm(rttm_path: Path, wav_name: str, segments: List[Dict[str, float]]) -> None:
    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    name = wav_name.replace(" ", "_")
    lines = [
        f"SPEAKER {name} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} "
        f"<NA> <NA> {s['label']} <NA> <NA>\n"
        for s in segments
    ]
    rttm_path.write_text("".join(lines), encoding="utf-8")


def characterise(segments: List[Dict[str, float]], audio_sec: float) -> Dict[str, object]:
    durs = [s["end"] - s["start"] for s in segments] or [0.0]
    speech_sec = sum(durs)
    per_speaker: Dict[str, float] = {}
    for s in segments:
        per_speaker[s["label"]] = per_speaker.get(s["label"], 0.0) + (s["end"] - s["start"])
    return {
        "n_speakers": len(per_speaker),
        "n_segments": len(segments),
        "speech_sec": round(speech_sec, 2),
        "speech_ratio": round(speech_sec / audio_sec, 4) if audio_sec else 0.0,
        "mean_seg_sec": round(statistics.fmean(durs), 3),
        "median_seg_sec": round(statistics.median(durs), 3),
        "max_seg_sec": round(max(durs), 3),
        "speaker_speech_json": json.dumps({k: round(v, 1) for k, v in sorted(per_speaker.items())}),
    }


# --------------------------------------------------------------------------- #
# backends: each returns (load_fn, diarize_fn, meta_dict)
# --------------------------------------------------------------------------- #
def backend_pyannote():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    # measure the raw model, not the pipeline's post-processing merge
    os.environ.setdefault("INTENSION_MERGE_SHORT_DIARIZATION_SEGMENTS", "0")

    import torch  # noqa: F401  (imported for the caller's GPU stats)
    from diarize import Diarizer

    state: Dict[str, object] = {}

    def load() -> Dict[str, str]:
        d = Diarizer()
        d._ensure_model()
        state["model"] = d.model
        import whisperx

        return {
            "model_name": "pyannote/speaker-diarization-3.1",
            "whisperx": getattr(whisperx, "__version__", "?"),
        }

    def diarize(wav_path: Path) -> List[Dict[str, float]]:
        result = state["model"](str(wav_path))
        df = result["diarization"] if isinstance(result, dict) and "diarization" in result else result
        segs: List[Dict[str, float]] = []
        for _, row in df.iterrows():
            segs.append({
                "start": float(row["start"]),
                "end": float(row["end"]),
                "label": row.get("speaker") or row.get("label"),
            })
        return segs

    return load, diarize


def backend_nemo():
    raise SystemExit(
        "backend 'nemo' is not wired up yet (Phase 2 - needs the isolated "
        "tools/nemo_diarize/ env). Run without --backend for the pyannote baseline."
    )


BACKENDS: Dict[str, Callable] = {"pyannote": backend_pyannote, "nemo": backend_nemo}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "?"
    except Exception:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="pyannote", choices=sorted(BACKENDS))
    ap.add_argument("--sample", type=Path, default=BENCH_DIR / "sample.txt")
    ap.add_argument("--limit", type=int, default=0, help="only the first N sample files")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results and redo every file")
    args = ap.parse_args()

    sample = load_sample(args.sample)
    if args.limit:
        sample = sample[: args.limit]

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    hyp_dir = BENCH_DIR / f"hyp_{args.backend}"
    hyp_dir.mkdir(exist_ok=True)
    results_csv = BENCH_DIR / f"results_{args.backend}.csv"
    runs_jsonl = BENCH_DIR / f"runs_{args.backend}.jsonl"

    done: set[str] = set()
    if results_csv.exists() and not args.fresh:
        with results_csv.open(encoding="utf-8") as fh:
            done = {r["file"] for r in csv.DictReader(fh)}

    todo = [p for p in sample if str(p.relative_to(ROOT)).replace("\\", "/") not in done]
    print(f"backend={args.backend}  sample={len(sample)}  already done={len(sample) - len(todo)}  to run={len(todo)}")
    if not todo:
        print("nothing to do (use --fresh to redo)")
        return

    import torch

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    load_fn, diarize_fn = BACKENDS[args.backend]()

    t0 = time.perf_counter()
    backend_meta = load_fn()
    load_sec = time.perf_counter() - t0
    print(f"model loaded in {load_sec:.1f}s")

    warm = make_warmup_clip(Path(os.environ.get("TEMP", "/tmp")) / "bench_diarize_warmup.wav")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        diarize_fn(warm)
    except Exception as e:  # warmup failure isn't fatal, but say so
        print(f"  warmup call raised {e!r}")
    warm_sec = time.perf_counter() - t0
    print(f"warmup ({warm_sec:.1f}s) done\n")

    new_file = not results_csv.exists()
    fh = results_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()

    totals = {"audio": 0.0, "wall": 0.0}
    for i, wav in enumerate(todo, 1):
        rel = str(wav.relative_to(ROOT)).replace("\\", "/")
        probe = wav_probe(wav)
        audio_sec = probe["audio_sec"]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"[{i}/{len(todo)}] {wav.name}  ({audio_sec / 60:.1f} min, {probe['subtype']})")

        t0 = time.perf_counter()
        segments = diarize_fn(wav)
        wall_sec = time.perf_counter() - t0
        gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0

        segments_to_rttm(hyp_dir / f"{wav.stem}.rttm", wav.name, segments)
        stats = characterise(segments, audio_sec)
        row = {
            "run_ts": run_ts, "backend": args.backend,
            "channel": channel_of(wav), "file": rel,
            "audio_sec": round(audio_sec, 2), "sr": probe["sr"], "subtype": probe["subtype"],
            "wall_sec": round(wall_sec, 2),
            "rtx": round(audio_sec / wall_sec, 2) if wall_sec else 0.0,
            "torch_gpu_peak_mb": round(gpu_mb, 1),
            **stats,
        }
        writer.writerow(row)
        fh.flush()
        totals["audio"] += audio_sec
        totals["wall"] += wall_sec
        print(
            f"      {row['wall_sec']:.1f}s  {row['rtx']:.1f}xRT  "
            f"{row['n_speakers']} spk  {row['n_segments']} seg  "
            f"speech {row['speech_ratio'] * 100:.0f}%  gpu {row['torch_gpu_peak_mb']:.0f}MB"
        )

    fh.close()

    with runs_jsonl.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps({
            "run_ts": run_ts,
            "backend": args.backend,
            "git": git_commit(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "model_load_sec": round(load_sec, 2),
            "warmup_sec": round(warm_sec, 2),
            "files": len(todo),
            "audio_sec": round(totals["audio"], 1),
            "wall_sec": round(totals["wall"], 1),
            "overall_rtx": round(totals["audio"] / totals["wall"], 2) if totals["wall"] else 0.0,
            **backend_meta,
        }) + "\n")

    print(
        f"\n=== {args.backend}: {len(todo)} files, "
        f"{totals['audio'] / 3600:.2f}h audio in {totals['wall'] / 60:.1f} min "
        f"=> {totals['audio'] / totals['wall']:.1f}x realtime (excl. {load_sec:.0f}s load) ==="
    )


if __name__ == "__main__":
    main()
