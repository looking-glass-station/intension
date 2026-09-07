"""
Diarization benchmark harness.

Compares diarization backends on the fixed sample in ``diarization/sample.txt``:

  * ``pyannote`` (default) - the project's current diarizer
    (pyannote/speaker-diarization-3.1 via whisperx), run in-process.
  * ``nemo-sf-offline``    - NeMo diar_sortformer_4spk-v1 (OOMs on long audio)
  * ``nemo-sf-stream``     - NeMo diar_streaming_sortformer_4spk-v2.1
  * ``nemo-clust-general`` - NeMo ClusteringDiarizer, diar_infer_general.yaml
  * ``nemo-clust-meeting`` - NeMo ClusteringDiarizer, diar_infer_meeting.yaml

  The nemo-* backends run through the isolated ``tools/nemo_diarize`` env as a
  subprocess (one call for the whole batch - NeMo's import alone costs ~2 min).

Both produce the same per-file record: realtime factor, GPU peak, and a
characterisation of the speaker segmentation. Segmentation stats are always
recomputed here from the hypothesis RTTM, so the two backends are measured by
identical code.

Outputs (under benchmarks/diarization/, git-tracked):
  results_<backend>.csv        one row per file, appended; existing files skipped
  runs_<backend>.jsonl         one row per invocation (env, device, git, totals)
  hyp_<backend>/<stem>.rttm    hypothesis segments

Never reads or writes anything under data/.

    uv run python benchmarks/bench_diarize.py                 # pyannote
    uv run python benchmarks/bench_diarize.py --backend nemo
    uv run python benchmarks/bench_diarize.py --limit 2 --fresh
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
from typing import Dict, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BENCH_DIR = ROOT / "benchmarks" / "diarization"
NEMO_DIR = ROOT / "tools" / "nemo_diarize"

print = functools.partial(print, flush=True)  # noqa: A001  progress must survive piping

CSV_FIELDS = [
    "run_ts", "backend", "channel", "file",
    "audio_sec", "sr", "subtype", "wall_sec", "rtx", "torch_gpu_peak_mb",
    "n_speakers", "n_segments",
    "speech_sec", "speech_ratio",
    "mean_seg_sec", "median_seg_sec", "max_seg_sec",
    "speaker_speech_json",
]

Segment = Dict[str, float]


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


def rel_of(wav_path: Path) -> str:
    return str(wav_path.relative_to(ROOT)).replace("\\", "/")


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
            block = int(i / sr / 2.5) % 2
            freq = 180 if block == 0 else 320
            amp = 0 if (i // sr) % 5 == 4 else 6000
            frames += struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sr)))
        w.writeframes(bytes(frames))
    return dest


def write_rttm(rttm_path: Path, wav_name: str, segments: List[Segment]) -> None:
    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    name = wav_name.replace(" ", "_")
    rttm_path.write_text(
        "".join(
            f"SPEAKER {name} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} "
            f"<NA> <NA> {s['label']} <NA> <NA>\n"
            for s in segments
        ),
        encoding="utf-8",
    )


def read_rttm(rttm_path: Path) -> List[Segment]:
    segs: List[Segment] = []
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        start, dur = float(p[3]), float(p[4])
        segs.append({"start": start, "end": start + dur, "label": p[7]})
    return segs


def characterise(segments: List[Segment], audio_sec: float) -> Dict[str, object]:
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
# backends
#
# each yields (wav_path, wall_sec, gpu_mb) per file, having already written
# hyp_<backend>/<stem>.rttm. Segmentation stats are computed by the caller from
# that RTTM. Returns a (meta_dict, load_sec) pair via the generator's return.
# --------------------------------------------------------------------------- #
def run_pyannote(todo: List[Path], hyp_dir: Path) -> Iterator[Tuple[Path, float, float]]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    os.environ.setdefault("INTENSION_MERGE_SHORT_DIARIZATION_SEGMENTS", "0")  # measure raw model

    import torch
    import whisperx  # noqa: F401
    from diarize import Diarizer

    t0 = time.perf_counter()
    diarizer = Diarizer()
    diarizer._ensure_model()
    model = diarizer.model
    load_sec = time.perf_counter() - t0
    print(f"pyannote model loaded in {load_sec:.1f}s")

    warm = make_warmup_clip(Path(os.environ.get("TEMP", "/tmp")) / "bench_diarize_warmup.wav")
    t0 = time.perf_counter()
    try:
        model(str(warm))
    except Exception as e:
        print(f"  warmup raised {e!r}")
    print(f"warmup ({time.perf_counter() - t0:.1f}s) done\n")

    def diarize(wav: Path) -> List[Segment]:
        result = model(str(wav))
        df = result["diarization"] if isinstance(result, dict) and "diarization" in result else result
        return [
            {"start": float(r["start"]), "end": float(r["end"]),
             "label": r.get("speaker") or r.get("label")}
            for _, r in df.iterrows()
        ]

    for wav in todo:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        segs = diarize(wav)
        wall = time.perf_counter() - t0
        gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
        write_rttm(hyp_dir / f"{wav.stem}.rttm", wav.name, segs)
        yield wav, wall, gpu_mb

    return {
        "model_name": "pyannote/speaker-diarization-3.1",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "model_load_sec": round(load_sec, 2),
    }


# bench backend name -> nemo_diarize.py --mode
NEMO_MODES = {
    "nemo-sf-offline": "sortformer-offline",
    "nemo-sf-stream": "sortformer-streaming",
    "nemo-clust-general": "clustering-general",
    "nemo-clust-meeting": "clustering-meeting",
}


def make_nemo_runner(mode: str):
    def run(todo: List[Path], hyp_dir: Path) -> Iterator[Tuple[Path, float, float]]:
        py = NEMO_DIR / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not py.exists():
            raise SystemExit(f"NeMo env not synced: {py} missing (run `uv sync` in {NEMO_DIR})")

        summary_path = BENCH_DIR / f"_nemo_summary_{mode}.json"
        cmd = [
            str(py), str(NEMO_DIR / "nemo_diarize.py"),
            "--mode", mode,
            "--out-dir", str(hyp_dir),
            "--json-out", str(summary_path),
            "--device", "auto",
            "--audio", *[str(p) for p in todo],
        ]
        print(f"running NeMo worker (mode={mode}) over {len(todo)} files "
              f"(one process; NeMo import ~2 min)...")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd)
        print(f"NeMo worker finished in {(time.perf_counter() - t0) / 60:.1f} min (rc={proc.returncode})")
        if not summary_path.exists():
            raise SystemExit(f"NeMo worker produced no summary (rc={proc.returncode})")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        by_stem = {r["stem"]: r for r in summary["files"]}
        for wav in todo:
            r = by_stem.get(wav.stem)
            if r is None:
                print(f"  !! no NeMo result for {wav.stem}")
                continue
            if r.get("error"):
                print(f"  !! {wav.stem}: {r['error']}")
            yield wav, float(r["wall_sec"]), float(r.get("torch_gpu_peak_mb") or 0.0)

        return {
            "model_name": summary.get("model"),
            "device": summary.get("device"),
            "torch": summary.get("torch"),
            "model_load_sec": summary.get("model_load_sec"),
        }

    return run


RUNNERS = {"pyannote": run_pyannote, **{k: make_nemo_runner(v) for k, v in NEMO_MODES.items()}}


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
    ap.add_argument("--backend", default="pyannote", choices=sorted(RUNNERS))
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

    todo = [p for p in sample if rel_of(p) not in done]
    print(f"backend={args.backend}  sample={len(sample)}  done={len(sample) - len(todo)}  to run={len(todo)}")
    if not todo:
        print("nothing to do (use --fresh to redo)")
        return

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    probes = {wav: wav_probe(wav) for wav in todo}

    fh = results_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if fh.tell() == 0:
        writer.writeheader()

    totals = {"audio": 0.0, "wall": 0.0}
    gen = RUNNERS[args.backend](todo, hyp_dir)
    meta: Dict[str, object] = {}
    try:
        while True:
            wav, wall_sec, gpu_mb = next(gen)
            probe = probes[wav]
            audio_sec = float(probe["audio_sec"])
            segs = read_rttm(hyp_dir / f"{wav.stem}.rttm")
            stats = characterise(segs, audio_sec)
            row = {
                "run_ts": run_ts, "backend": args.backend,
                "channel": channel_of(wav), "file": rel_of(wav),
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
                f"  {wav.name[:55]:55}  {row['wall_sec']:7.1f}s  {row['rtx']:6.1f}xRT  "
                f"{row['n_speakers']:2} spk  {row['n_segments']:5} seg  speech {row['speech_ratio'] * 100:3.0f}%"
            )
    except StopIteration as stop:
        meta = stop.value or {}
    finally:
        fh.close()

    with runs_jsonl.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps({
            "run_ts": run_ts, "backend": args.backend, "git": git_commit(),
            "files": len(todo),
            "audio_sec": round(totals["audio"], 1),
            "wall_sec": round(totals["wall"], 1),
            "overall_rtx": round(totals["audio"] / totals["wall"], 2) if totals["wall"] else 0.0,
            **meta,
        }) + "\n")

    if totals["wall"]:
        print(
            f"\n=== {args.backend}: {len(todo)} files, {totals['audio'] / 3600:.2f}h audio "
            f"in {totals['wall'] / 60:.1f} min => {totals['audio'] / totals['wall']:.1f}x realtime ==="
        )


if __name__ == "__main__":
    main()
