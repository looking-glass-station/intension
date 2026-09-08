"""
Q2: does the diarizer's segmentation change the transcription?

`transcribe.py` runs ASR per RTTM segment, so a different diarization = a
different set of ASR calls. This runs the project's Transcriber over the same
WAVs with the pyannote RTTM vs the sf-stream-ch RTTM (both from bench_diarize's
hyp_<backend>/) and compares wall-clock + the resulting transcript.

    uv run python benchmarks/bench_transcribe.py
    uv run python benchmarks/bench_transcribe.py --files "#361" "20 Lawyer" "FREEZING"

Writes benchmarks/diarization/tr_<backend>/<stem>.csv and compare_transcribe.md.
Needs ffmpeg on PATH and the whisper model cached.
"""
from __future__ import annotations

import argparse
import csv
import functools
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks" / "diarization"
print = functools.partial(print, flush=True)  # noqa: A001

DEFAULT_FILES = ["#361", "20 Lawyer Fight", "FREEZING PROTEST"]
DEFAULT_BACKENDS = ["pyannote", "nemo-sf-stream-ch"]


def sample_wavs() -> list[Path]:
    out = []
    for line in (BENCH / "sample.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append((ROOT / line).resolve())
    return out


def rttm_segments(p: Path) -> list[dict]:
    segs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        x = line.split()
        if len(x) >= 8 and x[0] == "SPEAKER":
            s, d = float(x[-7]), float(x[-6])
            segs.append({"start": s, "end": s + d, "spk": x[-3]})
    return segs


def read_transcript(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def wordcount(rows: list[dict]) -> int:
    return sum(len(re.findall(r"\S+", r.get("text") or "")) for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="substrings matched against wav stems")
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    args = ap.parse_args()

    wavs = sample_wavs()
    picked = []
    for needle in args.files:
        hit = next((w for w in wavs if needle.lower() in w.stem.lower()), None)
        if hit is None:
            raise SystemExit(f"no sample wav matches {needle!r}")
        picked.append(hit)

    sys.path.insert(0, str(SRC))
    import soundfile as sf
    from transcribe import Transcriber

    t0 = time.perf_counter()
    tr = Transcriber()
    print(f"Transcriber loaded in {time.perf_counter() - t0:.1f}s "
          f"(batched={tr.use_batched_transcribe}, batch_size={tr.transcribe_batch_size})\n")

    rows_out: list[dict] = []
    for wav in picked:
        audio_sec = sf.info(str(wav)).frames / sf.info(str(wav)).samplerate
        for backend in args.backends:
            rttm = BENCH / f"hyp_{backend}" / f"{wav.stem}.rttm"
            if not rttm.exists():
                print(f"  !! {backend}: {rttm.name} missing, skip")
                continue
            segs = rttm_segments(rttm)
            valid = sum(1 for s in segs if s["end"] - s["start"] >= 0.5)
            out_dir = BENCH / f"tr_{backend}"
            out_dir.mkdir(exist_ok=True)
            tcsv = out_dir / f"{wav.stem}.csv"
            taud = out_dir / f"{wav.stem}.audacity.txt"
            for f in (tcsv, taud):
                f.unlink(missing_ok=True)

            t0 = time.perf_counter()
            tr.segment_and_transcribe(rttm, wav, tcsv, taud)
            wall = time.perf_counter() - t0

            trows = read_transcript(tcsv)
            rows_out.append({
                "file": wav.stem[:40], "backend": backend,
                "audio_min": round(audio_sec / 60, 1),
                "rttm_seg": len(segs), "valid_seg": valid,
                "tr_rows": len(trows), "words": wordcount(trows),
                "wall_s": round(wall, 1),
                "xRT": round(audio_sec / wall, 1) if wall else 0,
            })
            r = rows_out[-1]
            print(f"  {wav.stem[:45]:45} {backend:18} "
                  f"{r['rttm_seg']:5} seg -> {r['valid_seg']:5} valid -> {r['tr_rows']:4} rows  "
                  f"{r['words']:6} words  {r['wall_s']:6.1f}s ({r['xRT']}xRT)")

    # summary table
    lines = ["# Q2: transcription with pyannote vs sf-stream-ch RTTM\n",
             "| file | min | backend | rttm seg | valid | tr rows | words | wall s | xRT |",
             "|---|--:|---|--:|--:|--:|--:|--:|--:|"]
    for r in rows_out:
        lines.append(f"| {r['file']} | {r['audio_min']} | {r['backend']} | {r['rttm_seg']} | "
                     f"{r['valid_seg']} | {r['tr_rows']} | {r['words']} | {r['wall_s']} | {r['xRT']} |")
    by_backend: dict[str, dict] = {}
    for r in rows_out:
        b = by_backend.setdefault(r["backend"], {"wall": 0.0, "words": 0, "rows": 0})
        b["wall"] += r["wall_s"]; b["words"] += r["words"]; b["rows"] += r["tr_rows"]
    lines.append("")
    for b, v in by_backend.items():
        lines.append(f"**{b}** - total {v['wall']:.0f}s transcription, {v['rows']} rows, {v['words']} words")

    out = BENCH / "compare_transcribe.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwritten to {out}")
    print(f"transcripts under {BENCH}/tr_<backend>/ - diff them for text quality")


if __name__ == "__main__":
    main()
