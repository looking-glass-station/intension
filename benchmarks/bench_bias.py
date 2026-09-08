"""
Q2b: does the diarizer's word-loss change bias detection?

sf-stream-ch drops ~18% of transcribed words on reaction streams vs pyannote.
This runs bias.py's classifier over both transcripts (from bench_transcribe's
tr_<backend>/) and diffs: how many segments survive the keyword pre-filter, how
many bias findings, and which findings one transcript has that the other misses.

    uv run python benchmarks/bench_bias.py
    uv run python benchmarks/bench_bias.py --files "#361" "FREEZING"

Needs Ollama running with the bias.json model pulled. Writes bias_<backend>/ and
compare_bias.md.
"""
from __future__ import annotations

import argparse
import csv
import functools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "diarization"
print = functools.partial(print, flush=True)  # noqa: A001

DEFAULT_FILES = ["#361", "FREEZING PROTEST"]
DEFAULT_BACKENDS = ["pyannote", "nemo-sf-stream-ch"]


def sample_stems() -> list[str]:
    out = []
    for line in (BENCH / "sample.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(Path(line).stem)
    return out


def overlaps(a: dict, b: dict, tol: float = 15.0) -> bool:
    return abs(float(a["start_time"]) - float(b["start_time"])) <= tol and a["topic"] == b["topic"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    args = ap.parse_args()

    stems = sample_stems()
    picked = []
    for needle in args.files:
        hit = next((s for s in stems if needle.lower() in s.lower()), None)
        if hit is None:
            raise SystemExit(f"no sample stem matches {needle!r}")
        picked.append(hit)

    sys.path.insert(0, str(ROOT / "src"))
    import ollama
    from bias import (add_channel_topic, build_system_prompt, classify_bias_for_transcript,
                      load_bias_config, segment_matches_any_topic)
    from logger import global_logger

    cfg = load_bias_config()
    topics = cfg["topics"]
    add_channel_topic(topics)
    system_prompt = build_system_prompt(topics)
    model = cfg.get("model", "gemma3:12b")
    batch_size = cfg.get("batch_size", 10)
    min_dur = cfg.get("min_segment_duration", 5)
    ollama.show(model)  # fail early if Ollama/model missing
    logger = global_logger("bench_bias")

    lines = ["# Q2b: bias detection - pyannote vs sf-stream-ch transcript\n"]
    findings_by = {}
    for stem in picked:
        lines.append(f"## {stem}\n")
        lines.append("| backend | tr rows | dur>5s | keyword hit | findings | wall s |")
        lines.append("|---|--:|--:|--:|--:|--:|")
        for be in args.backends:
            tcsv = BENCH / f"tr_{be}" / f"{stem}.csv"
            if not tcsv.exists():
                print(f"  !! {be}: {tcsv} missing"); continue
            rows = list(csv.DictReader(tcsv.open(encoding="utf-8")))
            dur_ok = [r for r in rows if float(r["end_time"]) - float(r["start_time"]) > min_dur]
            kw_hit = [r for r in dur_ok if segment_matches_any_topic(r.get("text") or "", topics)]

            out_dir = BENCH / f"bias_{be}"
            out_dir.mkdir(exist_ok=True)
            bcsv, baud = out_dir / f"{stem}.csv", out_dir / f"{stem}.audacity.txt"
            for f in (bcsv, baud):
                f.unlink(missing_ok=True)

            t0 = time.perf_counter()
            classify_bias_for_transcript(tcsv, bcsv, baud, topics, system_prompt,
                                         model, batch_size, min_dur, logger)
            wall = time.perf_counter() - t0
            findings = list(csv.DictReader(bcsv.open(encoding="utf-8"))) if bcsv.exists() else []
            findings_by[(stem, be)] = findings
            lines.append(f"| {be} | {len(rows)} | {len(dur_ok)} | {len(kw_hit)} | {len(findings)} | {wall:.0f} |")
            print(f"  {stem[:40]:40} {be:18} rows={len(rows):4} kw={len(kw_hit):3} findings={len(findings):3} ({wall:.0f}s)")

        # diff findings between the two backends
        a, b = args.backends[0], args.backends[1]
        fa, fb = findings_by.get((stem, a), []), findings_by.get((stem, b), [])
        only_a = [f for f in fa if not any(overlaps(f, g) for g in fb)]
        only_b = [f for f in fb if not any(overlaps(f, g) for g in fa)]
        lines.append(f"\n**shared** ~{len(fa) - len(only_a)}  |  **only {a}**: {len(only_a)}  |  **only {b}**: {len(only_b)}\n")
        for tag, fs in ((f"only {a}", only_a), (f"only {b}", only_b)):
            for f in fs[:8]:
                lines.append(f"- _{tag}_ [{float(f['start_time']):.0f}s {f['topic']}/{f['stance']}] "
                             f"score {f['score']} - {(f['text'] or '')[:120]}")
        lines.append("")

    out = BENCH / "compare_bias.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
