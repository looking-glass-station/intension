"""
Compare two diarization backends' benchmark output.

Reads ``results_<a>.csv`` / ``results_<b>.csv`` and the matching
``hyp_<a>/`` / ``hyp_<b>/`` RTTMs, then reports:

  * speed (xRT) and output shape (speaker count, segment count, speech %) side by
    side, per file;
  * "dominant speakers" - labels holding >= --dominant-sec of speech - since raw
    label counts are inflated by clip/music fragments;
  * how far apart the two segmentations are, via pyannote.metrics DER with <a> as
    reference and <b> as hypothesis (confusion / missed / false-alarm split).

Writes ``compare_<a>_vs_<b>.md`` and prints it.

    uv run python benchmarks/bench_compare.py                 # pyannote vs nemo
    uv run python benchmarks/bench_compare.py --a pyannote --b nemo
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

BENCH_DIR = Path(__file__).resolve().parent / "diarization"


def read_results(backend: str) -> Dict[str, dict]:
    path = BENCH_DIR / f"results_{backend}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} - run bench_diarize.py --backend {backend}")
    with path.open(encoding="utf-8") as fh:
        return {r["file"]: r for r in csv.DictReader(fh)}


def read_rttm(path: Path) -> List[tuple]:
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            segs.append((start, start + dur, p[7]))
    return segs


def dominant_speakers(speaker_speech_json: str, min_sec: float) -> int:
    per = json.loads(speaker_speech_json)
    return sum(1 for v in per.values() if v >= min_sec)


def der_split(ref_segs, hyp_segs) -> dict:
    """pyannote.metrics DER of hyp against ref, as fractions of ref speech."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def to_annotation(segs) -> Annotation:
        ann = Annotation()
        for i, (s, e, label) in enumerate(segs):
            if e > s:
                ann[Segment(s, e), i] = label
        return ann

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    c = metric(to_annotation(ref_segs), to_annotation(hyp_segs), detailed=True)
    total = c["total"] or 1.0
    return {
        "der": round((c["confusion"] + c["missed detection"] + c["false alarm"]) / total, 3),
        "confusion": round(c["confusion"] / total, 3),
        "missed": round(c["missed detection"] / total, 3),
        "false_alarm": round(c["false alarm"] / total, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="pyannote", help="reference backend")
    ap.add_argument("--b", default="nemo", help="hypothesis backend")
    ap.add_argument("--dominant-sec", type=float, default=60.0,
                    help="a label counts as a 'real' speaker at >= this many seconds of speech")
    args = ap.parse_args()

    ra, rb = read_results(args.a), read_results(args.b)
    files = [f for f in ra if f in rb]
    if not files:
        raise SystemExit("no files in common between the two result sets")

    lines: List[str] = []
    w = lines.append
    w(f"# Diarization compare: `{args.a}` (ref) vs `{args.b}` (hyp)\n")
    w(f"`dominant` = labels with >= {args.dominant_sec:.0f}s of speech. "
      f"`DER` is {args.b} scored against {args.a} as reference (collar 0.25s) - "
      f"it measures *divergence between the two*, not correctness.\n")
    w(f"| file | min | {args.a} xRT | {args.b} xRT | "
      f"{args.a} spk (dom) | {args.b} spk (dom) | {args.a} seg | {args.b} seg | "
      f"DER | conf | miss | FA |")
    w("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")

    agg = {"audio": 0.0, "wall_a": 0.0, "wall_b": 0.0}
    der_rows = []
    for f in files:
        a, b = ra[f], rb[f]
        stem = Path(f).stem
        audio_min = float(a["audio_sec"]) / 60
        dom_a = dominant_speakers(a["speaker_speech_json"], args.dominant_sec)
        dom_b = dominant_speakers(b["speaker_speech_json"], args.dominant_sec)

        d = {"der": "-", "confusion": "-", "missed": "-", "false_alarm": "-"}
        pa, pb = BENCH_DIR / f"hyp_{args.a}" / f"{stem}.rttm", BENCH_DIR / f"hyp_{args.b}" / f"{stem}.rttm"
        if pa.exists() and pb.exists():
            try:
                d = der_split(read_rttm(pa), read_rttm(pb))
                der_rows.append(d)
            except Exception as e:  # noqa: BLE001
                d = {"der": f"err:{e}", "confusion": "-", "missed": "-", "false_alarm": "-"}

        w(f"| {stem[:40]} | {audio_min:.0f} | {a['rtx']} | {b['rtx']} | "
          f"{a['n_speakers']} ({dom_a}) | {b['n_speakers']} ({dom_b}) | "
          f"{a['n_segments']} | {b['n_segments']} | "
          f"{d['der']} | {d['confusion']} | {d['missed']} | {d['false_alarm']} |")

        agg["audio"] += float(a["audio_sec"])
        agg["wall_a"] += float(a["wall_sec"])
        agg["wall_b"] += float(b["wall_sec"])

    w("")
    w(f"**Overall speed** - {args.a}: {agg['audio'] / agg['wall_a']:.0f}xRT "
      f"({agg['wall_a'] / 60:.1f} min), "
      f"{args.b}: {agg['audio'] / agg['wall_b']:.0f}xRT ({agg['wall_b'] / 60:.1f} min), "
      f"for {agg['audio'] / 3600:.1f}h audio.")
    if der_rows:
        n = len(der_rows)
        w(f"**Mean divergence** - DER {sum(x['der'] for x in der_rows) / n:.2f} "
          f"(confusion {sum(x['confusion'] for x in der_rows) / n:.2f}, "
          f"missed {sum(x['missed'] for x in der_rows) / n:.2f}, "
          f"false-alarm {sum(x['false_alarm'] for x in der_rows) / n:.2f}).")

    out = BENCH_DIR / f"compare_{args.a}_vs_{args.b}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
