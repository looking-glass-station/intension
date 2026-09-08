"""
Whisper compute-type A/B: is `int8_float16` a safe default vs `float16`?

`transcribe.py` uses `float16` on GPU and only drops to `int8_float16` as the
`INTENSION_GPU_SAFE_MODE` crash fallback. CTranslate2's int8-quantized weights
are usually faster with little accuracy cost - this measures both on the frozen
sample so we can decide whether to flip the default.

For each compute type it runs the project's Transcriber over the sample WAVs
using the sf-stream-ch RTTMs (what the pipeline feeds now), timing each file and
saving the transcript. Then it diffs the transcripts row-by-row (same RTTM =
row-aligned) against the first compute type as reference: % rows identical,
word-level change on the rest, net word-count delta.

    uv run python benchmarks/bench_compute_type.py
    uv run python benchmarks/bench_compute_type.py --compute-types float16 int8_float16 int8
    uv run python benchmarks/bench_compute_type.py --files "#361" "20 Lawyer" "FREEZING"

Writes benchmarks/transcription/tr_<compute_type>/<stem>.csv and
benchmarks/transcription/compare_compute_type.md. Needs the whisper model cached
and the sf-stream-ch RTTMs under benchmarks/diarization/hyp_nemo-sf-stream-ch/.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import functools
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DIAR = ROOT / "benchmarks" / "diarization"
OUT = ROOT / "benchmarks" / "transcription"
print = functools.partial(print, flush=True)  # noqa: A001

DEFAULT_COMPUTE_TYPES = ["float16", "int8_float16"]
DEFAULT_RTTM_SET = "hyp_nemo-sf-stream-ch"


def sample_wavs() -> list[Path]:
    out = []
    for line in (DIAR / "sample.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append((ROOT / line).resolve())
    return out


def words(text: str) -> list[str]:
    return re.findall(r"\S+", (text or "").lower())


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def diff_transcripts(ref: list[dict], hyp: list[dict]) -> dict:
    """Row-aligned (same RTTM). Reports how much the text moved."""
    n = min(len(ref), len(hyp))
    ref_w = hyp_w = changed = word_edits = 0
    for i in range(n):
        rw, hw = words(ref[i].get("text", "")), words(hyp[i].get("text", ""))
        ref_w += len(rw)
        hyp_w += len(hw)
        if rw != hw:
            changed += 1
            sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
            word_edits += sum(
                max(i2 - i1, j2 - j1)
                for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"
            )
    return {
        "rows": n,
        "rows_changed": changed,
        "rows_identical_pct": round(100 * (n - changed) / n, 2) if n else 100.0,
        "ref_words": ref_w,
        "hyp_words": hyp_w,
        "word_delta": hyp_w - ref_w,
        # word-level divergence: edits / reference words, over the whole transcript
        "wdr_pct": round(100 * word_edits / ref_w, 2) if ref_w else 0.0,
        "len_mismatch": len(hyp) - len(ref),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compute-types", nargs="+", default=DEFAULT_COMPUTE_TYPES)
    ap.add_argument("--files", nargs="+", default=None, help="substrings matched against wav stems (default: all)")
    ap.add_argument("--rttm-set", default=DEFAULT_RTTM_SET, help="dir under benchmarks/diarization/ holding the RTTMs")
    args = ap.parse_args()

    wavs = sample_wavs()
    if args.files:
        picked = []
        for needle in args.files:
            hit = next((w for w in wavs if needle.lower() in w.stem.lower()), None)
            if hit is None:
                raise SystemExit(f"no sample wav matches {needle!r}")
            picked.append(hit)
        wavs = picked

    rttm_dir = DIAR / args.rttm_set
    pairs = []
    for w in wavs:
        r = rttm_dir / f"{w.stem}.rttm"
        if not r.exists():
            print(f"  !! no RTTM for {w.stem} under {args.rttm_set}, skip")
            continue
        pairs.append((w, r))
    if not pairs:
        raise SystemExit("no wav/RTTM pairs")

    sys.path.insert(0, str(SRC))
    import soundfile as sf

    OUT.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    per_ct: dict[str, dict] = {}

    for ct in args.compute_types:
        # Transcriber._load_model picks int8_float16 when INTENSION_GPU_SAFE_MODE is set,
        # else float16 on CUDA. For anything else, patch the model build directly.
        os.environ.pop("INTENSION_GPU_SAFE_MODE", None)
        patch = None
        if ct == "int8_float16":
            os.environ["INTENSION_GPU_SAFE_MODE"] = "1"
        elif ct != "float16":
            patch = ct  # e.g. "int8", "bfloat16"

        # fresh import each round so a new WhisperModel is built
        for m in ("transcribe",):
            sys.modules.pop(m, None)
        from transcribe import Transcriber
        import transcribe as _tr

        if patch:
            _orig = _tr.WhisperModel
            _tr.WhisperModel = lambda *a, compute_type=None, **k: _orig(*a, compute_type=patch, **k)  # noqa: E731

        t0 = time.perf_counter()
        tr = Transcriber()
        load_s = time.perf_counter() - t0
        actual = "?"
        for obj in (getattr(tr.model, "model", None), tr.model):
            ct_attr = getattr(obj, "compute_type", None)
            if ct_attr:
                actual = ct_attr
                break
        print(f"\n=== {ct}  (model reports: {actual}, load {load_s:.1f}s, "
              f"batched={tr.use_batched_transcribe} bs={tr.transcribe_batch_size}) ===")

        ct_dir = OUT / f"tr_{ct}"
        ct_dir.mkdir(exist_ok=True)
        rows = []
        for w, r in pairs:
            info = sf.info(str(w))
            audio_sec = info.frames / float(info.samplerate)
            tcsv = ct_dir / f"{w.stem}.csv"
            taud = ct_dir / f"{w.stem}.audacity.txt"
            for f in (tcsv, taud):
                f.unlink(missing_ok=True)
            t0 = time.perf_counter()
            tr.segment_and_transcribe(r, w, tcsv, taud)
            wall = time.perf_counter() - t0
            trows = read_rows(tcsv)
            wc = sum(len(words(x.get("text", ""))) for x in trows)
            rows.append({"stem": w.stem, "audio_sec": round(audio_sec, 1),
                         "wall_s": round(wall, 2), "xRT": round(audio_sec / wall, 1) if wall else 0,
                         "tr_rows": len(trows), "words": wc})
            print(f"  {w.stem[:44]:44} {wall:7.1f}s  {rows[-1]['xRT']:6.1f}xRT  "
                  f"{len(trows):4} rows  {wc:6} words")

        if patch:
            _tr.WhisperModel = _orig
        audio = sum(x["audio_sec"] for x in rows)
        wall = sum(x["wall_s"] for x in rows)
        per_ct[ct] = {"rows": rows, "audio_sec": audio, "wall_s": wall,
                      "xRT": round(audio / wall, 1) if wall else 0, "actual": str(actual)}
        del tr

    # ---- report ----
    ref_ct = args.compute_types[0]
    lines = [f"# Whisper compute-type A/B ({run_ts})\n",
             f"Sample: {len(pairs)} files, {per_ct[ref_ct]['audio_sec'] / 3600:.2f} h audio, "
             f"RTTMs from `{args.rttm_set}`. Reference = **{ref_ct}**.\n",
             "## Speed\n",
             "| compute type | model reports | wall (s) | xRT | vs ref |",
             "|---|---|--:|--:|--:|"]
    ref_wall = per_ct[ref_ct]["wall_s"]
    for ct in args.compute_types:
        v = per_ct[ct]
        speedup = f"{ref_wall / v['wall_s']:.2f}x" if v["wall_s"] and ct != ref_ct else "-"
        lines.append(f"| {ct} | {v['actual']} | {v['wall_s']:.1f} | {v['xRT']} | {speedup} |")

    lines += ["\n## Per file (wall s)\n",
              "| file | audio min | " + " | ".join(args.compute_types) + " |",
              "|---|--:|" + "|".join("--:" for _ in args.compute_types) + "|"]
    by_stem = {ct: {r["stem"]: r for r in per_ct[ct]["rows"]} for ct in args.compute_types}
    for w, _ in pairs:
        cells = [f"{by_stem[ct][w.stem]['wall_s']:.1f}" for ct in args.compute_types]
        amin = by_stem[ref_ct][w.stem]["audio_sec"] / 60
        lines.append(f"| {w.stem[:44]} | {amin:.1f} | " + " | ".join(cells) + " |")

    lines += ["\n## Transcript divergence vs " + ref_ct + "\n",
              "`wdr` = word-level edits / reference words (row-aligned). "
              "No ground truth - this is drift from float16, not error.\n",
              "| compute type | rows identical | wdr | net word delta |",
              "|---|--:|--:|--:|"]
    diffs = {}
    for ct in args.compute_types:
        if ct == ref_ct:
            lines.append(f"| {ct} | - (reference) | - | - |")
            continue
        agg = {"rows": 0, "rows_changed": 0, "ref_words": 0, "hyp_words": 0, "word_edits_num": 0}
        for w, _ in pairs:
            d = diff_transcripts(read_rows(OUT / f"tr_{ref_ct}" / f"{w.stem}.csv"),
                                 read_rows(OUT / f"tr_{ct}" / f"{w.stem}.csv"))
            diffs[(ct, w.stem)] = d
            agg["rows"] += d["rows"]; agg["rows_changed"] += d["rows_changed"]
            agg["ref_words"] += d["ref_words"]; agg["hyp_words"] += d["hyp_words"]
            agg["word_edits_num"] += d["wdr_pct"] / 100 * d["ref_words"]
        rid = round(100 * (agg["rows"] - agg["rows_changed"]) / agg["rows"], 2) if agg["rows"] else 100
        wdr = round(100 * agg["word_edits_num"] / agg["ref_words"], 2) if agg["ref_words"] else 0
        lines.append(f"| {ct} | {rid}% | {wdr}% | {agg['hyp_words'] - agg['ref_words']:+d} |")

    if len(args.compute_types) > 1:
        lines.append("\n### Per-file divergence\n")
        lines.append("| file | " + " | ".join(f"{ct} rows id% / wdr%" for ct in args.compute_types[1:]) + " |")
        lines.append("|---|" + "|".join("---" for _ in args.compute_types[1:]) + "|")
        for w, _ in pairs:
            cells = []
            for ct in args.compute_types[1:]:
                d = diffs[(ct, w.stem)]
                cells.append(f"{d['rows_identical_pct']}% / {d['wdr_pct']}%")
            lines.append(f"| {w.stem[:40]} | " + " | ".join(cells) + " |")

    (OUT / "compare_compute_type.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "runs_compute_type.jsonl").open("a", encoding="utf-8").write(
        json.dumps({"run_ts": run_ts, "compute_types": args.compute_types,
                    "totals": {ct: {"wall_s": round(per_ct[ct]["wall_s"], 1), "xRT": per_ct[ct]["xRT"]}
                               for ct in args.compute_types}}) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwritten to {OUT / 'compare_compute_type.md'}")


if __name__ == "__main__":
    main()
