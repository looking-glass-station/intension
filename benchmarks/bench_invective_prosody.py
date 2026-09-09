"""
How does prosody.py's aggression signal land in invective.py output?

After the audio refactor, invective drops its own emotion model and takes the
tone signal from prosody's per-segment aggression score. That score is *recorded*
per occurrence (`prosody_aggression` column) and used only to (a) keep an
identity term said with venom out of the NEUTRAL bucket and (b) confirm a
likely-invective slur into HIGH - it can't create a finding on its own.

This runs prosody + the new invective over a sample and reports: how many
findings are text-driven, the prosody_aggression distribution on the occurrences
invective looked at, and any occurrence where the two signals strongly diverge
(loud delivery on a keyword that is NOT invective, or a calm slur).

    uv run python benchmarks/bench_invective_prosody.py --limit 12

Writes benchmarks/prosody_eval/invective_prosody.md (+ .csv, local - quotes text).
"""
from __future__ import annotations

import argparse
import csv
import functools
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
EVAL = ROOT / "benchmarks" / "prosody_eval"
print = functools.partial(print, flush=True)  # noqa: A001


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    sys.path.insert(0, str(SRC))
    import audio_affect
    import prosody
    import invective
    from logger import global_logger

    logger = global_logger("bench_invective_prosody")
    inv_cfg = invective.load_invective_config(ROOT / "confs" / "invectives.json")
    pcfg = prosody.load_prosody_config()

    pairs = []
    for lab in (ROOT / "data").glob("*/*/*/transcription_labeled/*.csv"):
        wav = lab.parents[1] / "wav" / f"{lab.stem}.wav"
        if wav.exists():
            pairs.append((lab, wav))
        if len(pairs) >= args.limit:
            break
    if not pairs:
        raise SystemExit("no transcript/wav pairs")

    ser = audio_affect.build_ser(pcfg["ser_model"])
    if ser is None:
        raise SystemExit("SER model unavailable")
    dtx, card = invective.build_models()
    rng = random.Random(0)

    n_find = n_text_driven = 0
    aggs_on_occ: list[float] = []
    diverge = []
    for lab, wav in pairs:
        pros_csv = lab.parents[1] / "prosody" / lab.name
        pros_aud = lab.parents[1] / "prosody_audacity" / f"{lab.stem}.txt"
        pros_csv.parent.mkdir(parents=True, exist_ok=True)
        pros_aud.parent.mkdir(parents=True, exist_ok=True)
        prosody.process_transcript(lab, wav, pros_csv, pros_aud, pcfg, ser, logger, rng)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / lab.name
            invective.process_transcript(lab, wav, out, Path(td) / "clips",
                                         inv_cfg["bad_terms"], inv_cfg["good_terms"],
                                         inv_cfg["term_to_group"], dtx, card, min_score=0.5)
            rows = list(csv.DictReader(out.open(encoding="utf-8"))) if out.exists() else []

        for r in rows:
            n_find += 1
            ia, tox = float(r["detoxify_identity_attack"]), float(r["detoxify_toxicity"])
            lk = r["rule_bucket"] == "likely_invective"
            if ia >= 0.7 or tox >= 0.7 or (lk and (ia >= 0.6 or tox >= 0.6)):
                n_text_driven += 1
            pa = r.get("prosody_aggression")
            if pa not in (None, ""):
                pa = float(pa)
                aggs_on_occ.append(pa)
                # loud delivery on a keyword that invective did NOT call invective
                if pa >= 0.75 and r["final_label"].startswith(("NON_", "PRAISE")):
                    diverge.append((lab.stem, r["timestamp_start"], "loud, not invective",
                                    r["final_label"], pa, (r.get("sentence") or "")[:130]))
                # a slur invective flagged HIGH/MEDIUM but delivered calmly
                if pa <= 0.35 and r["final_label"].startswith("INVECTIVE_"):
                    diverge.append((lab.stem, r["timestamp_start"], "calm slur",
                                    r["final_label"], pa, (r.get("sentence") or "")[:130]))
        print(f"  {lab.stem[:44]:44} {len(rows)} findings")

    aggs_on_occ.sort()
    q = lambda p: aggs_on_occ[int(len(aggs_on_occ) * p)] if aggs_on_occ else 0.0  # noqa: E731
    lines = ["# invective x prosody signal\n",
             f"{len(pairs)} transcripts, {n_find} invective findings, "
             f"{n_text_driven}/{n_find} text-driven (the rest are NEUTRAL-block / HIGH-boost only).\n",
             f"prosody_aggression on the {len(aggs_on_occ)} scored occurrences: "
             f"p25 {q(0.25):.2f} / p50 {q(0.5):.2f} / p75 {q(0.75):.2f} / max "
             f"{aggs_on_occ[-1] if aggs_on_occ else 0:.2f}\n",
             "## strong divergence (delivery vs. words)\n",
             "| file | t | kind | invective label | aggression |",
             "|---|--:|---|---|--:|"]
    for stem, t, kind, lab_, pa, _ in diverge:
        lines.append(f"| {stem[:30]} | {t} | {kind} | {lab_} | {pa:.2f} |")
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "invective_prosody.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (EVAL / "invective_prosody_diverge.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "t", "kind", "invective_label", "aggression", "sentence"])
        w.writerows(diverge)
    print("\n" + "\n".join(lines[:6]))
    print(f"\ndetail: {EVAL / 'invective_prosody.md'}")


if __name__ == "__main__":
    main()
