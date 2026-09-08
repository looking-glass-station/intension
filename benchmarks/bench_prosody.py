"""
Validate prosody.py's Tier-2 model, and give the thresholds something to be
tuned against.

Runs Tier 1 over a fixed sample, extracts the top aggression clips plus a random
control set as .wav files, and scores each with the audeering dimensional model
and (for a second opinion) superb/wav2vec2-base-superb-er (IEMOCAP anger class).
Writes a review table you open and listen through: mark each clip
aggressive/not, then adjust confs/prosody.json.

    uv run python benchmarks/bench_prosody.py
    uv run python benchmarks/bench_prosody.py --files "LEFTY HOMOPHOBIA" "FREEZING" --top 12

Clips and the review table quote/contain real audio, so
benchmarks/prosody_eval/ is gitignored. Only the aggregate goes to benchmark.md.
"""
from __future__ import annotations

import argparse
import csv
import functools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
EVAL = ROOT / "benchmarks" / "prosody_eval"
print = functools.partial(print, flush=True)  # noqa: A001


def find_pairs(needles, limit):
    out = []
    for lab in (ROOT / "data").glob("*/*/*/transcription_labeled/*.csv"):
        if needles and not any(n.lower() in lab.stem.lower() for n in needles):
            continue
        wav = lab.parents[1] / "wav" / f"{lab.stem}.wav"
        if wav.exists():
            out.append((lab, wav))
        if len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=3, help="how many transcripts")
    ap.add_argument("--top", type=int, default=10, help="top-N Tier-1 clips per file")
    ap.add_argument("--control", type=int, default=6, help="random control clips per file")
    args = ap.parse_args()

    sys.path.insert(0, str(SRC))
    import numpy as np
    import soundfile as sf
    import librosa
    import pandas as pd
    import prosody
    from logger import global_logger

    cfg = prosody.load_prosody_config()
    logger = global_logger("bench_prosody")
    pairs = find_pairs(args.files, args.limit)
    if not pairs:
        raise SystemExit("no transcript/wav pairs found")

    ser = prosody.build_ser(cfg["ser_model"])
    if ser is None:
        raise SystemExit(f"could not build {cfg['ser_model']}")
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        superb = hf_pipeline("audio-classification", model="superb/wav2vec2-base-superb-er",
                             device=0 if torch.cuda.is_available() else -1)
    except Exception as e:  # noqa: BLE001
        print(f"  (superb second-opinion model unavailable: {e})")
        superb = None

    clips_dir = EVAL / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    review = []

    for lab, wav in pairs:
        df = pd.read_csv(lab)
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        frames = prosody.frame_features(audio, sr)
        rows = []
        for _, seg in df.iterrows():
            seg = seg.to_dict()
            try:
                if float(seg["end_time"]) - float(seg["start_time"]) < cfg["min_segment_duration"]:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            r = prosody.segment_row(seg, frames)
            if r:
                r["text"] = str(seg.get("text", ""))
                rows.append(r)
        if not rows:
            continue
        prosody.add_speaker_zscores(rows)
        for r in rows:
            r["tier1_score"] = prosody.tier1_score(r, cfg["tier1_weights"])

        rows.sort(key=lambda r: -r["tier1_score"])
        picked = rows[: args.top]
        pool = rows[args.top:]
        picked += rng.sample(pool, min(args.control, len(pool)))

        for r in picked:
            a, b = int(r["start_time"] * sr), int(r["end_time"] * sr)
            clip = audio[a:b]
            if clip.size < sr // 2:
                continue
            name = f"{lab.stem[:30]}_{r['start_time']:.0f}s.wav".replace(" ", "_")
            cpath = clips_dir / name
            sf.write(cpath, clip, sr)

            c16 = librosa.resample(clip, orig_sr=sr, target_sr=16000) if sr != 16000 else clip
            adv = prosody.ser_adv(clip, sr, ser)
            t2 = prosody.tier2_from_adv(adv, cfg)
            sup = ""
            if superb is not None:
                try:
                    preds = {p["label"]: p["score"] for p in superb(c16, top_k=4)}
                    sup = round(float(preds.get("ang", 0.0)), 3)
                except Exception:  # noqa: BLE001
                    sup = "err"
            review.append({
                "clip": name, "file": lab.stem[:36], "spk": r["speaker_name"],
                "tier1": round(r["tier1_score"], 2),
                "arousal": round(adv["arousal"], 3), "dominance": round(adv["dominance"], 3),
                "valence": round(adv["valence"], 3), "tier2": t2, "superb_ang": sup,
                "in_top": r in rows[: args.top],
                "text": r["text"][:200],
            })
        print(f"  {lab.stem[:44]:44} {len(picked)} clips")

    review.sort(key=lambda x: -x["tier2"])
    fields = ["clip", "file", "spk", "tier1", "arousal", "dominance", "valence",
              "tier2", "superb_ang", "in_top", "text"]
    with (EVAL / "clips_review.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["is_aggressive_YN"])
        w.writeheader()
        for r in review:
            w.writerow({**r, "is_aggressive_YN": ""})

    md = ["# Prosody Tier-2 review\n",
          f"{len(review)} clips in `prosody_eval/clips/`. Listen, fill `is_aggressive_YN` "
          "in clips_review.csv, then compare against tier2 / superb_ang to set "
          "`confs/prosody.json` thresholds.\n",
          "| clip | spk | tier1 | arousal | dom | val | tier2 | superb_ang | top? |",
          "|---|---|--:|--:|--:|--:|--:|--:|:-:|"]
    for r in review:
        md.append(f"| {r['clip']} | {r['spk']} | {r['tier1']} | {r['arousal']} | "
                  f"{r['dominance']} | {r['valence']} | {r['tier2']} | {r['superb_ang']} | "
                  f"{'Y' if r['in_top'] else ''} |")
    (EVAL / "clips_review.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n{len(review)} clips -> {clips_dir}")
    print(f"review: {EVAL / 'clips_review.md'} / .csv")


if __name__ == "__main__":
    main()
