"""
Right-size the bias-classification model.

`bias.py` runs `gemma3:12b`. The keyword pre-filter already narrows what the LLM
sees, so the remaining job is closer to structured classification than
open-ended reasoning - a smaller model may do it as well, faster, and leave VRAM
for `OLLAMA_NUM_PARALLEL>1`.

Runs the project's `classify_bias_for_transcript` (temperature 0) over a fixed
set of labeled transcripts with each candidate model, and reports per model:
speed, findings volume, JSON-parse reliability, and agreement with the reference
model (gemma3:12b) - a finding matches if it's the same segment + topic.

    uv run python benchmarks/bench_bias_model.py
    uv run python benchmarks/bench_bias_model.py --models gemma3:12b qwen2.5:7b llama3.1:8b
    uv run python benchmarks/bench_bias_model.py --max-batches 10 --transcripts "FREEZING" "#361"

Per-model finding CSVs and the disagreement detail (they quote real transcripts)
go under benchmarks/bias_eval/ and are gitignored. The aggregate table is printed
and written to benchmarks/bias_eval/compare_bias_model.md.
"""
from __future__ import annotations

import argparse
import csv
import functools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
EVAL = ROOT / "benchmarks" / "bias_eval"
print = functools.partial(print, flush=True)  # noqa: A001

DEFAULT_MODELS = ["gemma3:12b", "qwen2.5:7b", "llama3.1:8b"]
# frozen sample: labeled transcripts spanning interview / debate / reaction stream
DEFAULT_TRANSCRIPTS = [
    "data/hasan/youtube/HasanAbi/transcription_labeled/-20_ FREEZING PROTEST ALMOST BROKE US.csv",
    "data/lex_fridman/youtube/lexfridman/transcription_labeled/Aaron Smith-Levin Scientology  Lex Fridman Podcast #361.csv",
    "data/bretweinstein/youtube/DarkHorsePod/transcription_labeled/100 Years of Wisdom B17 Pilot on the DarkHorse Podcast.csv",
]


def slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def load_findings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def key_of(f: dict) -> tuple:
    # segment identity = rounded start time; plus topic
    return (round(float(f["start_time"]) / 5) * 5, (f.get("topic") or "").strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--transcripts", nargs="+", default=None,
                    help="substrings matched against DEFAULT_TRANSCRIPTS, or full paths")
    ap.add_argument("--max-batches", type=int, default=0, help="cap batches per transcript (0 = all)")
    args = ap.parse_args()

    # resolve transcripts
    if args.transcripts:
        picked = []
        for t in args.transcripts:
            p = Path(t)
            if p.exists():
                picked.append(p)
                continue
            hit = next((d for d in DEFAULT_TRANSCRIPTS if t.lower() in d.lower()), None)
            picked.append(ROOT / hit) if hit else print(f"  !! no transcript matches {t!r}")
        tpaths = [p for p in picked if p and p.exists()]
    else:
        tpaths = [ROOT / d for d in DEFAULT_TRANSCRIPTS if (ROOT / d).exists()]
    if not tpaths:
        raise SystemExit("no transcripts found")

    sys.path.insert(0, str(SRC))
    import ollama
    from bias import (add_channel_topic, build_system_prompt, classify_bias_for_transcript,
                      load_bias_config, segment_matches_any_topic)
    from logger import global_logger
    import file_utils

    cfg = load_bias_config()
    topics = cfg["topics"]
    add_channel_topic(topics)
    system_prompt = build_system_prompt(topics)
    batch_size = cfg.get("batch_size", 10)
    min_dur = cfg.get("min_segment_duration", 5)
    logger = global_logger("bench_bias_model")

    # optional batch cap: monkeypatch classify_bias_for_transcript's view by trimming rows upstream
    EVAL.mkdir(parents=True, exist_ok=True)

    _listed = ollama.list()
    _models = _listed.get("models", []) if isinstance(_listed, dict) else getattr(_listed, "models", [])
    have = {(m.get("model") if isinstance(m, dict) else getattr(m, "model", getattr(m, "name", "")))
            for m in _models}
    models = []
    for m in args.models:
        if m in have or any(h.startswith(m) for h in have):
            models.append(m)
        else:
            print(f"  !! {m} not pulled, skipping (ollama pull {m})")
    if not models:
        raise SystemExit("no requested models available")

    # segment counts per transcript (for context in the report)
    tinfo = {}
    for tp in tpaths:
        rows = file_utils.csv_to_dict(tp) or []
        valid = [r for r in rows
                 if isinstance(r.get("text"), str) and r["text"].strip()
                 and float(r["end_time"]) - float(r["start_time"]) > min_dur
                 and segment_matches_any_topic(r["text"], topics)]
        n_batches = (len(valid) + batch_size - 1) // batch_size
        if args.max_batches:
            n_batches = min(n_batches, args.max_batches)
        tinfo[tp] = (len(valid), n_batches)

    results: dict[str, dict] = {}
    for model in models:
        print(f"\n=== {model} ===")
        try:  # load the model so its load time doesn't skew the first transcript
            ollama.chat(model=model, messages=[{"role": "user", "content": "hi"}],
                        options={"temperature": 0})
        except Exception as e:  # noqa: BLE001
            print(f"  !! warm-up failed: {e}")
        results[model] = {"wall": 0.0, "batches": 0, "findings": 0, "per_t": {}}
        mdir = EVAL / slug(model)
        mdir.mkdir(exist_ok=True)
        for tp in tpaths:
            n_valid, n_batches = tinfo[tp]
            src = tp
            if args.max_batches:  # write a trimmed copy so the harness only runs N batches
                trimmed = EVAL / f"_trim_{tp.stem}.csv"
                rows = file_utils.csv_to_dict(tp) or []
                # keep enough rows that N batches of keyword-matching segments survive
                keep, seen = [], 0
                for r in rows:
                    keep.append(r)
                    if (isinstance(r.get("text"), str) and r["text"].strip()
                            and float(r["end_time"]) - float(r["start_time"]) > min_dur
                            and segment_matches_any_topic(r["text"], topics)):
                        seen += 1
                    if seen >= args.max_batches * batch_size:
                        break
                file_utils.dict_to_csv(trimmed, keep)
                src = trimmed
            bcsv = mdir / f"{tp.stem}.csv"
            baud = mdir / f"{tp.stem}.audacity.txt"
            for f in (bcsv, baud):
                f.unlink(missing_ok=True)
            t0 = time.perf_counter()
            classify_bias_for_transcript(src, bcsv, baud, topics, system_prompt,
                                         model, batch_size, min_dur, logger)
            wall = time.perf_counter() - t0
            finds = load_findings(bcsv)
            results[model]["wall"] += wall
            results[model]["batches"] += n_batches
            results[model]["findings"] += len(finds)
            results[model]["per_t"][tp] = {"wall": wall, "n_batches": n_batches, "finds": finds}
            print(f"  {tp.stem[:44]:44} {n_batches:3} batches  {wall:6.1f}s  {len(finds):3} findings")

    # ---- agreement vs reference (first model) ----
    ref = models[0]
    lines = [f"# Bias model comparison ({ref} = reference)\n",
             f"{len(tpaths)} transcripts, batch_size {batch_size}, "
             f"{sum(b for _, b in tinfo.values())} batches total, temperature 0.\n",
             "| model | wall s | batches/min | findings | vs gemma: shared / gemma-only / model-only |",
             "| --- | --: | --: | --: | --- |"]
    ref_keys_all = {tp: {key_of(f) for f in results[ref]["per_t"][tp]["finds"]} for tp in tpaths}
    disagree = [f"# Bias model disagreements vs {ref}\n"]
    for model in models:
        r = results[model]
        bpm = r["batches"] / r["wall"] * 60 if r["wall"] else 0
        shared = gonly = monly = 0
        for tp in tpaths:
            mkeys = {key_of(f) for f in r["per_t"][tp]["finds"]}
            rk = ref_keys_all[tp]
            shared += len(mkeys & rk)
            gonly += len(rk - mkeys)
            monly += len(mkeys - rk)
        cmp = "reference" if model == ref else f"{shared} / {gonly} / {monly}"
        lines.append(f"| {model} | {r['wall']:.0f} | {bpm:.1f} | {r['findings']} | {cmp} |")

        if model != ref:
            disagree.append(f"\n## {model}\n")
            for tp in tpaths:
                mfinds = {key_of(f): f for f in r["per_t"][tp]["finds"]}
                gfinds = {key_of(f): f for f in results[ref]["per_t"][tp]["finds"]}
                go = [gfinds[k] for k in gfinds.keys() - mfinds.keys()]
                mo = [mfinds[k] for k in mfinds.keys() - gfinds.keys()]
                if not go and not mo:
                    continue
                disagree.append(f"\n### {tp.stem}\n")
                for tag, fs in ((f"{ref} only", go), (f"{model} only", mo)):
                    for f in fs[:15]:
                        disagree.append(
                            f"- _{tag}_ [{float(f['start_time']):.0f}s {f.get('topic')}/{f.get('stance')} "
                            f"score {f.get('score')}] target={f.get('target')!r} :: "
                            f"{(f.get('text') or '')[:160]}")

    (EVAL / "compare_bias_model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EVAL / "disagreements.md").write_text("\n".join(disagree) + "\n", encoding="utf-8")
    for f in EVAL.glob("_trim_*.csv"):
        f.unlink()
    print("\n" + "\n".join(lines))
    print(f"\naggregate -> {EVAL / 'compare_bias_model.md'}")
    print(f"disagreement detail (local) -> {EVAL / 'disagreements.md'}")


if __name__ == "__main__":
    main()
