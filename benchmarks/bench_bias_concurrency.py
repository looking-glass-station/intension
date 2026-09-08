"""
bias.py Ollama-batch concurrency A/B.

`classify_bias_for_transcript` sends one `ollama.chat()` per batch, serially.
Ollama can serve requests in parallel (`OLLAMA_NUM_PARALLEL`) if the model's KV
slots fit in VRAM. This fires a fixed set of real batches through a
`ThreadPoolExecutor` at several worker counts and reports throughput + whether
the findings are stable (each batch is independent, so they must be).

    uv run python benchmarks/bench_bias_concurrency.py
    uv run python benchmarks/bench_bias_concurrency.py --workers 1 2 3 4 6 --batches 20
    uv run python benchmarks/bench_bias_concurrency.py --transcript "path/to/labeled.csv"

Needs Ollama running with confs/bias.json's model pulled. Writes
benchmarks/transcription/compare_bias_concurrency.md.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import functools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "benchmarks" / "transcription"
print = functools.partial(print, flush=True)  # noqa: A001

DEFAULT_WORKERS = [1, 2, 3, 4]
DEFAULT_BATCHES = 16


def pick_transcript() -> Path:
    for c in (ROOT / "data").glob("*/*/*/transcription_labeled/*.csv"):
        if c.stat().st_size > 80_000:  # something with enough keyword-matching content
            return c
    raise SystemExit("no labeled transcript found; pass --transcript")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", nargs="+", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--batches", type=int, default=DEFAULT_BATCHES, help="how many batches to run each round")
    ap.add_argument("--transcript", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(SRC))
    import ollama
    import file_utils
    from bias import (add_channel_topic, build_system_prompt, classify_batch,
                      load_bias_config, segment_matches_any_topic)
    from logger import global_logger

    # bias.classify_batch calls ollama.chat with no temperature, so the model
    # samples and findings drift run-to-run - that would swamp the concurrency
    # signal. Pin temperature to 0 for the benchmark so a findings diff means a
    # real concurrency effect.
    _orig_chat = ollama.chat

    def _chat_t0(*a, **k):
        k.setdefault("options", {}).setdefault("temperature", 0)
        return _orig_chat(*a, **k)

    ollama.chat = _chat_t0

    cfg = load_bias_config()
    topics = cfg["topics"]
    add_channel_topic(topics)
    system_prompt = build_system_prompt(topics)
    model = cfg.get("model", "gemma3:12b")
    batch_size = cfg.get("batch_size", 10)
    min_dur = cfg.get("min_segment_duration", 5)
    logger = global_logger("bench_bias_concurrency")

    tpath = args.transcript or pick_transcript()
    rows = file_utils.csv_to_dict(tpath) or []
    valid = []
    for r in rows:
        t = r.get("text", "")
        if not isinstance(t, str) or not t.strip():
            continue
        if float(r["end_time"]) - float(r["start_time"]) <= min_dur:
            continue
        if not segment_matches_any_topic(t, topics):
            continue
        valid.append(r)

    batches = [valid[i:i + batch_size] for i in range(0, len(valid), batch_size)]
    batches = batches[: args.batches]
    if len(batches) < 2:
        raise SystemExit(f"only {len(batches)} batches from {tpath.name}; pick a bigger transcript or --batches")

    print(f"transcript: {tpath.name}")
    print(f"{len(valid)} valid segments -> using {len(batches)} batches of {batch_size}  (model {model})\n")

    # warm up (load model, one real call not counted)
    print("warming model...")
    t0 = time.perf_counter()
    classify_batch(system_prompt, batches[0], 0, model, logger)
    print(f"warm call: {time.perf_counter() - t0:.1f}s\n")

    def run_all(workers: int):
        t0 = time.perf_counter()
        out = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(classify_batch, system_prompt, b, i * batch_size, model, logger): i
                    for i, b in enumerate(batches)}
            for f in concurrent.futures.as_completed(futs):
                out[futs[f]] = f.result()
        wall = time.perf_counter() - t0
        # canonical finding key set for stability check
        keys = set()
        for i, finds in out.items():
            for fd in finds:
                keys.add((i, fd.get("segment"), fd.get("topic"), fd.get("target")))
        return wall, keys

    results = []
    ref_keys = None
    for w in args.workers:
        wall, keys = run_all(w)
        if ref_keys is None:
            ref_keys = keys
            stab = "reference"
        else:
            miss = len(ref_keys - keys)
            extra = len(keys - ref_keys)
            stab = "same findings" if (miss == 0 and extra == 0) else f"-{miss}/+{extra} vs ref"
        bpm = len(batches) / wall * 60
        results.append((w, wall, bpm, stab))
        print(f"  workers={w:2}  {wall:6.1f}s  {bpm:5.1f} batches/min  ({stab})")

    base = results[0][1]
    lines = ["# bias.py Ollama batch concurrency\n",
             f"`{tpath.name}` - {len(batches)} batches of {batch_size}, model {model}, "
             f"Ollama {getattr(ollama, '__version__', '?')}.\n",
             "| workers | wall s | batches/min | speedup | findings |",
             "| --: | --: | --: | --: | --- |"]
    for w, wall, bpm, stab in results:
        lines.append(f"| {w} | {wall:.1f} | {bpm:.1f} | {base / wall:.2f}x | {stab} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare_bias_concurrency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwritten to {OUT / 'compare_bias_concurrency.md'}")


if __name__ == "__main__":
    main()
