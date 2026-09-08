"""
label_speakers.py thread-count A/B.

`label_speakers.main()` runs its per-file `ThreadPoolExecutor` at
`max_workers_base * 1` = 2 threads (`confs/global.json`), despite many cores.
Labeling one file is: full-file `sf.read` + a per-segment loop of
`VoiceEncoder.embed_utterance` (CPU mel-spectrogram + a tiny GPU LSTM) + host
cosine matching - mostly GIL-releasing CPU work, so more threads *should* help.

This times the stage over a fixed sample at several worker counts, checks the
labeled output is identical regardless of worker count, and breaks down where a
single file's time goes.

    uv run python benchmarks/bench_label_speakers.py
    uv run python benchmarks/bench_label_speakers.py --workers 1 2 4 8 16 --encoder-device cpu
    uv run python benchmarks/bench_label_speakers.py --per-thread-encoder

Writes nothing under data/. Scratch output goes to a temp dir; a short report to
benchmarks/transcription/compare_label_speakers.md.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import functools
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DIAR = ROOT / "benchmarks" / "diarization"
OUT = ROOT / "benchmarks" / "transcription"
print = functools.partial(print, flush=True)  # noqa: A001

DEFAULT_WORKERS = [2, 4, 8, 16]


def sample_stems() -> list[str]:
    out = []
    for line in (DIAR / "sample.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(Path(line).stem)
    return out


def gather_jobs(stems: list[str]):
    """(wav, transcript, [HostEmbedding], channel) for every sample stem that has
    a transcript, a wav, and a sibling training/ dir with a host embedding.

    Scans data/ directly rather than via configs - the download configs get
    pruned for privacy, but the data dirs (and this frozen sample) outlive them.
    """
    from configs import _build_host_embeddings

    want = set(stems)
    jobs = []
    for tcsv in (ROOT / "data").glob("**/transcription/*.csv"):
        if tcsv.stem not in want:
            continue
        base = tcsv.parents[1]  # <channel>/<source>/<name>
        wav = base / "wav" / f"{tcsv.stem}.wav"
        train = base / "training"
        if not wav.exists() or not train.exists():
            continue
        # host count = number of embeddings_speaker_*.npy present
        n_hosts = len(list(train.glob("embeddings_speaker_*.npy")))
        emb = [e for e in _build_host_embeddings(base, [""] * n_hosts)
               if Path(e.embeddings_file).exists()]
        if not emb:
            continue
        try:
            channel = str(base.relative_to(ROOT / "data").parts[0]) + "/" + base.name
        except ValueError:
            channel = base.name
        jobs.append((wav, tcsv, emb, channel))
    return jobs


def profile_one(job, encoder) -> dict:
    """Time the phases of a single label_speakers-style pass (no file writes)."""
    import numpy as np
    import soundfile as sf

    wav, tcsv, emb, _ = job
    t = {}
    t0 = time.perf_counter()
    audio, sr = sf.read(wav)
    t["sf_read_s"] = time.perf_counter() - t0

    with tcsv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    t0 = time.perf_counter()
    n_embed = 0
    sums: dict[str, np.ndarray] = {}
    durs: dict[str, float] = {}
    for r in rows:
        try:
            dur = float(r["duration"]); start = float(r["start_time"])
        except (KeyError, ValueError):
            continue
        if dur <= 0.05 or dur > 3600:
            continue
        a, b = int(start * sr), int(start * sr) + int(dur * sr)
        if b <= a or a < 0 or b > len(audio):
            continue
        e = encoder.embed_utterance(audio[a:b])
        sums[r["speaker"]] = sums.get(r["speaker"], np.zeros_like(e)) + e * dur
        durs[r["speaker"]] = durs.get(r["speaker"], 0.0) + dur
        n_embed += 1
    t["embed_loop_s"] = time.perf_counter() - t0
    t["n_rows"] = len(rows)
    t["n_embed"] = n_embed
    t["audio_min"] = round(len(audio) / sr / 60, 1)
    return t


def read_labels(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return {(r.get("start_time"), r.get("speaker")): r.get("speaker_name")
                for r in csv.DictReader(fh)}


def run_at(workers: int, jobs, *, per_thread_encoder: bool, encoder_device, shared_encoder):
    from label_speakers import label_speakers
    from resemblyzer import VoiceEncoder

    scratch = Path(tempfile.mkdtemp(prefix=f"bench_label_{workers}_"))
    _local = {}

    def one(job):
        wav, tcsv, emb, _ = job
        if per_thread_encoder:
            import threading
            key = threading.get_ident()
            enc = _local.get(key)
            if enc is None:
                enc = _local[key] = VoiceEncoder(device=encoder_device, verbose=False)
        else:
            enc = shared_encoder
        out_csv = scratch / f"{tcsv.stem}.csv"
        out_aud = scratch / f"{tcsv.stem}.txt"
        label_speakers(wav, tcsv, out_csv, out_aud, enc, emb)
        return out_csv

    t0 = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for f in concurrent.futures.as_completed(futs):
            j = futs[f]
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"  !! {j[1].stem}: {type(e).__name__}: {e}")
    wall = time.perf_counter() - t0
    labels = {p.stem: read_labels(p) for p in results if p.exists()}
    shutil.rmtree(scratch, ignore_errors=True)
    return wall, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", nargs="+", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--encoder-device", default=None, help="cuda | cpu (default: resemblyzer's choice)")
    ap.add_argument("--per-thread-encoder", action="store_true",
                    help="one VoiceEncoder per thread instead of a shared instance")
    ap.add_argument("--files", nargs="+", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(SRC))
    from resemblyzer import VoiceEncoder

    stems = sample_stems()
    if args.files:
        stems = [s for s in stems if any(n.lower() in s.lower() for n in args.files)]
    jobs = gather_jobs(stems)
    if not jobs:
        raise SystemExit("no labelable sample files (need trained host embeddings + transcripts)")
    print(f"{len(jobs)} files: " + ", ".join(f"{j[1].stem[:28]} ({j[3]})" for j in jobs) + "\n")

    shared = None if args.per_thread_encoder else VoiceEncoder(device=args.encoder_device, verbose=True)

    # one-file phase breakdown (serial, shared encoder)
    enc_for_profile = shared or VoiceEncoder(device=args.encoder_device, verbose=False)
    print("phase breakdown (serial):")
    for j in jobs:
        p = profile_one(j, enc_for_profile)
        print(f"  {j[1].stem[:34]:34} {p['audio_min']:6}min  {p['n_embed']:4} embeds  "
              f"sf_read {p['sf_read_s']:5.1f}s  embed_loop {p['embed_loop_s']:6.1f}s")
    print()

    rows = []
    ref_labels = None
    for w in args.workers:
        wall, labels = run_at(w, jobs, per_thread_encoder=args.per_thread_encoder,
                              encoder_device=args.encoder_device, shared_encoder=shared)
        mismatch = ""
        if ref_labels is None:
            ref_labels = labels
        else:
            diffs = sum(1 for stem in ref_labels for k, v in ref_labels[stem].items()
                        if labels.get(stem, {}).get(k) != v)
            mismatch = "ok" if diffs == 0 else f"{diffs} label diffs vs ref!"
        rows.append((w, wall, mismatch))
        print(f"  workers={w:2}  {wall:7.1f}s  {mismatch}")

    base = rows[0][1]
    lines = ["# label_speakers thread-count A/B\n",
             f"{len(jobs)} files, encoder device = "
             f"{(shared or enc_for_profile).device.type}, "
             f"{'per-thread' if args.per_thread_encoder else 'shared'} encoder.\n",
             "| workers | wall s | speedup | output |",
             "|--:|--:|--:|---|"]
    for w, wall, mm in rows:
        lines.append(f"| {w} | {wall:.1f} | {base / wall:.2f}x | {mm or 'reference'} |")
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "compare_label_speakers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwritten to {OUT / 'compare_label_speakers.md'}")


if __name__ == "__main__":
    main()
