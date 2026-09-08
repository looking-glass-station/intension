# benchmarks/

Measurement harnesses for the efficiency pass. The **results** live in
[`../benchmark.md`](../benchmark.md); this directory is the tooling.

## What's tracked vs. local

| tracked | local only (gitignored) |
|---|---|
| `bench_*.py` — the harnesses | `diarization/`, `transcription/`, `bias_eval/` — all outputs |
| `../benchmark.md` — the write-up | `*/NOTES.md`, `*/compare_*.md`, `*/sample*.txt`, hypothesis RTTMs, transcripts, findings, per-file result tables |

**Nothing that identifies what the pipeline was run on gets committed** — no
channel names, episode titles, transcript text, or people. Those appear
throughout the raw outputs and the working `NOTES.md` files, so those stay local.
Put durable conclusions in `../benchmark.md`, keeping them subject-neutral
(describe content by *format* — "2-person interview", "reaction stream" — not by
who or what). See also the repo-root `CLAUDE.md`.

## Running them

All harnesses read from `../data/` (also gitignored) and a frozen sample list you
create locally:

- `diarization/sample.txt` — one repo-relative WAV path per line, `#` for
  comments. e.g. `data/<channel>/<source>/<name>/wav/<file>.wav`
- `diarization/sample_stress.txt` — same, for the long-audio cases

Then, from the repo root:

```
uv run python benchmarks/bench_diarize.py --backend pyannote          # diarization speed/quality
uv run python benchmarks/bench_diarize.py --backend nemo-sf-stream-ch
uv run python benchmarks/bench_compare.py --a pyannote --b nemo-sf-stream-ch

uv run python benchmarks/bench_transcribe.py       # does the RTTM source change the transcript
uv run python benchmarks/bench_compute_type.py     # float16 vs int8_float16
uv run python benchmarks/bench_label_speakers.py   # voice-encoder device + thread count

uv run python benchmarks/bench_bias.py             # does diarization word-loss reach bias.py
uv run python benchmarks/bench_bias_concurrency.py # Ollama batch concurrency
uv run python benchmarks/bench_bias_model.py       # gemma3:12b vs smaller models
```

The `nemo-*` diarization backends also need `tools/nemo_diarize/` synced and its
`.nemo` checkpoints in place. The bias harnesses need Ollama running with the
model from `confs/bias.json` pulled.

Each harness prints its table and drops the detail under its output directory;
several also append a `runs_*.jsonl` history row.
