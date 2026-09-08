# CLAUDE.md

Config-driven pipeline: download (YouTube / Twitch / Patreon) → diarize →
transcribe → match host voices → classify bias / invective / (planned) prosody.

## Keep the public repo subject-neutral

**Never commit anything that identifies what this pipeline has been run on, or
any personal information** — channel names, creator/person names, episode titles,
verbatim transcript text, classification findings, or file lists that contain any
of those.

- `data/`, `confs/download_configurations/*` (except `pbs.json`), `logs/`,
  `benchmarks/{diarization,transcription,bias_eval}/`, and `TODO.md` are
  gitignored for this reason. Keep it that way.
- Benchmark write-ups: put durable conclusions in `benchmark.md`, described by
  *format* ("2-person interview", "reaction stream", "large multi-party call") —
  never by who or what. The raw `benchmarks/*/NOTES.md` and `compare_*.md` name
  things freely and stay local.
- Before any commit, scan the diff for names/titles/quotes. If unsure, leave it
  out.

## Environment

- `uv` manages the main env (`uv run python ...`, `uv sync`). Python 3.11.
- `tools/nemo_diarize/` is an isolated `uv` env (NeMo pins conflicting
  torch/numpy) invoked as a subprocess for the Sortformer diarization backend.
- `bias.py` classifies via local Ollama (`gemma3:12b`), not an API.
- Windows box; `git` is at `C:\Program Files\Git\cmd`. PowerShell has no
  heredocs — use `git commit -F <file>` or `@'...'@` here-strings.

## Working notes

- The efficiency-pass backlog and its results are in `TODO.md` (local) and
  `benchmark.md` (committed).
- Benchmark before/after on pipeline changes; for quality-affecting changes, diff
  the actual output too. See `benchmarks/README.md`.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
