# Pipeline benchmarks

Host-neutral summary of the efficiency-pass measurements — this file is the
public summary; the harnesses live in `benchmarks/` and regenerate the detailed
per-file tables locally (some hold verbatim transcript text and stay untracked).

Two reports so far: **diarization backend** (pyannote → streaming Sortformer) and
**transcription compute type** (float16 → int8_float16).

---

## Diarization backend

Why the pipeline diarizes with **streaming Sortformer** (NeMo) by default instead
of pyannote, and what the trade-off is.

### TL;DR

| | pyannote 3.1 (old default) | streaming Sortformer + CallHome post-proc (new default) |
|---|---|---|
| Speed (diarization) | ~77× realtime | **~370× realtime** |
| End-to-end on a 2-hour interview | baseline | **~6× faster** |
| Speakers on a clean 2-person interview | 5–12 labels (2 real, rest are music/clip fragments) | **2** |
| Segment count | choppy (sub-second median on busy audio) | **40–55 % fewer**, cleaner turns |
| Speaker tracking over 2 h+ | stable | stable (no label drift) |
| Hard limit | none | **max 4 speakers** — collapses anything above that |
| Transcript content (interview) | baseline | ~99 % of the words, fewer edge-clip artifacts |
| Transcript content (busy multi-speaker stream) | baseline | ~18 % fewer words (mostly crosstalk / soundbite fragments) |

**Decision:** streaming Sortformer everywhere by default. pyannote stays one
config flag away for the rare show that genuinely needs more than four
simultaneous speakers separated.

### What was compared

- **pyannote** — `pyannote/speaker-diarization-3.1` (via whisperx), the previous
  default. Clustering-based, no speaker limit.
- **Sortformer, offline** — NeMo `diar_sortformer_4spk-v1`. End-to-end neural, ≤4
  speakers.
- **Sortformer, streaming** — NeMo `diar_streaming_sortformer_4spk-v2.1`.
  End-to-end neural with streaming state, ≤4 speakers.
- **NeMo ClusteringDiarizer** — `general` and `meeting` presets (VAD + speaker
  embeddings + spectral clustering). No speaker limit.

The Sortformer / clustering backends run in an isolated environment
(`tools/nemo_diarize/`, NeMo pins conflicting torch/numpy versions) and are
invoked as a subprocess. See NOTES.md for setup.

### Test set

8 files, **~12 hours** of audio, 16 kHz mono, drawn from what the pipeline
actually has on disk (so a mix of lossy and lossless codecs). Chosen to span the
formats the project ingests:

| profile | length | notes |
|---|---|---|
| clean 2-person interview | 2 h | two well-separated voices |
| clean 2-person interview | 4 h | stress-test speaker tracking over long audio |
| small-panel debate | 30 min | 3–4 voices, some overlap |
| debate with inserted clips | 70 min | 2 main voices + short third-party clips |
| streamed debate | 55 min | 2–3 voices over music / silence |
| large multi-party call | 95 min | ~15–20 participants, chaotic overlap |
| solo commentary + reaction clips | 25 min | one main voice, many short inserts |
| solo commentary + reaction clips | 90 min | one main voice, many short inserts |

A separate stress set of three **7.5–10 hour** solo-commentary streams was used
for the long-audio findings below.

There is **no ground-truth diarization** for this audio, so "accuracy" here means
manual spot-checks plus agreement/disagreement between backends, not a DER score
against a reference.

### Results

#### Speed

Over the full ~12 h sample, one file at a time, on a single consumer GPU:

| backend | throughput | wall time | verdict |
|---|--:|--:|---|
| Sortformer, streaming | ~370× realtime | ~2 min | fast |
| pyannote 3.1 | ~77× realtime | ~10 min | fine |
| NeMo clustering (general) | ~13× realtime | ~58 min | too slow |
| NeMo clustering (meeting) | ~11× realtime | ~69 min | too slow |
| Sortformer, offline | — | — | **unusable** — processes the whole file in one attention pass; tries to allocate hundreds of GB on anything longer than ~90 s |

Speed alone doesn't decide it — pyannote is already fast enough. Streaming
Sortformer only wins if it's also *cleaner*.

#### Speaker counting

pyannote's speaker count is deliberately uncapped, and on real content that means
every bit of non-primary audio — intro music, a phone caller, an inserted clip —
becomes its own speaker label. On the clean 2-person interviews it reported 5 and
12 speakers; the true pair is recoverable (two labels hold >99 % of the speech)
but only after post-processing.

Streaming Sortformer tracks a fixed roster of up to 4 speakers. On the interviews
and debates that lands on the right number. On the busy content it caps at 4 and
lumps the long tail together.

| profile | real | pyannote (labels / dominant) | streaming Sortformer |
|---|---|--:|--:|
| 2-person interview | 2 | 5 / 3 | **2** |
| 2-person interview (long) | 2 | 12 / 2 | **2** |
| small-panel debate | ~4 | 4 / 4 | 4 |
| debate + clips | 2 | 4 / 2 | 3 |
| streamed debate | 2–3 | 7 / 2 | 4 |
| large multi-party call | ~15–20 | 17 / 8 | 4 |
| solo + clips (short) | 1 | 9 / 3 | 4 |
| solo + clips (long) | 1 | 25 / 16 | 4 |

Neither is "correct" on the bottom three rows — 17 labels and 4 labels are both
wrong for a 15-person call. For this pipeline's downstream steps (transcript,
topic and stance tagging, matching the host voice) a small stable roster is more
useful than dozens of one-off fragments.

#### Segmentation quality

pyannote's raw output is choppy — sub-second median segment length on the busier
files, with words frequently clipped at segment boundaries. Streaming Sortformer
with conversational post-processing produces **40–55 % fewer segments** covering
the same speech, with turn boundaries that line up with actual pauses.

#### Long audio

- pyannote handled 10-hour files (~53× realtime) but returned **55–135 speaker
  labels** — every clip and caller its own label.
- Streaming Sortformer initially **crashed above ~5.5 h** (a whole-signal GPU
  operation in its front-end). Fixed by running the audio front-end on CPU above
  that threshold; a 7.5 h file then diarizes in ~2 min. Output stays at 4
  speakers.

#### Stability over time

On a 2-hour+ interview, the streaming model's per-5-minute speaker assignment
matches pyannote's window by window — no label swaps, no identity drift. Its
arrival-order speaker cache holds identities across the whole file.

### Tuning

The streaming model ships with a low-latency real-time preset and no
post-processing. For offline batch use both are wrong. Two changes:

1. **High-latency preset** (large look-ahead buffer). Latency is irrelevant for
   batch; this is both faster and slightly cleaner.
2. **Conversational post-processing** (onset/offset/minimum-duration parameters
   tuned for 2–6 speaker conversational audio). Drops sub-0.5 s fragments, merges
   neighbouring same-speaker segments.

Post-processing only cleans boundaries and removes fragments — it never
reassigns a speaker (speaker-confusion between the tuned and untuned runs is
zero). The tuned config is the only one that gets both clean interviews down to
exactly 2 speakers.

### Downstream impact

The point of diarization here is to feed transcription and classification, so the
comparison that matters is what those steps produce from each backend's segments.

#### Transcription impact

One ASR call per segment. Fewer, cleaner segments →

- **~1.5× faster transcription** (fewer calls), on top of the ~5× faster
  diarization → **~6× faster end-to-end** on interview content.
- **Interviews: effectively the same transcript** — ~99 % of the words, with
  *fewer* artifacts (pyannote's long segments clipped words at the edges).
- **Busy multi-speaker streams: ~18 % fewer words.** The 4-speaker cap can't
  place overlapping clip/crosstalk audio, and the tighter voice-activity
  detection drops more. Spot-checking, the dropped material is a mix of genuine
  speech and garbled 1–2 s soundbite fragments — its value is genuinely
  ambiguous.

#### Classification impact (topic / stance tagging)

The classifier only sees segments that pass a keyword pre-filter and a minimum
duration. Running it over both transcripts of an interview and a busy stream:

| profile | backend | segments after keyword filter | findings |
|---|---|--:|--:|
| interview | pyannote | 17 | 2 |
| interview | streaming Sortformer | 17 | 2 |
| busy stream | pyannote | 22 | 2 |
| busy stream | streaming Sortformer | 20 | 2 |

The transcription word-loss on streams **barely touches the classifier** — it
loses 2 candidate segments out of 22 on the stream, none on the interview, and
the finding count is unchanged. The words streaming Sortformer drops are
overwhelmingly non-substantive chatter. On the stream the *specific* findings
differ between the two, but that's the language model reacting to
differently-chunked context, not one backend systematically missing content;
neither set is clearly better and the sample is small.

### The catch: 4-speaker ceiling

Streaming Sortformer separates **at most 4 speakers**. This is structural — the
model's published error rate rises sharply past 4. For content with more than
four genuinely distinct, individually-relevant speakers (large panels, group
calls) it will merge some of them.

For this project that's an acceptable default because:

- The common formats (interview, 1–2 host debate, solo commentary) are all ≤4.
- On >4-speaker content *neither* backend is right, and the downstream steps
  degrade gracefully — the host stays the dominant speaker, and non-primary
  audio is what the host-matching step already flags as unknown.
- The rare show that needs it can opt back into pyannote per-config.

### How to choose the backend

Default is set in `confs/global.json`:

```json
{ "diarization_backend": "sortformer" }
```

Override per channel in its download config (`confs/download_configurations/*.json`):

```json
{ "diarization_backend": "pyannote" }
```

Accepted values: `"sortformer"` (streaming Sortformer + conversational
post-processing) and `"pyannote"`. Missing / unset falls back to the global
default. If the NeMo environment or model files aren't present, the pipeline logs
a warning and uses pyannote.

### Rejected options

- **Sortformer offline** — cannot process anything longer than a short clip.
- **NeMo clustering** — 5–7× slower than pyannote *and* no better on speaker
  counting. No reason to use it over pyannote.
- **Per-channel backend routing based on expected speaker count** — the project
  ingests unknown new channels, so the "expected" count isn't known ahead of
  time. A manual per-config override covers the real need.
- **A voice-activity tripwire that re-runs pyannote when Sortformer looks like
  it's under-attributing speech** — held in reserve; the classification results
  suggest it isn't needed.

---

## Transcription: Whisper compute type

CTranslate2 (faster-whisper's backend) can run the model weights at different
precisions. The pipeline was on `float16`; `int8_float16` was only a crash-retry
fallback. Question: make `int8_float16` the default?

Model: `Systran/faster-distil-whisper-large-v3`, batched, on the same 12 h sample,
transcribing the streaming-Sortformer segments. `float16` is the reference — there
is no ground truth, so "drift" below is distance *from float16*, not error.

### Result

| compute type | wall (12 h audio) | throughput | vs float16 |
| --- | --: | --: | --: |
| float16 | 679 s | ~65× realtime | — |
| **int8_float16** | **633 s** | **~70× realtime** | **1.07× faster** |

~7 % faster, consistently across every file, and lower VRAM.

**Transcript drift: ~2.5 % of words**, and spot-checking every changed row it is
**noise, not content loss**:

- alternate spellings of proper nouns and jargon that are *already* ASR guesses
  in both versions (the model invents a spelling either way — neither is "right");
- punctuation and sentence-boundary differences;
- disfluency rendering — `int8_float16` often captures *more* of a stammer;
- unintelligible crowd/chant audio, where neither transcript is meaningful (this
  is the whole reason one noisy file shows 6 % drift — short garbage rows inflate
  the ratio).

No dropped sentences, no flipped meaning. On clean speech `int8_float16` was
occasionally *more* accurate ("court reporter" vs "poor reporter").

### Decision

`int8_float16` is the GPU default (`src/transcribe.py`). Overrides:

- `INTENSION_TRANSCRIBE_COMPUTE_TYPE=float16` — previous behaviour
- `INTENSION_TRANSCRIBE_COMPUTE_TYPE=float32` — full precision
- `INTENSION_GPU_SAFE_MODE=1` — drops to plain `int8` (the native-crash retry path)
