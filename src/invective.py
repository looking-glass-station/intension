"""invective.py

Detects invective/slur usage and positive mentions of harmful entities in transcripts.

Features:
- Dual detection: "bad" terms used as slurs, "good" terms used positively
- Multi-model scoring:
  - Text: Detoxify (identity_attack) + cardiffnlp (hate speech)
  - Audio: wav2vec2 emotion recognition (anger, tone)
- Rule-based heuristics to reduce false positives
- Audio clip extraction and analysis for all detections
- CSV output per transcript

Usage:
    python -m src.invective

Dependencies:
    pip install detoxify transformers torch pandas librosa soundfile speechbrain
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator, Optional, Literal

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from detoxify import Detoxify
from transformers import pipeline

import audio_affect
from configs import iter_processing_configs
from logger import global_logger

# prosody.py runs ahead of this stage and scores every diarized segment for
# aggressive delivery; a segment at/above this counts as "hostile audio" here.
HOSTILE_AGGRESSION_THRESHOLD = 0.55


@dataclass(frozen=True)
class Occurrence:
    """A single occurrence (possibly multiple terms from same group) in a transcript."""

    line_idx: int
    char_start: int
    char_end: int
    term_group: str  # Group name (e.g., "queer", "jew", "terrorist")
    matched_terms: list[str]  # All terms from this group found in the sentence
    term_type: Literal["bad", "good"]  # Whether from "bad" or "good" list
    speaker: str
    speaker_name: str
    start_time: float
    end_time: float
    sentence: str
    timestamp_url: str
    video_name: str  # Added for audio clip naming


@dataclass(frozen=True)
class ScoredOccurrence:
    """Occurrence with multi-model scores and classification."""

    occ: Occurrence
    context_window: str
    rule_bucket: str
    detoxify_identity_attack: float
    detoxify_toxicity: float
    cardiffnlp_label: str
    cardiffnlp_score: float
    prosody_aggression: Optional[float]  # prosody.py's per-segment score, None if that pass hasn't run
    final_label: str
    final_score: float
    audio_clip_path: Optional[str]


def load_invective_config(config_path: Path) -> dict:
    """Load invective configuration from JSON.

    Returns:
        dict with keys:
            'bad_groups': dict mapping group_name -> list of terms
            'good_groups': dict mapping group_name -> list of terms
            'bad_terms': flat list of all bad terms
            'good_terms': flat list of all good terms
            'term_to_group': dict mapping term -> (type, group_name)
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)

    bad_groups = {}
    good_groups = {}
    term_to_group = {}

    # Parse "bad" groups
    if isinstance(raw_config.get("bad"), list):
        for group_dict in raw_config["bad"]:
            for group_name, terms in group_dict.items():
                bad_groups[group_name] = terms
                for term in terms:
                    term_to_group[term.lower()] = ("bad", group_name)

    # Parse "good" groups
    if isinstance(raw_config.get("good"), list):
        for group_dict in raw_config["good"]:
            for group_name, terms in group_dict.items():
                good_groups[group_name] = terms
                for term in terms:
                    term_to_group[term.lower()] = ("good", group_name)

    # Flatten to get all terms
    bad_terms = [term for terms in bad_groups.values() for term in terms]
    good_terms = [term for terms in good_groups.values() for term in terms]

    return {
        'bad_groups': bad_groups,
        'good_groups': good_groups,
        'bad_terms': bad_terms,
        'good_terms': good_terms,
        'term_to_group': term_to_group
    }


def read_transcript_csv(path: Path) -> pd.DataFrame:
    """Read transcript CSV and filter out Guest speakers."""
    df = pd.read_csv(path)
    # Only process non-Guest speakers
    return df[df['speaker_name'] != 'Guest'].copy()


def iter_occurrences(
    df: pd.DataFrame,
    bad_terms: list[str],
    good_terms: list[str],
    term_to_group: dict[str, tuple[str, str]],
    video_name: str
) -> Iterator[Occurrence]:
    """Yield grouped occurrences of target terms in transcript DataFrame.

    Groups multiple terms from the same category in the same sentence into one occurrence.
    """

    # Build regex patterns for both term types
    all_terms = set(bad_terms + good_terms)
    if not all_terms:
        return

    escaped = [re.escape(t) for t in sorted(all_terms, key=len, reverse=True)]
    pat = re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)

    for idx, row in df.iterrows():
        text = str(row['text'])

        # Find all matches in this sentence
        matches = list(pat.finditer(text))
        if not matches:
            continue

        # Group matches by term group
        group_matches: dict[tuple[str, str], list] = {}  # (type, group) -> list of matches
        for m in matches:
            matched_term = m.group(1)
            term_lower = matched_term.lower()

            if term_lower in term_to_group:
                term_type, group_name = term_to_group[term_lower]
                key = (term_type, group_name)
                if key not in group_matches:
                    group_matches[key] = []
                group_matches[key].append((matched_term, m.start(1), m.end(1)))

        # Yield one Occurrence per group per sentence
        for (term_type, group_name), term_matches in group_matches.items():
            # Get unique matched terms and earliest position
            unique_terms = list(dict.fromkeys([t[0] for t in term_matches]))
            earliest_match = min(term_matches, key=lambda x: x[1])

            yield Occurrence(
                line_idx=int(idx),
                char_start=earliest_match[1],
                char_end=earliest_match[2],
                term_group=group_name,
                matched_terms=unique_terms,
                term_type=term_type,
                speaker=str(row['speaker']),
                speaker_name=str(row['speaker_name']),
                start_time=float(row['start_time']),
                end_time=float(row['end_time']),
                sentence=text,
                timestamp_url=str(row['timestamp_url']),
                video_name=video_name
            )


def context_window(df: pd.DataFrame, center_idx: int, before: int = 2, after: int = 2) -> str:
    """Join a small window of transcript lines around center_idx."""

    lo = max(0, center_idx - before)
    hi = min(len(df), center_idx + after + 1)

    # Get rows in window and concatenate their text
    window_rows = df.iloc[lo:hi]
    return " ".join(window_rows['text'].astype(str)).strip()


def rule_bucket_for_bad_term(matched_terms: list[str], sentence: str, window_text: str) -> str:
    """Classify usage of 'bad' terms (potential slurs).

    Buckets:
    - mention_or_metalinguistic: discussing the word itself
    - self_id_or_neutral: identity/neutral descriptor
    - likely_invective: likely used as insult/slur
    - unknown: unclear
    """

    # Check patterns for any of the matched terms
    for term in matched_terms:
        # 1) Metalinguistic / mentions
        if re.search(r"\b(the\s+word|term|phrase)\b.*\b" + re.escape(term) + r"\b", sentence, flags=re.I):
            return "mention_or_metalinguistic"
        if re.search(r"\b(meaning|definition|defined\s+as)\b.*\b" + re.escape(term) + r"\b", sentence, flags=re.I):
            return "mention_or_metalinguistic"
        if re.search(r"['\"]\s*" + re.escape(term) + r"\s*['\"]", sentence, flags=re.I):
            return "mention_or_metalinguistic"

        # 2) Identity / neutral descriptor
        if re.search(r"\b(i\s*am|i'm|im|we're|we\s+are)\b\s+" + re.escape(term) + r"\b", sentence, flags=re.I):
            return "self_id_or_neutral"
        if re.search(r"\b(he\s+is|she\s+is|they\s+are|he's|she's|they're)\b\s+" + re.escape(term) + r"\b", sentence, flags=re.I):
            return "self_id_or_neutral"
        if re.search(r"\b" + re.escape(term) + r"\b\s+\b(people|person|men|women|community|rights|marriage|couple|relationships)\b", sentence, flags=re.I):
            return "self_id_or_neutral"

        # 3) Likely invective usage
        if re.search(r"\b(you\s+are|you're|u\s+r|ur)\b\s+\b" + re.escape(term) + r"\b", sentence, flags=re.I):
            return "likely_invective"
        if re.search(r"\b(that('?s)?\s+so|thats\s+so)\b\s+\b" + re.escape(term) + r"\b", sentence, flags=re.I):
            return "likely_invective"
        if re.search(r"\b(so|really|very)\b\s+\b" + re.escape(term) + r"\b", sentence, flags=re.I):
            return "likely_invective"
        if re.search(r"\b(what\s+a|such\s+a)\b\s+\b" + re.escape(term) + r"\b", sentence, flags=re.I):
            return "likely_invective"

    # 4) Check window for hostile context
    if re.search(r"\b(shut\s+up|idiot|moron|loser|stupid|f\*\*k|fuck|damn|hate)\b", window_text, flags=re.I):
        return "likely_invective"

    return "unknown"


def rule_bucket_for_good_term(matched_terms: list[str], sentence: str, window_text: str) -> str:
    """Classify usage of 'good' terms (entities we want to detect positive mentions of).

    Buckets:
    - factual_or_critical: neutral reporting or criticism (normal, don't flag)
    - likely_positive: praising/supporting (FLAG THIS)
    - unknown: unclear
    """

    # 1) Factual/critical/negative mentions (normal)
    if re.search(r"\b(terrorist|terrorism|attack|killed|violence|war|conflict)\b", window_text, flags=re.I):
        return "factual_or_critical"
    if re.search(r"\b(condemned|condemn|denounce|against|oppose|opposes)\b", window_text, flags=re.I):
        return "factual_or_critical"

    # 2) Positive/supportive mentions (FLAG)
    if re.search(r"\b(support|supports|supporting|hero|brave|fight|fighting\s+for|resistance|freedom\s+fighter)\b", window_text, flags=re.I):
        return "likely_positive"

    for term in matched_terms:
        if re.search(r"\b(love|respect|admire|based|based\s+on)\b.*\b" + re.escape(term) + r"\b", sentence, flags=re.I):
            return "likely_positive"

    if re.search(r"\b(legitimate|justified|righteous)\b", window_text, flags=re.I):
        return "likely_positive"

    return "unknown"


def classify_occurrence(
    term_type: str,
    bucket: str,
    identity_attack: float,
    toxicity: float,
    hostile_audio: bool,
) -> tuple[str, float]:
    """
    Final label + score from the rule bucket, the text-model scores, and the
    tone signal (hostile_audio, now from prosody.py). Split out so it can be
    replayed against stored rows when the tone source changes.
    """
    if term_type == "bad":
        # hostile_audio (prosody's aggression score) only *modulates* a text
        # signal here - it can keep an identity term said with venom out of the
        # NEUTRAL bucket, and confirm a likely-invective slur into HIGH. It must
        # not manufacture a finding on its own: a heated debate *about* a group
        # is aggressive delivery, not invective, and would otherwise land in
        # MEDIUM via "hostile + some identity_attack".
        if bucket == "mention_or_metalinguistic":
            return "NON_INVECTIVE_MENTION", 0.10
        if bucket == "self_id_or_neutral" and identity_attack < 0.5 and not hostile_audio:
            return "NON_INVECTIVE_NEUTRAL", 0.15
        if bucket == "likely_invective" and (identity_attack >= 0.6 or toxicity >= 0.6) and hostile_audio:
            return "INVECTIVE_HIGH_CONFIDENCE", 0.80 + 0.20 * max(identity_attack, toxicity)
        if bucket == "likely_invective" and (identity_attack >= 0.6 or toxicity >= 0.6):
            return "INVECTIVE_HIGH_CONFIDENCE", 0.70 + 0.30 * max(identity_attack, toxicity)
        if identity_attack >= 0.7 or toxicity >= 0.7:
            return "INVECTIVE_MEDIUM_CONFIDENCE", 0.60 + 0.30 * max(identity_attack, toxicity)
        if bucket == "likely_invective" or identity_attack >= 0.4:
            return "INVECTIVE_LOW_CONFIDENCE", 0.30 + 0.20 * max(identity_attack, toxicity)
        return "NON_INVECTIVE_AMBIGUOUS", 0.20

    # term_type == "good": detect positive/praising usage (inverse logic)
    if bucket == "factual_or_critical":
        return "NON_PRAISE_CRITICAL", 0.10
    if bucket == "likely_positive" and toxicity < 0.4:
        return "PRAISE_HIGH_CONFIDENCE", 0.70 + 0.30 * (1.0 - toxicity)
    if toxicity < 0.3:
        return "PRAISE_MEDIUM_CONFIDENCE", 0.50 + 0.20 * (1.0 - toxicity)
    return "NON_PRAISE_AMBIGUOUS", 0.20


def build_models() -> tuple[Detoxify, pipeline]:
    """Detoxify (identity_attack + toxicity) + cardiffnlp hate speech.

    The audio/tone signal now comes from prosody.py (shared audeering model,
    scored once per segment) - see load_prosody_scores.
    """
    # transformers' pipeline(device_map="auto") needs the `accelerate` package;
    # cardiffnlp is small, so pass an explicit device instead (as topics.py does).
    device = 0 if torch.cuda.is_available() else -1
    detoxify_model = Detoxify('unbiased')
    cardiffnlp_model = pipeline(
        task="text-classification",
        model="cardiffnlp/twitter-roberta-base-hate",
        tokenizer="cardiffnlp/twitter-roberta-base-hate",
        truncation=True,
        device=device,
    )
    return detoxify_model, cardiffnlp_model


def load_prosody_scores(transcript_path: Path) -> dict[float, float]:
    """{segment start_time -> aggression_score} from prosody/<stem>.csv, if present."""
    p = transcript_path.parent.parent / "prosody" / transcript_path.name
    if not p.exists():
        return {}
    try:
        rows = pd.read_csv(p).to_dict("records")
        return {round(float(r["start_time"]), 2): float(r["aggression_score"]) for r in rows}
    except Exception:
        return {}


def score_occurrences(
    df: pd.DataFrame,
    occs: list[Occurrence],
    detoxify_model: Detoxify,
    cardiffnlp_model: pipeline,
    audio: Optional[np.ndarray],
    sr: int,
    clips_dir: Path,
    prosody_scores: dict[float, float],
    window_before: int = 2,
    window_after: int = 2,
) -> list[ScoredOccurrence]:
    """Score occurrences with the text models plus prosody.py's aggression score."""

    channel_dir = clips_dir.parents[1]  # <channel>/invective/clips -> <channel>
    results: list[ScoredOccurrence] = []

    for occ in occs:
        window = context_window(df, occ.line_idx, before=window_before, after=window_after)

        terms_str = "_".join(occ.matched_terms[:2])
        clip_path = clips_dir / f"{occ.video_name}_{occ.line_idx}_{terms_str}_temp.wav"

        # Review clip: slice the already-loaded array (no per-occurrence file read)
        if audio is not None:
            try:
                cs, ce = keyword_clip_window(occ.start_time, occ.end_time, occ.sentence,
                                             occ.char_start, lead_up_seconds=1.0)
                audio_affect.extract_clip(audio, sr, cs, ce, clip_path)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: Could not write clip for line {occ.line_idx}: {e}")

        # Get detoxify scores
        detox_results = detoxify_model.predict(window)
        identity_attack = float(detox_results['identity_attack'])
        toxicity = float(detox_results['toxicity'])

        # Get cardiffnlp scores
        cardiff_pred = cardiffnlp_model(window[:512], top_k=1)[0]  # Truncate for model limits
        cardiff_label = str(cardiff_pred.get("label", ""))
        cardiff_score = float(cardiff_pred.get("score", 0.0))

        # Tone signal from prosody.py's per-segment aggression score
        prosody_aggression = prosody_scores.get(round(occ.start_time, 2))
        hostile_audio = prosody_aggression is not None and prosody_aggression >= HOSTILE_AGGRESSION_THRESHOLD

        if occ.term_type == "bad":
            bucket = rule_bucket_for_bad_term(occ.matched_terms, occ.sentence, window)
        else:
            bucket = rule_bucket_for_good_term(occ.matched_terms, occ.sentence, window)
        final_label, final_score = classify_occurrence(
            occ.term_type, bucket, identity_attack, toxicity, hostile_audio)

        # Rename temp clip to final label
        final_clip_path = None
        if clip_path.exists():
            dest = clips_dir / f"{occ.video_name}_{occ.line_idx}_{terms_str}_{final_label}.wav"
            try:
                clip_path.rename(dest)
                final_clip_path = str(dest.relative_to(channel_dir))
            except Exception as e:
                print(f"Warning: Could not rename clip {clip_path}: {e}")
                final_clip_path = str(clip_path.relative_to(channel_dir))

        results.append(
            ScoredOccurrence(
                occ=occ,
                context_window=window,
                rule_bucket=bucket,
                detoxify_identity_attack=identity_attack,
                detoxify_toxicity=toxicity,
                cardiffnlp_label=cardiff_label,
                cardiffnlp_score=cardiff_score,
                prosody_aggression=prosody_aggression,
                final_label=final_label,
                final_score=final_score,
                audio_clip_path=final_clip_path,
            )
        )

    return results


def keyword_clip_window(
    start_time: float,
    end_time: float,
    sentence: str,
    keyword_char_pos: int,
    lead_up_seconds: float = 1.5,
    max_clip_duration: float = 8.0,
) -> tuple[float, float]:
    """
    (clip_start, clip_end) seconds for a review clip centred on the keyword:
    lead-up before the estimated keyword time, out to the end of the sentence or
    a max duration. The actual write is audio_affect.extract_clip on the loaded array.
    """
    if len(sentence) > 0:
        estimated_keyword_time = start_time + (end_time - start_time) * (keyword_char_pos / len(sentence))
    else:
        estimated_keyword_time = start_time
    clip_start = max(0.0, estimated_keyword_time - lead_up_seconds)
    clip_end = min(end_time, clip_start + max_clip_duration)
    return clip_start, clip_end




def to_dataframe(scored: list[ScoredOccurrence]) -> pd.DataFrame:
    """Convert results to a DataFrame."""

    rows = []
    for s in scored:
        o = s.occ
        rows.append(
            {
                "timestamp_start": o.start_time,
                "timestamp_end": o.end_time,
                "speaker": o.speaker,
                "speaker_name": o.speaker_name,
                "term_group": o.term_group,
                "matched_terms": ", ".join(o.matched_terms),
                "term_type": o.term_type,
                "sentence": o.sentence,
                "context_window": s.context_window,
                "rule_bucket": s.rule_bucket,
                "detoxify_identity_attack": round(s.detoxify_identity_attack, 3),
                "detoxify_toxicity": round(s.detoxify_toxicity, 3),
                "cardiffnlp_label": s.cardiffnlp_label,
                "cardiffnlp_score": round(s.cardiffnlp_score, 3),
                "prosody_aggression": ("" if s.prosody_aggression is None
                                       else round(s.prosody_aggression, 3)),
                "final_label": s.final_label,
                "final_score": round(s.final_score, 3),
                "timestamp_url": o.timestamp_url,
                "audio_clip_path": s.audio_clip_path or "",
            }
        )
    return pd.DataFrame(rows)


def process_transcript(
    transcript_path: Path,
    wav_path: Path,
    output_csv: Path,
    clips_dir: Path,
    bad_terms: list[str],
    good_terms: list[str],
    term_to_group: dict[str, tuple[str, str]],
    detoxify_model: Detoxify,
    cardiffnlp_model: pipeline,
    min_score: float = 0.5
) -> int:
    """Process a single transcript file."""

    df = read_transcript_csv(transcript_path)
    if df.empty:
        return 0

    video_name = transcript_path.stem
    occs = list(iter_occurrences(df, bad_terms, good_terms, term_to_group, video_name))
    if not occs:
        return 0

    # Load the wav once (for review clips) and prosody's per-segment scores.
    audio: Optional[np.ndarray] = None
    sr = 16000
    if wav_path.exists():
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not read {wav_path.name}: {e}")
    prosody_scores = load_prosody_scores(transcript_path)

    scored = score_occurrences(
        df, occs, detoxify_model, cardiffnlp_model,
        audio, sr, clips_dir, prosody_scores,
    )

    # Filter by minimum score
    scored = [s for s in scored if s.final_score >= min_score]
    if not scored:
        return 0

    # Convert to DataFrame and save
    results_df = to_dataframe(scored)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    return len(results_df)


def main() -> None:
    """Process all transcripts across all configured channels."""

    logger = global_logger("invective_detection")

    # Load configuration
    config_path = Path(__file__).parent.parent / "confs" / "invectives.json"
    config = load_invective_config(config_path)
    bad_terms = config['bad_terms']
    good_terms = config['good_terms']
    term_to_group = config['term_to_group']
    bad_groups = config['bad_groups']
    good_groups = config['good_groups']

    if not bad_terms and not good_terms:
        logger.warning("No terms configured in invectives.json")
        return

    logger.info(f"Monitoring {len(bad_groups)} 'bad' groups ({len(bad_terms)} terms) "
                f"and {len(good_groups)} 'good' groups ({len(good_terms)} terms)")

    # First pass: work out whether there's anything to do at all, so a model
    # download failure doesn't halt the pipeline when there's nothing pending.
    pending: list[tuple] = []  # (cfg, transcript_csv, wav_path, output_csv, clips_dir)
    for cfg in iter_processing_configs(include_manual=True):
        # Use labeled transcripts as canonical source; RTTM may be out of sync with transcript rows.
        transcription_dir = cfg.output_path / 'transcription_labeled'
        wav_dir = cfg.output_path / 'wav'
        invective_dir = cfg.output_path / 'invective'
        clips_dir = invective_dir / 'clips'

        if not transcription_dir.exists():
            continue

        for transcript_csv in transcription_dir.glob('*.csv'):
            output_csv = invective_dir / f"{transcript_csv.stem}.csv"
            # Skip if already processed and the transcript hasn't changed since.
            if output_csv.exists() and output_csv.stat().st_mtime > transcript_csv.stat().st_mtime:
                continue
            wav_path = wav_dir / f"{transcript_csv.stem}.wav"
            pending.append((cfg, transcript_csv, wav_path, output_csv, clips_dir))

    if not pending:
        logger.info("No transcripts to process for invective detection")
        return

    # Initialize models
    logger.info(f"Loading models (Detoxify + cardiffnlp) for {len(pending)} file(s)...")
    try:
        detoxify_model, cardiffnlp_model = build_models()
    except Exception as e:
        # If the text models can't load, log and let the pipeline continue
        # rather than halting everything.
        logger.error(f"Could not load invective models, skipping this stage: {e}", exc_info=True)
        return

    total_files = 0
    total_detections = 0
    for cfg, transcript_csv, wav_path, output_csv, clips_dir in tqdm(pending, desc="invective"):
        video_name = transcript_csv.stem
        try:
            detections = process_transcript(
                transcript_csv,
                wav_path,
                output_csv,
                clips_dir,
                bad_terms,
                good_terms,
                term_to_group,
                detoxify_model,
                cardiffnlp_model,
                min_score=0.5
            )
            total_files += 1
            total_detections += detections
        except Exception as e:
            logger.error(f"Error processing {video_name}: {e}", exc_info=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processed {total_files} files, found {total_detections} total detections")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
