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

import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
from detoxify import Detoxify
from transformers import pipeline

from configs import get_configs
from logger import global_logger


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
    audio_emotion_label: str  # e.g., "angry", "neutral", "happy"
    audio_emotion_score: float  # confidence in that emotion
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


def build_models() -> tuple[Detoxify, pipeline, pipeline]:
    """Initialize Detoxify, cardiffnlp, and audio emotion models."""

    # Detoxify unbiased model (for identity_attack + toxicity)
    detoxify_model = Detoxify('unbiased')

    # Cardiff NLP hate speech model (for casual/social media speech)
    cardiffnlp_model = pipeline(
        task="text-classification",
        model="cardiffnlp/twitter-roberta-base-hate",
        tokenizer="cardiffnlp/twitter-roberta-base-hate",
        truncation=True,
        device_map="auto",
    )

    # Audio emotion recognition model (for tone/sentiment in voice)
    audio_emotion_model = pipeline(
        task="audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device_map="auto",
    )

    return detoxify_model, cardiffnlp_model, audio_emotion_model


def score_occurrences(
    df: pd.DataFrame,
    occs: list[Occurrence],
    detoxify_model: Detoxify,
    cardiffnlp_model: pipeline,
    audio_emotion_model: pipeline,
    wav_path: Path,
    clips_dir: Path,
    window_before: int = 2,
    window_after: int = 2,
) -> list[ScoredOccurrence]:
    """Score occurrences using text models + audio emotion analysis."""

    results: list[ScoredOccurrence] = []

    for occ in occs:
        window = context_window(df, occ.line_idx, before=window_before, after=window_after)

        # Extract audio clip first (for emotion analysis)
        terms_str = "_".join(occ.matched_terms[:2])
        clip_filename = f"{occ.video_name}_{occ.line_idx}_{terms_str}_temp.wav"
        clip_path = clips_dir / clip_filename

        audio_emotion_label = "unknown"
        audio_emotion_score = 0.0

        if wav_path.exists():
            try:
                clip_path.parent.mkdir(parents=True, exist_ok=True)
                extract_audio_clip(
                    wav_path,
                    occ.start_time,
                    occ.end_time,
                    clip_path,
                    sentence=occ.sentence,
                    keyword_char_pos=occ.char_start,
                    lead_up_seconds=1.0
                )
                # Score audio emotion
                audio_emotion_label, audio_emotion_score = score_audio_emotion(clip_path, audio_emotion_model)
            except Exception as e:
                print(f"Warning: Could not process audio for line {occ.line_idx}: {e}")

        # Get detoxify scores
        detox_results = detoxify_model.predict(window)
        identity_attack = float(detox_results['identity_attack'])
        toxicity = float(detox_results['toxicity'])

        # Get cardiffnlp scores
        cardiff_pred = cardiffnlp_model(window[:512], top_k=1)[0]  # Truncate for model limits
        cardiff_label = str(cardiff_pred.get("label", ""))
        cardiff_score = float(cardiff_pred.get("score", 0.0))

        # Check if audio indicates hostile/angry tone
        hostile_audio = audio_emotion_label.lower() in ["angry", "disgust", "fear"] and audio_emotion_score >= 0.5

        # Different logic for "bad" vs "good" terms
        if occ.term_type == "bad":
            # For bad terms: detect invective usage
            bucket = rule_bucket_for_bad_term(occ.matched_terms, occ.sentence, window)

            # Metalinguistic mentions are not invective
            if bucket == "mention_or_metalinguistic":
                final_label = "NON_INVECTIVE_MENTION"
                final_score = 0.10

            # Neutral/identity mentions with low toxicity
            elif bucket == "self_id_or_neutral" and identity_attack < 0.5 and not hostile_audio:
                final_label = "NON_INVECTIVE_NEUTRAL"
                final_score = 0.15

            # High confidence invective: rules + models + audio agree
            elif bucket == "likely_invective" and (identity_attack >= 0.6 or toxicity >= 0.6) and hostile_audio:
                final_label = "INVECTIVE_HIGH_CONFIDENCE"
                final_score = 0.80 + 0.20 * max(identity_attack, toxicity)

            # High confidence from text models only
            elif bucket == "likely_invective" and (identity_attack >= 0.6 or toxicity >= 0.6):
                final_label = "INVECTIVE_HIGH_CONFIDENCE"
                final_score = 0.70 + 0.30 * max(identity_attack, toxicity)

            # Medium confidence: strong text signal OR hostile audio + some text signal
            elif identity_attack >= 0.7 or toxicity >= 0.7 or (hostile_audio and identity_attack >= 0.5):
                final_label = "INVECTIVE_MEDIUM_CONFIDENCE"
                final_score = 0.60 + 0.30 * max(identity_attack, toxicity)

            # Low confidence: some signals but unclear
            elif bucket == "likely_invective" or identity_attack >= 0.4 or hostile_audio:
                final_label = "INVECTIVE_LOW_CONFIDENCE"
                final_score = 0.30 + 0.20 * max(identity_attack, toxicity)

            else:
                final_label = "NON_INVECTIVE_AMBIGUOUS"
                final_score = 0.20

        else:  # term_type == "good"
            # For good terms: detect POSITIVE usage (inverse logic)
            bucket = rule_bucket_for_good_term(occ.matched_terms, occ.sentence, window)

            # Critical/factual mentions are normal (low toxicity expected)
            if bucket == "factual_or_critical":
                final_label = "NON_PRAISE_CRITICAL"
                final_score = 0.10

            # Positive mentions with LOW toxicity = praising terrorists (FLAG)
            elif bucket == "likely_positive" and toxicity < 0.4:
                final_label = "PRAISE_HIGH_CONFIDENCE"
                final_score = 0.70 + 0.30 * (1.0 - toxicity)

            # Low toxicity without clear positive rules
            elif toxicity < 0.3:
                final_label = "PRAISE_MEDIUM_CONFIDENCE"
                final_score = 0.50 + 0.20 * (1.0 - toxicity)

            else:
                final_label = "NON_PRAISE_AMBIGUOUS"
                final_score = 0.20

        # Rename temp clip to final label
        final_clip_path = None
        if clip_path.exists():
            final_clip_filename = f"{occ.video_name}_{occ.line_idx}_{terms_str}_{final_label}.wav"
            final_clip_path = clips_dir / final_clip_filename
            try:
                clip_path.rename(final_clip_path)
                final_clip_path = str(final_clip_path.relative_to(wav_path.parent.parent))
            except Exception as e:
                print(f"Warning: Could not rename clip {clip_path}: {e}")
                final_clip_path = str(clip_path.relative_to(wav_path.parent.parent))

        results.append(
            ScoredOccurrence(
                occ=occ,
                context_window=window,
                rule_bucket=bucket,
                detoxify_identity_attack=identity_attack,
                detoxify_toxicity=toxicity,
                cardiffnlp_label=cardiff_label,
                cardiffnlp_score=cardiff_score,
                audio_emotion_label=audio_emotion_label,
                audio_emotion_score=audio_emotion_score,
                final_label=final_label,
                final_score=final_score,
                audio_clip_path=final_clip_path,
            )
        )

    return results


def extract_audio_clip(
    wav_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    sentence: str,
    keyword_char_pos: int,
    lead_up_seconds: float = 1.5,
    max_clip_duration: float = 8.0
) -> bool:
    """Extract audio segment with keyword context.

    Extracts a short clip with lead-up to the keyword and a few seconds after.
    Caps the maximum clip duration to keep files manageable.

    Returns:
        True if extraction succeeded, False otherwise
    """

    sentence_duration = end_time - start_time
    sentence_length = len(sentence)

    # Estimate keyword position in audio based on character position
    # Assume roughly uniform speech rate across the sentence
    if sentence_length > 0:
        keyword_ratio = keyword_char_pos / sentence_length
        estimated_keyword_time = start_time + (sentence_duration * keyword_ratio)
    else:
        estimated_keyword_time = start_time

    # Extract from lead_up before keyword
    clip_start = max(0, estimated_keyword_time - lead_up_seconds)

    # Extract until end of sentence OR max duration, whichever is shorter
    clip_end = min(
        end_time,  # End of sentence
        clip_start + max_clip_duration  # Maximum clip length
    )

    clip_duration = clip_end - clip_start

    # Load and extract segment
    audio, sr = librosa.load(
        str(wav_path),
        offset=clip_start,
        duration=clip_duration,
        sr=16000,
        mono=True
    )

    # Save clip
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr)
    return True


def score_audio_emotion(audio_path: Path, emotion_model: pipeline) -> tuple[str, float]:
    """Score audio clip for emotion/tone using wav2vec2 model.

    Returns:
        (emotion_label, confidence_score)
    """
    try:
        # Run emotion classification
        result = emotion_model(str(audio_path), top_k=1)

        if result and len(result) > 0:
            top_emotion = result[0]
            label = str(top_emotion.get("label", "unknown"))
            score = float(top_emotion.get("score", 0.0))
            return label, score
        else:
            return "unknown", 0.0

    except Exception as e:
        print(f"Warning: Could not score audio emotion for {audio_path}: {e}")
        return "error", 0.0




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
                "audio_emotion_label": s.audio_emotion_label,
                "audio_emotion_score": round(s.audio_emotion_score, 3),
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
    audio_emotion_model: pipeline,
    min_score: float = 0.5
) -> int:
    """Process a single transcript file."""

    # Read transcript
    df = read_transcript_csv(transcript_path)
    if df.empty:
        return 0

    # Get video name for audio clip naming
    video_name = transcript_path.stem

    # Find occurrences
    occs = list(iter_occurrences(df, bad_terms, good_terms, term_to_group, video_name))
    if not occs:
        return 0

    # Score occurrences (now includes audio extraction and emotion analysis)
    scored = score_occurrences(
        df,
        occs,
        detoxify_model,
        cardiffnlp_model,
        audio_emotion_model,
        wav_path,
        clips_dir
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

    # Initialize models
    logger.info("Loading models (Detoxify + cardiffnlp + audio emotion)...")
    detoxify_model, cardiffnlp_model, audio_emotion_model = build_models()

    # Process all configured channels
    total_files = 0
    total_detections = 0

    for channel_cfg in get_configs():
        for config_list in vars(channel_cfg.download_configs).values():
            if not isinstance(config_list, list) or not config_list:
                continue

            for cfg in config_list:
                # Use labeled transcripts as canonical source; RTTM may be out of sync with transcript rows.
                transcription_dir = cfg.output_path / 'transcription_labeled'
                wav_dir = cfg.output_path / 'wav'
                invective_dir = cfg.output_path / 'invective'
                clips_dir = invective_dir / 'clips'

                if not transcription_dir.exists():
                    continue

                # Collect all transcripts to process
                all_transcripts = list(transcription_dir.glob('*.csv'))
                transcripts_to_process = []

                for transcript_csv in all_transcripts:
                    video_name = transcript_csv.stem
                    output_csv = invective_dir / f"{video_name}.csv"

                    # Skip if already processed and WAV hasn't changed
                    if output_csv.exists():
                        transcript_mtime = transcript_csv.stat().st_mtime
                        output_mtime = output_csv.stat().st_mtime
                        if output_mtime > transcript_mtime:
                            continue

                    transcripts_to_process.append(transcript_csv)

                if not transcripts_to_process:
                    continue

                # Process each transcript with progress bar
                progress_desc = f"{cfg.channel_name_or_term}"
                for transcript_csv in tqdm(transcripts_to_process, desc=progress_desc):
                    video_name = transcript_csv.stem
                    wav_path = wav_dir / f"{video_name}.wav"
                    output_csv = invective_dir / f"{video_name}.csv"

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
                            audio_emotion_model,
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
