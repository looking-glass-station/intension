"""
prosody.py - vocal-aggression detection from the audio signal, independent of
what words are said or what topic is discussed.

Two tiers, cheap-first (mirrors the keyword->LLM shape of bias.py / invective.py):

  Tier 1  DSP features per diarized segment - RMS energy, peak loudness, spectral
          centroid, speaking rate - converted to a **speaker-relative z-score**
          (loud/bright/fast *for this person*, so mic gain and per-speaker
          baselines drop out). One STFT per file; effectively free.
  Tier 2  A speech-emotion model on the Tier-1 shortlist plus a small random
          calibration sample, to separate genuine hostility from merely loud
          (hype, laughter, dramatic reads all spike arousal without aggression).
          audeering/wav2vec2 gives continuous arousal / dominance / valence;
          "aggressive" ~ high arousal + high dominance + low valence.

Output is a **complete** per-segment record, not just flagged rows:
  <output_path>/prosody/<video>.csv           one row per diarized segment
  <output_path>/prosody_audacity/<video>.txt  labels for segments over threshold

Config: confs/prosody.json. Runs over transcription_labeled/*.csv + wav/*.wav.
If the Tier-2 model can't load, Tier 1 still produces the full record.

    python -m src.prosody
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm_sound import TqdmSound

from audio_affect import build_ser, ser_adv, aggression_from_adv
from configs import get_global_config, iter_processing_configs
from file_utils import filter_files_by_stems, dict_to_csv, audacity_writer
from logger import global_logger

HOP = 512
N_FFT = 1024

CSV_FIELDS = [
    "speaker", "speaker_name", "start_time", "end_time", "duration", "words",
    "speaking_rate", "rms_mean", "rms_peak", "centroid_mean", "rolloff_mean",
    "rms_z", "peak_rms_z", "centroid_z", "speaking_rate_z",
    "tier1_score", "tier1_flag",
    "arousal", "dominance", "valence", "tier2_score",
    "aggression_score",
]

# feature -> the z-score column feeding tier1
_Z_OF = {"rms": "rms_z", "peak_rms": "peak_rms_z",
         "centroid": "centroid_z", "speaking_rate": "speaking_rate_z"}


def load_prosody_config() -> dict:
    path = Path(__file__).parent.parent / "confs" / "prosody.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Tier 1 - DSP
# --------------------------------------------------------------------------- #
def frame_features(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Frame-level features for the whole file (one STFT)."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    spec = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))
    rms = librosa.feature.rms(S=spec, frame_length=N_FFT, hop_length=HOP)[0]
    centroid = librosa.feature.spectral_centroid(S=spec, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=spec, sr=sr)[0]
    times = librosa.frames_to_time(np.arange(rms.shape[0]), sr=sr, hop_length=HOP)
    return {"rms": rms, "centroid": centroid, "rolloff": rolloff, "times": times}


def segment_row(seg: dict, frames: Dict[str, np.ndarray]) -> Optional[dict]:
    start, end = float(seg["start_time"]), float(seg["end_time"])
    dur = end - start
    times = frames["times"]
    mask = (times >= start) & (times < end)
    if mask.sum() < 2:
        return None
    rms = frames["rms"][mask]
    words = len((seg.get("text") or "").split())
    return {
        "speaker": seg.get("speaker", ""),
        "speaker_name": seg.get("speaker_name", ""),
        "start_time": round(start, 3),
        "end_time": round(end, 3),
        "duration": round(dur, 3),
        "words": words,
        "speaking_rate": round(words / dur, 3) if dur > 0 else 0.0,
        "rms_mean": float(rms.mean()),
        "rms_peak": float(rms.max()),
        "centroid_mean": float(frames["centroid"][mask].mean()),
        "rolloff_mean": float(frames["rolloff"][mask].mean()),
    }


def add_speaker_zscores(rows: List[dict]) -> None:
    """Per-speaker z-score for each raw feature, in place."""
    by_speaker: Dict[str, List[dict]] = {}
    for r in rows:
        by_speaker.setdefault(r["speaker"], []).append(r)
    raw_of = {"rms": "rms_mean", "peak_rms": "rms_peak",
              "centroid": "centroid_mean", "speaking_rate": "speaking_rate"}
    for group in by_speaker.values():
        for feat, raw_col in raw_of.items():
            vals = np.array([r[raw_col] for r in group], dtype=float)
            mean, std = vals.mean(), vals.std()
            for r in group:
                r[_Z_OF[feat]] = round(float((r[raw_col] - mean) / std) if std > 1e-9 else 0.0, 3)


def tier1_score(row: dict, weights: Dict[str, float]) -> float:
    return round(sum(weights.get(f, 0.0) * row[_Z_OF[f]] for f in _Z_OF), 3)


# --------------------------------------------------------------------------- #
# Tier 2 - speech emotion (arousal / dominance / valence). Model + scoring live
# in audio_affect.py so invective.py shares one implementation.
# --------------------------------------------------------------------------- #
def aggression_score(row: dict, cfg: dict) -> float:
    """
    Combine Tier 1 (relative loudness/pitch/rate) with Tier 2 (SER confirms
    hostile vs. merely aroused). 0-1. Every row gets a score; rows without a
    Tier-2 pass fall back to a Tier-1-only mapping.
    """
    agg = cfg["aggression"]
    t1 = 1.0 / (1.0 + np.exp(-(row["tier1_score"] - cfg["tier1_threshold"])))  # sigmoid around the flag point
    if row.get("tier2_score") in (None, ""):
        return round(float(t1), 3)
    t2 = float(row["tier2_score"])
    w1, w2 = agg["tier1_weight"], agg["tier2_weight"]
    return round(float((w1 * t1 + w2 * t2) / (w1 + w2)), 3)


def process_transcript(transcript_file: Path, wav_file: Path,
                       out_csv: Path, out_audacity: Path, cfg: dict, ser,
                       logger, rng: random.Random) -> int:
    df = pd.read_csv(transcript_file)
    if df.empty or not wav_file.exists():
        dict_to_csv(out_csv, [], fields=CSV_FIELDS)
        audacity_writer(out_audacity, [])
        return 0

    audio, sr = sf.read(wav_file, dtype="float32")
    frames = frame_features(audio, sr)

    min_dur = cfg.get("min_segment_duration", 1.0)
    hosts_only = cfg.get("host_speakers_only", False)

    rows: List[dict] = []
    for _, seg in df.iterrows():
        seg = seg.to_dict()
        if hosts_only and str(seg.get("speaker_name", "")).strip() in ("", "Guest"):
            continue
        try:
            if float(seg["end_time"]) - float(seg["start_time"]) < min_dur:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        row = segment_row(seg, frames)
        if row is not None:
            row["text"] = seg.get("text", "")
            rows.append(row)

    if not rows:
        dict_to_csv(out_csv, [], fields=CSV_FIELDS)
        audacity_writer(out_audacity, [])
        return 0

    add_speaker_zscores(rows)
    weights = cfg["tier1_weights"]
    for r in rows:
        r["tier1_score"] = tier1_score(r, weights)
        r["tier1_flag"] = r["tier1_score"] >= cfg["tier1_threshold"]
        r["arousal"] = r["dominance"] = r["valence"] = r["tier2_score"] = ""

    # Tier 2: flagged rows + a random calibration sample
    if ser is not None:
        idx_flag = [i for i, r in enumerate(rows) if r["tier1_flag"]]
        pool = [i for i in range(len(rows)) if i not in idx_flag]
        idx_calib = rng.sample(pool, min(cfg.get("calibration_sample", 12), len(pool)))
        for i in set(idx_flag) | set(idx_calib):
            r = rows[i]
            a, b = int(float(r["start_time"]) * sr), int(float(r["end_time"]) * sr)
            clip = audio[a:b]
            if clip.ndim > 1:
                clip = clip.mean(axis=1)
            if clip.size < sr // 2:
                continue
            try:
                adv = ser_adv(clip, sr, ser)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"SER failed on {transcript_file.stem} @ {r['start_time']}s: {e}")
                continue
            r["arousal"] = round(adv["arousal"], 3)
            r["dominance"] = round(adv["dominance"], 3)
            r["valence"] = round(adv["valence"], 3)
            r["tier2_score"] = aggression_from_adv(adv, cfg["aggression"])

    flag_thr = cfg["flag_threshold"]
    labels = []
    for r in rows:
        r["aggression_score"] = aggression_score(r, cfg)
        r.pop("text", None)
        if r["aggression_score"] >= flag_thr:
            labels.append({
                "start_time": r["start_time"], "end_time": r["end_time"],
                "text": f"AGGRO {r['aggression_score']:.2f} ({r['speaker_name'] or r['speaker']})",
            })

    dict_to_csv(out_csv, [{k: r.get(k, "") for k in CSV_FIELDS} for r in rows], fields=CSV_FIELDS)
    audacity_writer(out_audacity, labels)
    return len(labels)


def main() -> None:
    logger = global_logger("prosody")
    cfg = load_prosody_config()
    global_config = get_global_config()
    rng = random.Random(1234)

    # Which files need doing (before loading any model).
    pending: List[tuple] = []
    for pc in iter_processing_configs(include_manual=True):
        labeled = pc.output_path / "transcription_labeled"
        wav_dir = pc.output_path / "wav"
        prosody_dir = pc.output_path / "prosody"
        aud_dir = pc.output_path / "prosody_audacity"
        if not labeled.exists():
            continue
        prosody_dir.mkdir(parents=True, exist_ok=True)
        aud_dir.mkdir(parents=True, exist_ok=True)
        for t in filter_files_by_stems(labeled, "csv", [prosody_dir, aud_dir]):
            pending.append((pc, t, wav_dir / f"{t.stem}.wav",
                            prosody_dir / t.name, aud_dir / f"{t.stem}.txt"))

    if not pending:
        logger.info("No transcripts to process for prosody")
        return

    ser = build_ser(cfg["ser_model"])
    if ser is None:
        logger.warning(f"Tier-2 model {cfg['ser_model']} unavailable; Tier 1 only")

    progress = TqdmSound(
        activity_mute_seconds=0,
        dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json"),
    )
    bar = progress.progress_bar(pending, total=len(pending), desc="Prosody",
                                unit="file", leave=True, ten_percent_ticks=True)
    total = 0
    for pc, transcript, wav, out_csv, out_aud in bar:
        bar.set_description(f"{pc.name} ({pc.channel_name_or_term}) - {transcript.stem}")
        try:
            total += process_transcript(transcript, wav, out_csv, out_aud, cfg, ser, logger, rng)
        except Exception as e:
            logger.error(f"Error on {transcript.stem}: {e}", exc_info=True)
    bar.close()
    logger.info(f"Prosody: {len(pending)} files, {total} segments over the aggression threshold")


if __name__ == "__main__":
    main()
