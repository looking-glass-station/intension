import concurrent.futures
import csv
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder
from tqdm_sound import TqdmSound

import system_config
from configs import get_global_config, HostEmbedding, iter_processing_configs
from file_utils import dict_to_csv, audacity_writer
from logger import global_logger

logger = global_logger("label_speakers")


def label_speakers(
        wav_file: Path,
        transcript_file: Path,
        transcript_file_labeled: Path,
        audacity_file: Path,
        encoder: VoiceEncoder,
        host_embeddings: List[HostEmbedding]
):
    # Use transcript rows as the source of truth for text/timing.
    # RTTM files may not stay perfectly aligned with transcript rows over time.
    known_embeddings = [np.load(e.embeddings_file) for e in host_embeddings]
    known_labels = [e.label for e in host_embeddings]

    audio, sr = sf.read(wav_file)
    target_similarity = 0.95

    with open(transcript_file, mode="r", encoding='utf-8') as file:
        transcript_lines = list(csv.DictReader(file))

    speakers = list({d['speaker'] for d in transcript_lines if 'speaker' in d})
    speaker_dict = {s: 'Guest' for s in speakers}

    # Aggregate per-speaker embeddings to avoid repeated per-row encoding.
    speaker_durations = {}
    speaker_first_appearance = {}
    speaker_embedding_sums = {}

    for idx, transcript_line in enumerate(transcript_lines):
        speaker_id = transcript_line['speaker']
        duration = float(transcript_line['duration'])
        start_time = float(transcript_line['start_time'])

        # Skip invalid segments to prevent slicing errors.
        if duration <= 0.05 or duration > 3600:  # <0.05s or >1h
            continue

        speaker_first_appearance.setdefault(speaker_id, idx)
        speaker_durations[speaker_id] = speaker_durations.get(speaker_id, 0.0) + duration

        start_sample = int(start_time * sr)
        end_sample = start_sample + int(duration * sr)

        # Protect against bad slicing.
        if end_sample <= start_sample or start_sample < 0 or end_sample > len(audio):
            continue

        segment = audio[start_sample:end_sample]

        embedding = encoder.embed_utterance(segment)
        speaker_embedding_sums[speaker_id] = (
            speaker_embedding_sums.get(speaker_id, np.zeros_like(embedding)) + embedding * duration
        )

    speaker_embeddings = {
        spk: speaker_embedding_sums[spk] / speaker_durations[spk]
        for spk in speaker_embedding_sums
    }

    # Prioritize host matching by first speaker (primary) and longest speaker (secondary).
    first_speaker_id = min(speaker_first_appearance, key=speaker_first_appearance.get)
    sorted_by_duration = sorted(speaker_durations, key=lambda s: speaker_durations[s], reverse=True)
    candidate_speakers = [first_speaker_id] + [s for s in sorted_by_duration if s != first_speaker_id]

    speaker_to_host_sim = {}
    for speaker_id in candidate_speakers:
        emb = speaker_embeddings[speaker_id]
        speaker_to_host_sim[speaker_id] = [
            np.dot(emb, ke) / (np.linalg.norm(emb) * np.linalg.norm(ke))
            for ke in known_embeddings
        ]

    # Find primary host from prioritized speakers (first speaker -> longest speakers).
    selected_host_idx = None
    selected_host_label = None

    while selected_host_idx is None and target_similarity >= 0:
        for speaker_id in candidate_speakers:
            sims = speaker_to_host_sim[speaker_id]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]

            if best_sim >= target_similarity:
                selected_host_idx = best_idx
                selected_host_label = known_labels[best_idx]
                logger.info(
                    f"Selected primary host {selected_host_label} from speaker {speaker_id} "
                    f"(confidence {best_sim:.3f}, threshold {target_similarity:.2f})"
                )
                break

        if selected_host_idx is None:
            target_similarity -= 0.01

    if selected_host_idx is None:
        error_details = []
        for speaker_id in candidate_speakers:
            sims = speaker_to_host_sim[speaker_id]
            best_idx = int(np.argmax(sims))
            error_details.append(f"{speaker_id}: {known_labels[best_idx]} ({sims[best_idx]:.3f})")
        names = ", ".join(known_labels)
        error = (
            f"NO host labels ({names}) FOUND IN {wav_file} - consider lowering similarity below "
            f"{target_similarity + 0.01:.2f}. Details: {', '.join(error_details)}"
        )
        logger.error(error)
        raise RuntimeError(error)

    # Assign selected host label to all {@speaker_id} with sufficient similarity.
    min_host_assignment_similarity = max(target_similarity, 0.80)
    assigned_any = False
    for speaker_id, sims in speaker_to_host_sim.items():
        if sims[selected_host_idx] >= min_host_assignment_similarity:
            speaker_dict[speaker_id] = selected_host_label
            assigned_any = True
            logger.info(
                f"Assigned {selected_host_label} to speaker {speaker_id} "
                f"(confidence: {sims[selected_host_idx]:.3f})"
            )

    if not assigned_any:
        # This should not happen if we selected a host above, but include fallback protection.
        error = f"Matched host {selected_host_label} but could not assign to any speaker in {wav_file}."
        logger.error(error)
        raise RuntimeError(error)


    if not any(label in speaker_dict.values() for label in known_labels):
        names = ", ".join(known_labels)
        error = f"NO host labels ({names}) FOUND IN {wav_file} - consider lowering similarity below {target_similarity:.2f}"
        logger.error(error)
        raise RuntimeError(error)

    audacity_results = []
    for transcript_line in transcript_lines:
        speaker_id = transcript_line['speaker']
        transcript_line['speaker_name'] = speaker_dict[speaker_id]
        audacity_results.append({
            'start_time': transcript_line['start_time'],
            'end_time': transcript_line['end_time'],
            'text': speaker_dict[speaker_id]
        })

    dict_to_csv(transcript_file_labeled, transcript_lines)
    audacity_writer(audacity_file, audacity_results)

    logger.info(f"Wrote labels for {transcript_file.stem}")


def main():
    # resemblyzer's VoiceEncoder is a tiny 3-layer LSTM. Per short segment the GPU
    # is all kernel-launch + host<->device copy overhead, and it serialises the
    # per-file threads. On CPU it's ~1.8x faster over the sample and the mel
    # spectrogram (already CPU/librosa) feeds straight in - host labels come out
    # identical. See benchmarks/transcription/NOTES.md.
    # INTENSION_LABEL_SPEAKERS_DEVICE=cuda forces the GPU path back on.
    encoder_device = os.environ.get("INTENSION_LABEL_SPEAKERS_DEVICE", "cpu").strip() or "cpu"
    encoder = VoiceEncoder(device=encoder_device)
    global_config = get_global_config()

    for cfg in iter_processing_configs(include_manual=True):
        data_out = cfg.output_path
        training_folder = data_out / 'training'
        # If transcripts are intentionally skipped, just write host name label files per recording.
        if getattr(cfg, "no_transcript", False):
            wav_folder = data_out / "wav"
            audacity_host_folder = data_out / "audacity_labels_hostnames"
            diarized_audacity_folder = data_out / "audacity_labels"
            if not wav_folder.exists():
                logger.info(f"No wav folder for: {cfg.name} ({cfg.channel_name_or_term})")
                continue
            if not (cfg.hosts or []):
                logger.info(f"No hosts defined for: {cfg.name} ({cfg.channel_name_or_term})")
                continue
            audacity_host_folder.mkdir(parents=True, exist_ok=True)
            for wav_file in wav_folder.glob("*.wav"):
                label_path = audacity_host_folder / f"{wav_file.stem}.txt"
                diarized_label_path = diarized_audacity_folder / f"{wav_file.stem}.txt"
                if diarized_label_path.exists():
                    # Rewrite diarized labels replacing SPEAKER_* with host names when possible.
                    out_lines = []
                    speaker_re = re.compile(r"SPEAKER_(\d+)")
                    for line in diarized_label_path.read_text(encoding="utf-8").splitlines():
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            tag = parts[2]
                            m = speaker_re.match(tag)
                            if m:
                                idx = int(m.group(1))
                                if 0 <= idx < len(cfg.hosts):
                                    parts[2] = cfg.hosts[idx]
                            out_lines.append("\t".join(parts))
                    label_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
                    logger.info(f"Wrote host-mapped labels for {wav_file.stem}")
                else:
                    # No diarized labels; just emit a mapping of speaker tags to hosts.
                    lines = []
                    for idx, host in enumerate(cfg.hosts):
                        speaker_tag = f"SPEAKER_{idx:02d}"
                        lines.append(f"0.000000\t0.010000\t{speaker_tag} {host}\n")
                    label_path.write_text("".join(lines), encoding="utf-8")
                    logger.info(f"Wrote host mapping labels for {wav_file.stem}")
            # Fall through to transcript labeling if transcripts and embeddings exist.

        if not training_folder.exists():
            continue

        emb_entries = cfg.host_embeddings
        if not emb_entries:
            logger.info(f"No files to label for: {cfg.name} ({cfg.channel_name_or_term}) - no embeddings")
            continue

        # Canonical text/timing inputs are in `transcription/`; do not join against RTTM rows here.
        transcript_folder = data_out / 'transcription'
        labeled_folder = data_out / 'transcription_labeled'
        audacity_folder = data_out / 'transcription_labeled_audacity'

        if not transcript_folder.exists():
            logger.info(f"No transcript folder for: {cfg.name} ({cfg.channel_name_or_term})")
            continue

        os.makedirs(labeled_folder, exist_ok=True)
        os.makedirs(audacity_folder, exist_ok=True)

        # Find all .csv files in transcription, skip if labeled already exists
        transcript_files = [
            f for f in transcript_folder.glob("*.csv")
            if not (labeled_folder / f.name).exists()
        ]

        # Find corresponding .wav file for each stem (required for labeling)
        wav_folder = data_out / 'wav'
        files_to_label = []
        for transcript_file in transcript_files:
            stem = transcript_file.stem
            wav_file = wav_folder / f"{stem}.wav"
            transcript_labeled = labeled_folder / f"{stem}.csv"
            audacity_file = audacity_folder / f"{stem}.txt"

            if transcript_labeled.exists() and audacity_file.exists():
                continue

            if wav_file.exists():
                files_to_label.append(
                    (wav_file, transcript_file, transcript_labeled, audacity_file, encoder, emb_entries))
            else:
                logger.warning(f"WAV missing for {transcript_file.name}")

        if not files_to_label:
            logger.info(f"No files to label for: {cfg.name} ({cfg.channel_name_or_term})")
            continue

        progress = TqdmSound(
            activity_mute_seconds=0,
            dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json")
        )

        bar = progress.progress_bar(
            files_to_label,
            desc=f"Labeling {cfg.name}",
            total=len(files_to_label),
            leave=True,
            ten_percent_ticks=True,
        )

        # ~4 is this stage's sweet spot: the per-file work is mel-spectrogram +
        # tiny-LSTM throughput, not core-bound, and numpy already threads each
        # task - 8+ workers flat-lines or regresses. (benchmarks/transcription/NOTES.md)
        label_workers = system_config.max_workers(absolute_count=4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=label_workers) as executor:
            futures = {
                executor.submit(label_speakers, *args): args[1]  # transcript_file
                for args in files_to_label
            }

            for future in concurrent.futures.as_completed(futures):
                transcript_file = futures[future]
                bar.set_description(f"{cfg.name} ({cfg.channel_name_or_term}) - {transcript_file.stem}")
                bar.update(1)

        bar.close()


if __name__ == '__main__':
    main()
