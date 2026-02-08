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
from configs import get_configs, get_global_config, HostEmbedding
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
    known_embeddings = [np.load(e.embeddings_file) for e in host_embeddings]
    known_labels = [e.label for e in host_embeddings]

    audio, sr = sf.read(wav_file)
    target_similarity = 0.95

    with open(transcript_file, mode="r", encoding='utf-8') as file:
        transcript_lines = list(csv.DictReader(file))

    speakers = list({d['speaker'] for d in transcript_lines if 'speaker' in d})
    speaker_dict = {s: 'Guest' for s in speakers}

    unmatched = True
    while unmatched and target_similarity > 0:
        for transcript_line in transcript_lines:
            speaker_id = transcript_line['speaker']
            duration = float(transcript_line['duration'])
            start_time = float(transcript_line['start_time'])

            start_sample = int(start_time * sr)
            end_sample = start_sample + int(duration * sr)
            segment = audio[start_sample:end_sample]

            embedding = encoder.embed_utterance(segment)
            sims = [
                np.dot(embedding, ke) / (np.linalg.norm(embedding) * np.linalg.norm(ke))
                for ke in known_embeddings
            ]

            max_idx = int(np.argmax(sims))
            if sims[max_idx] >= target_similarity:
                speaker_dict[speaker_id] = known_labels[max_idx]
                unmatched = False

        target_similarity -= 0.01

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
    encoder = VoiceEncoder()
    global_config = get_global_config()

    for channel_cfg in get_configs():
        for config_list in vars(channel_cfg.download_configs).values():
            if not isinstance(config_list, list) or not config_list:
                continue

            for cfg in config_list:
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
                    continue

                if not training_folder.exists():
                    continue

                emb_entries = cfg.host_embeddings
                if not emb_entries:
                    logger.info(f"No files to label for: {cfg.name} ({cfg.channel_name_or_term}) - no embeddings")
                    continue

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

                with concurrent.futures.ThreadPoolExecutor(max_workers=system_config.max_workers(1)) as executor:
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
