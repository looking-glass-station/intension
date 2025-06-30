import csv
import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
from librosa import load
from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines import OverlappedSpeechDetection
from resemblyzer import VoiceEncoder

# Suppress non-critical warnings and verbose logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("resemblyzer").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r".*Model was trained with .*Bad things might happen.*",
    category=UserWarning,
)

from file_utils import parse_rttm
from configs import get_configs, get_global_config
import system_config
from tqdm_sound import TqdmSound
from logger import global_logger


def top_n_speaker_segments(
        rttm_path: Union[str, Path],
        top_n: int,
        min_seconds: float = 5,
        max_seconds: Optional[float] = 15,
        max_segments: Optional[int] = 2
) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    """
    Filter RTTM segments for each speaker by segment duration and select top-N speakers.

    Args:
        rttm_path (str or Path): Path to the RTTM file.
        top_n (int): Number of speakers to select.
        min_seconds (float): Minimum segment duration in seconds.
        max_seconds (Optional[float]): Maximum segment duration in seconds.
        max_segments (Optional[int]): Maximum number of segments per speaker.

    Returns:
        Dict[str, List[Dict]]: Mapping of speaker name to a list of their valid segments.
    """
    raw_segments = parse_rttm(Path(rttm_path))
    filtered = [
        s for s in raw_segments
        if min_seconds <= s['duration'] <= (max_seconds or s['duration'])
    ]

    totals: Dict[str, float] = {}
    for seg in filtered:
        spk = seg['speaker']
        totals[spk] = totals.get(spk, 0.0) + seg['duration']

    top_speakers = sorted(totals, key=totals.get, reverse=True)[:top_n]
    result: Dict[str, List[Dict[str, Union[str, float]]]] = {spk: [] for spk in top_speakers}
    for seg in filtered:
        if seg['speaker'] in result:
            result[seg['speaker']].append(seg)

    if max_segments is not None:
        for spk in result:
            result[spk] = result[spk][:max_segments]

    return result


def main():
    """
    For each YouTube config, extract clean speaker segments from wav/rttm,
    generate embeddings, and save progress parameters to avoid recomputation.
    """
    logger = global_logger("train_embeddings")

    global_config = get_global_config()
    satisfied_configs = []
    unsatisfied_configs = []

    # Initialize diarization pipeline
    pipeline = Pipeline.from_pretrained(
        'pyannote/speaker-diarization-3.1',
        use_auth_token=global_config.hf_token
    )
    pipeline.to(system_config.torch_device)

    # Initialize overlap detection model
    seg_model = Model.from_pretrained(
        'pyannote/segmentation',
        use_auth_token=global_config.hf_token
    )
    osd = OverlappedSpeechDetection(segmentation=seg_model)
    osd.instantiate({
        'onset': 1 - getattr(global_config.host_match, 'overlap_tolerance', 0.5),
        'offset': 1 - getattr(global_config.host_match, 'overlap_tolerance', 0.5),
        'min_duration_on': 0.0,
        'min_duration_off': 0.0,
    })

    encoder = VoiceEncoder()
    progress = TqdmSound(
        activity_mute_seconds=0,
        dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json")
    )

    for channel_cfg in get_configs():
        for config_list in vars(channel_cfg.download_configs).values():
            if not isinstance(config_list, list) or not config_list:
                continue

            for cfg in config_list:
                wav_dir = cfg.output_path / 'wav'
                rttm_dir = cfg.output_path / 'rttm'
                training_folder = cfg.output_path / 'training'
                training_folder.mkdir(exist_ok=True)
                param_file = training_folder / 'last_run_params.txt'

                # Gather all WAVs with matching RTTM files
                wav_files = [
                    p for p in wav_dir.glob('*.wav')
                    if (rttm_dir / f"{p.stem}.rttm").exists()
                ]
                available = len(wav_files)

                # Check if parameters and files have changed since last run
                if param_file.exists():
                    with open(param_file) as f:
                        params = list(csv.DictReader(f))[0]
                        keys = ['sample_file_count', 'max_segments', 'target_files', 'available_files']
                        if all(k in params for k in keys):
                            if all([
                                int(params['sample_file_count']) == global_config.host_match.sample_file_count,
                                int(params['max_segments']) == global_config.host_match.per_file_segment_count,
                                int(params['target_files']) == global_config.host_match.target_file_count,
                                int(params['available_files']) == available
                            ]):
                                logger.info(f"Skipping {cfg.channel_name_or_term}: config unchanged.")
                                satisfied_configs.append(cfg.channel_name_or_term)
                                continue

                current_count = int(len(list(training_folder.glob('*.wav'))) / max(1, len(cfg.hosts))) - 1
                if current_count >= global_config.host_match.target_file_count:
                    logger.info(f"Skipping {cfg.name}: already processed {current_count} files.")
                    satisfied_configs.append(cfg.name)
                    continue

                if available < global_config.host_match.sample_file_count:
                    logger.info(f"Skipping {cfg.channel_name_or_term}: insufficient wav files ({available}).")
                    unsatisfied_configs.append(cfg.channel_name_or_term)
                    continue

                # Remove previous training data
                for f in training_folder.iterdir():
                    if f.is_file():
                        f.unlink()

                sample_target = min(
                    len(wav_files),
                    global_config.host_match.sample_file_count * len(cfg.hosts)
                )

                wav_samples = [
                    wav_files[i]
                    for i in np.linspace(0, len(wav_files) - 1, sample_target, dtype=int)
                ]

                # Extract speaker segments
                audio_chunks = []
                for wav_path in progress.progress_bar(
                        wav_samples,
                        total=len(wav_samples),
                        desc=f"{cfg.name} ({cfg.channel_name_or_term}) - sampling"
                ):
                    rttm_path = rttm_dir / f"{wav_path.stem}.rttm"
                    segments = top_n_speaker_segments(
                        rttm_path,
                        len(cfg.hosts),
                        max_segments=global_config.host_match.per_file_segment_count
                    )
                    for segs in segments.values():
                        for seg in segs:
                            audio, _ = load(wav_path, offset=seg['start_time'], duration=seg['duration'], sr=16000)
                            audio_chunks.append(audio)

                combined_path = training_folder / 'combined_audio.wav'
                sf.write(combined_path, np.concatenate(audio_chunks), 16000)

                # Diarize combined audio
                annotation = pipeline(
                    str(combined_path),
                    min_speakers=len(cfg.hosts) + 1
                )
                diarized = [
                    {
                        'start': s.start,
                        'end': s.end,
                        'duration': s.end - s.start,
                        'label': l
                    }
                    for s, _, l in annotation.itertracks(yield_label=True)
                    if 5 <= (s.end - s.start) <= 30
                ]

                # Assign speakers to top-N frequent labels
                top_lbls = [
                    lbl for lbl, _ in Counter(d['label'] for d in diarized).most_common(len(cfg.hosts))
                ]
                grouped = {
                    idx: [d for d in diarized if d['label'] == lbl]
                    for idx, lbl in enumerate(top_lbls)
                }

                # Generate embeddings
                for idx, segs in grouped.items():
                    label_file = training_folder / f'embeddings_speaker_label_{idx}.txt'
                    emb_file = training_folder / f'embeddings_speaker_{idx}.npy'
                    embeddings = []

                    for i, seg in enumerate(progress.progress_bar(
                            segs, desc=f"{cfg.name} ({cfg.channel_name_or_term}) - embedding speaker {idx}"
                    )):
                        audio, _ = load(
                            combined_path,
                            offset=seg['start'],
                            duration=seg['duration'],
                            sr=16000
                        )
                        embeddings.append(encoder.embed_utterance(audio))
                        sf.write(training_folder / f'{idx}_{i}.wav', audio, 16000)

                    np.save(emb_file, np.mean(embeddings, axis=0) if embeddings else None)
                    with open(label_file, 'w', encoding='utf-8') as f:
                        f.write(cfg.hosts[idx] if idx < len(cfg.hosts) else f'Unknown host {idx}')

                with open(param_file, 'w', encoding='utf-8') as f:
                    f.write('sample_file_count,max_segments,target_files,available_files\n')
                    f.write(f"{global_config.host_match.sample_file_count},"
                            f"{global_config.host_match.per_file_segment_count},"
                            f"{global_config.host_match.target_file_count},{available}")

                satisfied_configs.append(cfg.channel_name_or_term)

        # Final summary
        if satisfied_configs:
            print(f"[*] Configurations satisfied: {', '.join(sorted(satisfied_configs))}")
        else:
            print("[ ] No configurations satisfied.")

        if unsatisfied_configs:
            print(
                f"Configs skipped due to insufficient data (RTTM/WAV pairs): "
                f"{', '.join(sorted(unsatisfied_configs))}"
            )


if __name__ == "__main__":
    main()
