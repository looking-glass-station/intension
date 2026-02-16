import csv
import re
import warnings
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers.utils import logging as hf_logging

# Suppress noisy logs/warnings from transformers and torch
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

from configs import get_global_config, iter_processing_configs
from system_config import device
from file_utils import filter_files_by_stems, dict_to_csv
from tqdm_sound import TqdmSound
from logger import global_logger


def group_id_to_word(group: str) -> str:
    """
    Convert NER model entity group to a human-readable label.

    Args:
        group (str): Model-generated entity group string (e.g., 'B-PER', 'I-ORG').

    Returns:
        str: Human-readable category (e.g., 'Person', 'Organization', etc.).
    """

    # Named Entity Mapping
    NER_LABEL_MAP = {
        'O': 'Outside',
        '-MISC': 'Miscellaneous',
        '-PER': 'Person',
        '-ORG': 'Organization',
        '-LOC': 'Location'
    }

    for key, value in NER_LABEL_MAP.items():
        if group.endswith(key):
            return value
    return "Unknown"


def get_transcript_subjects(transcript_file: Path, ner_pipeline) -> List[Dict[str, str]]:
    """
    Run NER extraction on a transcript CSV and count subject entities by speaker.

    Args:
        transcript_file (Path): Path to transcript CSV with 'speaker_name' and 'text' fields.
        ner_pipeline: Huggingface NER pipeline object.

    Returns:
        List[Dict[str, str]]: List of dicts containing speaker, word, group, and mention count.
    """
    group_data = defaultdict(lambda: defaultdict(Counter))
    speakers = {}
    sentences = []
    speaker_map = []

    with transcript_file.open(mode="r", encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
        for line in rows:
            text = line['text']
            speaker = line['speaker_name']
            if not text.strip():
                continue
            speakers[speaker] = speaker
            sentences.append(text)
            speaker_map.append(speaker)

    # Batch process sentences for efficient GPU inference
    all_results = ner_pipeline(sentences, batch_size=8)

    for text, speaker, results in zip(sentences, speaker_map, all_results):
        unique_subjects = {
            entity['word']: entity
            for entity in results
            if len(entity['word']) >= 2 and entity['score'] >= 0.5
        }
        for subject in unique_subjects.values():
            word = re.sub(r'[^a-zA-Z\s]', '', subject['word']).strip()
            if re.search(r'[a-zA-Z]', word):
                group = group_id_to_word(subject['entity_group'])
                group_data[speaker][group][word] += 1

    out = []
    for speaker, group_words in group_data.items():
        for group, words in group_words.items():
            for word, count in words.items():
                out.append({
                    'speaker_name': speakers[speaker],
                    'word': word,
                    'count': count,
                    'group': group
                })

    return out


def main():
    """
    Run named entity recognition over labeled transcripts for all configured YouTube channels,
    writing per-speaker entity counts to CSVs in the 'topics' subdirectory.
    """
    logger = global_logger("topics")

    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

    model = model.to(device)

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        grouped_entities=True,
        device=0 if torch.cuda.is_available() else -1
    )

    global_config = get_global_config()

    progress = TqdmSound(
        activity_mute_seconds=0,
        dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json")
    )

    for cfg in iter_processing_configs(include_manual=True):
        topics_path = cfg.output_path / 'topics'
        labeled_path = cfg.output_path / 'transcription_labeled'
        topics_path.mkdir(exist_ok=True, parents=True)

        transcript_files = filter_files_by_stems(labeled_path, 'csv',
                                                 [topics_path, labeled_path])

        bar = progress.progress_bar(
            transcript_files,
            desc=f"{cfg.name} ({cfg.channel_name_or_term}): Extracting Entities",
            unit="file",
            leave=True,
            ten_percent_ticks=True
        )

        for file in bar:
            bar.set_description(f"{cfg.name} ({cfg.channel_name_or_term}) - {file.stem}")

            topics_csv = topics_path / file.name

            if topics_csv.exists():
                continue

            subjects = get_transcript_subjects(file, ner_pipeline)
            dict_to_csv(topics_csv, subjects)

            logger.info(f"{cfg.name} ({cfg.channel_name_or_term}) - {file.stem}")


if __name__ == "__main__":
    main()
