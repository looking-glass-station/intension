import json
from pathlib import Path

import inflect
from tqdm_sound import TqdmSound
from transformers import pipeline

import file_utils
from configs import get_global_config, iter_processing_configs
from file_utils import filter_files_by_stems
from logger import global_logger
from system_config import device


def precompute_topics_with_plurals():
    """
    Adds plural forms to each topic's must_contain list.

    """
    global_config = get_global_config()
    inflection_engine = inflect.engine()

    # Load topic definitions once
    topics_path = global_config.project_root / "confs" / "bias.json"
    with open(topics_path, encoding="utf-8") as f:
        topics = json.load(f)

    for topic, data in topics.items():
        terms = data["terms"]
        must = data.get("must_contain", []) + terms
        must = set(inflection_engine.plural(w) for w in must) | set(must)
        data["terms"] = sorted(set(terms))
        data["must_contain"] = sorted(must)

    configs = [p.stem.replace('_', ' ').title()
               for p in Path(global_config.project_root / 'confs' / 'download_configurations').glob('*.json')]

    topics["Channels"] = {}
    topics["Channels"]["terms"] = sorted(set(configs))
    topics["Channels"]["must_contain"] = sorted(set(configs))

    return topics


def classify_bias_for_transcripts(
        transcript_file,
        bias_file,
        bias_labels_file,
        topics,
        classifier,
):
    """
    Processes a single transcript file for bias classification, outputs CSV and Audacity label.

    Args:
        transcript_file: Path to CSV transcript.
        bias_file: Path to output bias CSV.
        bias_labels_file: Path to output Audacity label file.
        topics: Bias topic dictionary.
        classifier: HuggingFace pipeline.
        progress: TqdmSound progress bar instance.
    """
    rows = file_utils.csv_to_dict(transcript_file)
    if not rows:
        return

    headers = [
        "speaker", "speaker_name", "text",
        "start_time", "end_time", "duration", "intersect_words"
    ]
    for topic, data in topics.items():
        for term in data["terms"]:
            headers.append(f"{term}_supports")

    results = []
    labels_out = []

    for row in rows:
        text = row["text"]
        start = float(row["start_time"])
        end = float(row["end_time"])
        duration = end - start
        if duration <= 5:
            continue

        out = {
            "speaker": row["speaker"],
            "speaker_name": row.get("speaker_name", ""),
            "text": text,
            "start_time": start,
            "end_time": end,
            "duration": duration,
            "intersect_words": ""
        }

        for topic, data in topics.items():
            matched_phrases = [
                must_phrase
                for must_phrase in data["must_contain"]
                if must_phrase.casefold() in text.casefold()
            ]

            if not matched_phrases:
                continue

            for term in data["terms"]:
                hypothesis = f"The author of this text {{}} {term}."
                res = classifier(text, ["supports"], hypothesis_template=hypothesis, multi_label=True)
                scores = {f"{term}_{lbl}": round(res["scores"][i], 2) for i, lbl in enumerate(res["labels"])}
                best = max(scores, key=scores.get)

                if scores[best] > 0:
                    out.update(scores)
                    out["primary_score_supports"] = scores[best]
                    out["primary_label_supports"] = best
                    out["intersect_words"] = ", ".join(sorted(matched_phrases))

                    labels_out.append({
                        "start_time": start,
                        "end_time": end,
                        "text": f"{topic}_{term} - {best}: {scores[best]}"
                    })

        # Only append if any bias scores were found (7 other columns)
        if len(out) > 7:
            results.append(out)

    file_utils.dict_to_csv(bias_file, results, fields=headers)
    file_utils.audacity_writer(bias_labels_file, labels_out)


def main():
    """
    Main entry: scan configs for transcripts, only process files needing bias outputs.
    """
    global_config = get_global_config()

    logger = global_logger("bias_classification")
    logger.info("Starting bias classification.")

    progress = TqdmSound(
        activity_mute_seconds=0,
        dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json")
    )

    classifier = pipeline(
        "zero-shot-classification",
        model="mlburnham/Political_DEBATE_base_v1.0",
        device=device,
        batch_size=8
    )

    topics = precompute_topics_with_plurals()

    for cfg in iter_processing_configs(include_manual=True):
        channel_data_dir = cfg.output_path
        transcripts_dir = channel_data_dir / "transcription_labeled"
        bias_dir = channel_data_dir / "bias"
        audacity_dir = channel_data_dir / "bias_audacity"

        if not transcripts_dir.exists():
            logger.info(f"No transcripts for: {cfg.name} ({cfg.channel_name_or_term})")
            continue

        transcript_files = filter_files_by_stems(transcripts_dir, 'csv',
                                                 [bias_dir, audacity_dir])

        if not transcript_files:
            logger.info("No transcript files")
            continue

        bias_dir.mkdir(parents=True, exist_ok=True)
        audacity_dir.mkdir(parents=True, exist_ok=True)

        # Wrap outer loop in progress bar for per-file progress and richer description
        file_bar = progress.progress_bar(
            transcript_files,
            desc=f"{cfg.name} ({cfg.channel_name_or_term}) - Bias files",
            unit="file",
            total=len(transcript_files),
            leave=True,
            ten_percent_ticks=True,
        )

        for transcript_file in file_bar:
            file_bar.set_description(
                f"{cfg.name} ({cfg.channel_name_or_term}) - {transcript_file.stem}"
            )

            stem = transcript_file.stem
            bias_file = bias_dir / f"{stem}.csv"
            bias_labels_file = audacity_dir / f"{stem}.txt"

            if bias_file.exists() and bias_labels_file.exists():
                continue

            logger.info(f"Processing {transcript_file} for bias...")
            classify_bias_for_transcripts(transcript_file, bias_file, bias_labels_file, topics, classifier)

    logger.info("Bias classification completed.")


if __name__ == "__main__":
    main()
