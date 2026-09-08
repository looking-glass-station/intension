import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any

import ollama
from tqdm_sound import TqdmSound

import file_utils
from configs import get_global_config, iter_processing_configs
from file_utils import filter_files_by_stems
from logger import global_logger


def load_bias_config() -> dict:
    global_config = get_global_config()
    path = global_config.project_root / "confs" / "bias.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def add_channel_topic(topics: dict) -> None:
    global_config = get_global_config()
    configs_dir = global_config.project_root / "confs" / "download_configurations"
    channel_names = sorted(
        p.stem.replace("_", " ").title()
        for p in configs_dir.glob("*.json")
    )
    if channel_names:
        topics["Channels"] = {
            "description": "References to other media channels or content creators being analyzed in this project.",
            "must_contain": [name.lower() for name in channel_names],
        }


def segment_matches_any_topic(text: str, topics: dict) -> bool:
    text_lower = text.casefold()
    for data in topics.values():
        for keyword in data.get("must_contain", []):
            if keyword.casefold() in text_lower:
                return True
    return False


SYSTEM_PROMPT_TEMPLATE = """You are a media bias analyst. You analyze transcript segments from political commentary and podcast content.

For each segment, determine whether the speaker exhibits bias — implicit or explicit — toward any of the topics listed below. Bias includes: stereotyping, dehumanization, hostility, dismissiveness, dog-whistles, loaded framing, or consistent one-sided characterization of a group.

CRITICAL DISTINCTIONS you must make:
- Distinguish between insulting a SPECIFIC PERSON vs. expressing bias toward an entire GROUP (e.g., calling Ben Shapiro a "pig" is a personal insult, not antisemitism)
- Distinguish between criticizing a GOVERNMENT'S POLICIES vs. bias against the PEOPLE of that nation/ethnicity
- Distinguish between REPORTING ON or DISCUSSING a topic vs. EXPRESSING BIAS about it
- Distinguish between QUOTING someone else's biased statement (to critique it) vs. ENDORSING that bias

Topics to analyze:
{topic_definitions}

You will receive numbered transcript segments with speaker names. For each segment where you detect bias, output a JSON object. If a segment contains no bias, omit it entirely.

Respond with ONLY a JSON array (no markdown fencing, no commentary). Each element must have:
- "segment": the segment number (integer)
- "topic": which topic the bias relates to (must match a topic name above)
- "stance": "negative", "positive", or "mixed" — the speaker's attitude toward the subject group
- "target": who/what the language is specifically directed at (e.g., "Ben Shapiro", "Jewish people", "Israeli government", "Republicans")
- "score": confidence from 0.0 to 1.0 that this constitutes genuine bias (not mere discussion)
- "reasoning": one sentence explaining your classification

A segment may produce multiple entries if bias toward different topics is detected. If NO segments contain bias, respond with an empty array: []"""


def build_system_prompt(topics: dict) -> str:
    topic_lines = []
    for name, data in topics.items():
        topic_lines.append(f"- **{name}**: {data['description']}")
    return SYSTEM_PROMPT_TEMPLATE.format(topic_definitions="\n".join(topic_lines))


def build_user_message(segments: List[Dict[str, Any]], start_index: int) -> str:
    parts = []
    for i, seg in enumerate(segments):
        idx = start_index + i + 1
        speaker = seg.get("speaker_name", seg.get("speaker", "Unknown"))
        text = seg["text"]
        parts.append(f"[Segment {idx}] Speaker: {speaker}\n{text}")
    return "Analyze these transcript segments for bias:\n\n" + "\n\n---\n\n".join(parts)


def parse_response(response_text: str, logger=None) -> List[Dict[str, Any]]:
    text = response_text.strip()
    # Strip markdown code fencing if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        results = json.loads(text)
        if isinstance(results, list):
            return results
        # Models often wrap arrays in an object like {"results": [...]}
        if isinstance(results, dict):
            for value in results.values():
                if isinstance(value, list):
                    return value
            # Single finding returned as a bare object
            if "segment" in results:
                return [results]
    except json.JSONDecodeError:
        if logger:
            logger.warning(f"Failed to parse JSON response: {text[:200]}")
    return []


def classify_batch(
        system_prompt: str,
        segments: List[Dict[str, Any]],
        start_index: int,
        model: str,
        logger,
) -> List[Dict[str, Any]]:
    user_message = build_user_message(segments, start_index)

    for attempt in range(3):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                format="json",
                # Deterministic: this is structured classification, and without
                # this gemma3 samples, so findings drift on every re-run.
                options={"temperature": 0},
            )
            raw = response["message"]["content"]
            parsed = parse_response(raw, logger)
            logger.info(f"Batch at index {start_index}: {len(segments)} segments -> {len(parsed)} findings")
            if not parsed and raw.strip() not in ("[]", "{}"):
                logger.warning(f"Unparsed response: {raw[:500]}")
            return parsed

        except ollama.ResponseError as e:
            logger.error(f"Ollama error on attempt {attempt + 1}/3: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                return []
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}/3: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                return []

    return []


def classify_bias_for_transcript(
        transcript_file: Path,
        bias_file: Path,
        bias_labels_file: Path,
        topics: dict,
        system_prompt: str,
        model: str,
        batch_size: int,
        min_duration: float,
        logger,
        progress=None,
        file_desc: str = "",
):
    rows = file_utils.csv_to_dict(transcript_file)
    if not rows:
        return

    headers = [
        "speaker", "speaker_name", "text",
        "start_time", "end_time", "duration",
        "topic", "stance", "target", "score", "reasoning",
    ]

    # Filter to segments worth analyzing
    valid_segments = []
    for row in rows:
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        start = float(row["start_time"])
        end = float(row["end_time"])
        if (end - start) <= min_duration:
            continue
        if not segment_matches_any_topic(text, topics):
            continue
        valid_segments.append(row)

    if not valid_segments:
        file_utils.dict_to_csv(bias_file, [], fields=headers)
        file_utils.audacity_writer(bias_labels_file, [])
        return

    # Process in batches
    all_findings: List[Dict[str, Any]] = []
    num_batches = (len(valid_segments) + batch_size - 1) // batch_size

    batch_iter = range(0, len(valid_segments), batch_size)
    if progress:
        batch_iter = progress.progress_bar(
            batch_iter,
            desc=f"  {file_desc} batches",
            unit="batch",
            total=num_batches,
            leave=False,
        )

    for batch_start in batch_iter:
        batch = valid_segments[batch_start:batch_start + batch_size]
        findings = classify_batch(system_prompt, batch, batch_start, model, logger)

        for finding in findings:
            seg_idx = finding.get("segment")
            if not isinstance(seg_idx, int):
                continue
            # Try global numbering first (segments labeled batch_start+1 .. batch_start+N)
            local_idx = seg_idx - batch_start - 1
            # Fall back to per-batch numbering (model used 1..N within this batch)
            if local_idx < 0 or local_idx >= len(batch):
                local_idx = seg_idx - 1
            if local_idx < 0 or local_idx >= len(batch):
                continue

            row = batch[local_idx]
            start = float(row["start_time"])
            end = float(row["end_time"])

            all_findings.append({
                "speaker": row.get("speaker", ""),
                "speaker_name": row.get("speaker_name", ""),
                "text": row.get("text", ""),
                "start_time": start,
                "end_time": end,
                "duration": round(end - start, 3),
                "topic": finding.get("topic", ""),
                "stance": finding.get("stance", ""),
                "target": finding.get("target", ""),
                "score": finding.get("score", 0),
                "reasoning": finding.get("reasoning", ""),
            })

    file_utils.dict_to_csv(bias_file, all_findings, fields=headers)

    labels_out = [
        {
            "start_time": f["start_time"],
            "end_time": f["end_time"],
            "text": f"{f['topic']} ({f['stance']}) -> {f['target']}: {f['score']}",
        }
        for f in all_findings
    ]
    file_utils.audacity_writer(bias_labels_file, labels_out)


def main():
    global_config = get_global_config()
    logger = global_logger("bias_classification")
    logger.info("Starting bias classification.")

    bias_config = load_bias_config()
    topics = bias_config["topics"]
    add_channel_topic(topics)

    model = bias_config.get("model", "gemma3:12b")
    batch_size = bias_config.get("batch_size", 10)
    min_duration = bias_config.get("min_segment_duration", 5)

    # Verify Ollama is running and model is available
    try:
        ollama.show(model)
    except ollama.ResponseError:
        print(f"ERROR: Model '{model}' not found. Run: ollama pull {model}")
        logger.error(f"Model '{model}' not available in Ollama.")
        return
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama. Is it running? ({e})")
        logger.error(f"Cannot connect to Ollama: {e}")
        return

    system_prompt = build_system_prompt(topics)

    progress = TqdmSound(
        activity_mute_seconds=0,
        dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json"),
    )

    for cfg in iter_processing_configs(include_manual=True):
        channel_data_dir = cfg.output_path
        transcripts_dir = channel_data_dir / "transcription_labeled"
        bias_dir = channel_data_dir / "bias"
        audacity_dir = channel_data_dir / "bias_audacity"

        if not transcripts_dir.exists():
            logger.info(f"No transcripts for: {cfg.name} ({cfg.channel_name_or_term})")
            continue

        transcript_files = filter_files_by_stems(
            transcripts_dir, "csv", [bias_dir, audacity_dir]
        )

        if not transcript_files:
            logger.info("No transcript files")
            continue

        bias_dir.mkdir(parents=True, exist_ok=True)
        audacity_dir.mkdir(parents=True, exist_ok=True)

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
            classify_bias_for_transcript(
                transcript_file, bias_file, bias_labels_file,
                topics, system_prompt, model,
                batch_size, min_duration, logger,
                progress=progress,
                file_desc=transcript_file.stem,
            )

    logger.info("Bias classification completed.")


if __name__ == "__main__":
    main()
