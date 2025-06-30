import logging
from pathlib import Path
from typing import Union, Optional

import ffmpeg

logger = logging.getLogger(__name__)


def convert_to_wav(
        input_file: Union[Path, str],
        output_file: Optional[Union[Path, str]] = None,
) -> Path:
    """
    Convert any audio/video file to a mono 16 kHz WAV.

    :param input_file: Path (or string) to the source audio or video file.
    :param output_file: Optional Path (or string) for the output WAV; defaults to the same name with a `.wav` suffix.
    :return: Path to the resulting WAV file.
    :raises FileNotFoundError: If `input_file` doesn't exist.
    :raises RuntimeError: If ffmpeg reports an error during conversion.
    """
    inp = Path(input_file)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    out = Path(output_file) if output_file is not None else inp.with_suffix(".wav")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Return early if the WAV already exists
    if out.exists():
        return out

    # Build ffmpeg command for PCM 16-bit WAV
    stream = (
        ffmpeg
        .input(str(inp))
        .output(
            str(out),
            acodec="pcm_s16le",
            ar="16000",
            ac="1",
            format="wav",
        )
    )

    try:
        stream.run(overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        stderr = getattr(e, 'stderr', None)
        err_msg = stderr.decode(errors="ignore").strip() if stderr else str(e)
        logging.error("ffmpeg conversion failed: %s", err_msg)
        raise RuntimeError(f"Failed to convert {inp} to WAV: {err_msg}") from e

    return out
