import os
import shutil
import subprocess
import tempfile
from typing import Dict, Any

import numpy as np
from PIL import Image


def video_is_static_from_info(
        info: Dict[str, Any],
        n_samples: int = 10,
        max_height: int = 144,
        diff_thresh: float = 10.0
) -> bool:
    """
    Determine if a video is static by sampling frames and computing mean pixel differences.

    Args:
        info (Dict[str, Any]): Metadata dict from yt-dlp with 'duration' and 'formats'.
        n_samples (int): Number of frames to sample evenly across video duration.
        max_height (int): Maximum frame height (px) to request for sampling.
        diff_thresh (float): Threshold for mean grayscale pixel difference to consider static.

    Returns:
        bool: True if the maximum observed frame difference is below diff_thresh.

    Raises:
        RuntimeError: If video duration or suitable formats cannot be determined.
    """
    duration = info.get('duration')
    if not duration:
        raise RuntimeError("Could not determine video duration")

    formats = [
        f for f in info['formats']
        if f.get('vcodec') != 'none' and f.get('height', 1) <= max_height
    ]
    if not formats:
        raise RuntimeError(f"No video formats <= {max_height}px")

    best = max(formats, key=lambda f: f['height'])
    video_url = best['url']
    tmpdir = tempfile.mkdtemp(prefix="yt_frames_")

    try:
        times = [(i + 1) * duration / (n_samples + 1) for i in range(n_samples)]
        ref_frame = None
        max_diff = 0.0
        for i, t in enumerate(times):
            out_path = os.path.join(tmpdir, f"frame_{i:02d}.jpg")
            subprocess.run([
                "ffmpeg",
                "-ss", f"{t:.3f}",
                "-i", video_url,
                "-frames:v", "1",
                "-q:v", "2",
                "-y", "-loglevel", "error",
                out_path
            ], check=True)
            img = Image.open(out_path).convert("L")
            arr = np.array(img, dtype=np.float32)
            if i == 0:
                ref_frame = arr
                max_diff = 0.0
            else:
                diff = np.abs(ref_frame - arr).mean()
                max_diff = max(max_diff, diff)
        return max_diff < diff_thresh

    finally:
        shutil.rmtree(tmpdir)
