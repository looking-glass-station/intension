# Intention

Intention is a configuration driven Youtube\Twitch downloader, sound file diarizer, transcriber and automatic host match and labeler. It also contains auto-topic classifiers and bias detection. 

It uses a collection of preexisting tools including:
* Torch
* ffmpeg
* transformers
* whisperx
* resemblyzer
* pyannote.audio
* yt_dlp
* Twitch-dl
* Patreon
* Manual files in a directory

## Installation

This project is managed with [uv](https://docs.astral.sh/uv/). The environment is
defined by `pyproject.toml` + `uv.lock`; `.venv` is a disposable build artifact
regenerated from the lockfile.

```powershell
# from project root
uv sync
```

That creates `.venv` with CPython 3.11 (pinned in `.python-version`) and installs
the locked dependencies, including the CUDA 11.8 PyTorch build pulled from
PyTorch's own package index (configured under `[tool.uv.sources]`).

Run everything through uv so it uses the project environment:

```powershell
uv run python -m pipelines.all
uv run python src/intention_cli.py process C:\path\to\audio.mp3 --output-dir C:\out
```

### Python version

Dependency hell! Use **Python 3.11** (3.10 also works). Python 3.12+ triggers
conflicts across the `whisperx` / `pyannote` / `av` / `ctranslate2` stack and fails
to build on Windows, so `requires-python` is pinned `>=3.11,<3.12`.

### JS runtime for YouTube downloads

`yt-dlp` needs a JavaScript runtime to solve YouTube's EJS player challenges
(`yt-dlp-ejs`). `src/download_youtube.py` looks for one, in order: `YT_DLP_JS` /
`DENO_EXE` env vars, `deno` on `PATH`, `~/.deno/bin/deno.exe`, then
`.venv/Scripts/deno.exe`. Node (`C:\Program Files\nodejs\node.exe`) also works.

Install Deno (2.9+) once, per user — it survives `.venv` rebuilds there:

```powershell
irm https://deno.land/install.ps1 | iex     # installs to ~/.deno/bin, adds it to PATH
```

Or drop `deno.exe` from https://github.com/denoland/deno/releases into
`~/.deno/bin/`. No `deno.json` or npm setup is needed — the project only uses Deno
as yt-dlp's challenge interpreter.

## 🔑 Generate authentication keys
You will need to generate authentication keys for: 
### Google's YouTube API 
https://developers.google.com/youtube/v3/getting-started
(You can use scrapetube instead but it sometimes fails to fetch all data)

### Huggingface
https://huggingface.co/docs/hub/en/security-tokens

For:
* pyannote/segmentation
* pyannote/speaker-diarization
* pyannote/segmentation-3.0
* pyannote/speaker-diarization-3.1

 ## Transcription:
 * Measured median transcription pipeline is 56.7x realtime:
 * About 16% faster than normal faster-whisper implementations
 * About 39% faster than large-v3-turbo


## Host Match
This project was written with content involving a common speaker across multiple recordings in mind, for example, an interviewer talking with many different guests. It:

1) Scans your configured channels for paired audio (WAV) and diarization (RTTM) files, skipping any that haven’t changed or don’t yet have enough material.
2) Extracts a small, representative sample of each speaker’s speech  and concatenates them into one composite clip.
3) Re-diarizes that combined clip to isolate the most prominent speaker (host(s)).
4) Embeds each host’s voice into and saves those embeddings.

This takes a lot of time-consuming work away from labeling content and is repeatable and accurate.


## ⌨️ CLI
I have included a simple CLI tool
### Process a local file with output directory:
`intention_cli.py process /path/to/audio.mp3 --output-dir /path/to/output`

### Download a YouTube video and convert only, with output directory:
`intention_cli.py download uHlFuPqoIVA --output-dir /path/to/output`

### Download, diarize, and transcribe:
`intention_cli.py download uHlFuPqoIVA --output-dir /path/to/output --diarize --transcribe`


## 🗃️ Configuration Files
### Global.json
<pre>
 {
  "data_directory": "data", # data output directory
  "keep_raw_downloads": false, # to keep the high quality downloads, false is correct unless you want to do something else with the raw data later on. 
  "min_free_disk_space_gb": 5, # space in gigabytes to leave free on disk
  "youtube": {
    "cookies_from": "firefox" # the browser you use for YouTube
  },
  "max_workers_base": 2, # the base number of threads to use for work, this auto scales, 2 is probably about right
  "host_match": {
    "sample_file_count": 90, # the number of files needed to compute auto host match
    "per_file_segment_count": 6, # the number of speaker turns to grab per file
    "target_file_count": 20, # the target number of valid sample files (because some samples won't be valid)
    "overlap_tolerance": 0.25 # speakers talking over each other
  }
}
</pre>

### Sound.json
Sound.json is for my own https://pypi.org/project/tqdm-sound/ which I use when I want to keep an ear on my code when I'm doing something else. You can just mute it:
<pre>
{
  "is_muted": true
}
</pre>

## Download configurations
<pre>
{
    "youtube": [
    {
      "channel_name_or_term": "NASA", # YouTube Channel named, playlist name etc
      "use_google_api": true, # set to false to use scrapetube
      "max_date": null,
      "is_playlist": true,
      "is_search": false,
      "min_length_mins": 1,
      "must_contain": ["Cosmic Dawn"],
      "must_exclude": [],
      "hosts": [
        "Brian Lamb"
      ],
      "overwrite": false,
      "get_static_videos": true,
      "audio_only": true,
      "guest_searches": [],
      "episode_prefix": null,
      "guest_replace": []
    }
  ],"twitch": [
    {
      "channel_name_or_term": "nasa",
      "is_clips": false,
      "max_date": null,
      "must_contain": ["NASA’s VIPER Moon Rover"],
      "must_exclude": [],
      "hosts": [
        "Lara", "Erica", "Anton", "DW Wheeler"
      ],
      "overwrite": true,
      "min_length_mins": 5,
      "get_static_videos": true,
      "audio_only": true,
      "guest_searches": [],
      "episode_prefix": null,
      "guest_replace": []
    }
  ]
}
</pre>


## 🚩Pipelines
Pipelines are separated into downloads and graphics card intensive work so that they can be run simultaneously. 
It takes about 20 seconds to load most GPU tasks into memory so it doesn't make sense to run tasks sequentially.

Download:

`python -m pipelines.download`

Process data:

`python -m pipelines.data`

Process all:

`python -m pipelines.all`

## 📋 Compatibility
Tested on Windows 11\Debian with an AMD CPU and NVIDIA 4070 and on an Intel CPU and NVIDIA 2050.


