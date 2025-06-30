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

## Installation
Install pytorch separately first, I find it heads off dependency issues

`pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118`

`pip3 install inflect transformers tqdm-sound dataclasses-json marshmallow whisperx numpy pillow yt_dlp pathvalidate pandas soundfile librosa resemblyzer pyannote.audio anyascii dateparser isodate scrapetube google-api-python-client sounddevice ffmpeg-python`

If you plan on fetching from Twitch, you'll need Ivan Habunek's library twitch-dl:

`pip3 install twitch-dl`

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


## 📋 Compatibility
Tested on Windows 11\Debian with an AMD CPU and NVIDIA 4070 and on an Intel CPU and NVIDIA 2050.



