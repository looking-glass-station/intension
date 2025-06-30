import argparse
import sys
from pathlib import Path

from diarize import Diarizer
from download_youtube import download_and_convert_by_id
from to_wav import convert_to_wav
from transcribe import Transcriber


class ExampleArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser subclass that shows usage examples on error.
    """

    def error(self, message):
        # Print usage, error message, and examples
        self.print_usage(sys.stderr)
        examples = (
            f"  # Process a local file with output directory:\n"
            f"  {self.prog} process /path/to/audio.mp3 --output-dir /path/to/output\n"
            f"  # Download a YouTube video and convert only, with output directory:\n"
            f"  {self.prog} download yfnJY2ydvc0 --output-dir /path/to/output\n"
            f"  # Download, diarize, and transcribe:\n"
            f"  {self.prog} download yfnJY2ydvc0 --output-dir /path/to/output --diarize --transcribe\n"
        )
        self.exit(2, f"{self.prog}: error: {message}\n\nExamples:\n{examples}")


def main():
    """
    Entry point for the audio processing pipeline.
    Supports processing local audio files or downloading YouTube videos by ID.
    Requires an output directory for all operations.
    """
    parser = ExampleArgumentParser(
        description="Download and/or process audio files: convert to WAV, diarize, and transcribe."
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-commands')

    # Subparser for processing local files
    process_parser = subparsers.add_parser('process', help='Process a local audio file.')
    process_parser.add_argument(
        'input_file',
        type=Path,
        help="Path to the input audio file (e.g., .m4a, .mp3)."
    )
    process_parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help="Directory where output WAV and transcripts will be stored."
    )

    # Subparser for downloading YouTube videos
    download_parser = subparsers.add_parser('download', help='Download a YouTube video by ID and convert to WAV.')
    download_parser.add_argument(
        'video_id',
        help="YouTube video ID or URL to download (e.g., 'uHlFuPqoIVA' or full URL)."
    )
    download_parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help="Directory where downloaded files and WAV will be stored."
    )
    download_parser.add_argument(
        '--diarize',
        action='store_true',
        help="Run speaker diarization after conversion."
    )
    download_parser.add_argument(
        '--transcribe',
        action='store_true',
        help="Run transcription after diarization."
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir: Path = getattr(args, 'output_dir', None)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == 'process':
        input_path: Path = args.input_file
        if not input_path.exists():
            parser.error(f"Input file does not exist: {input_path}")

        wav_path = output_dir / f"{input_path.stem}.wav"
        print(f"[*] Converting {input_path} to WAV at {wav_path}")
        convert_to_wav(input_path, wav_path)

        print(f"[*] Running diarization on {wav_path}")
        diarizer = Diarizer()
        diarizer.diarize_file(wav_path)

        print(f"[*] Transcribing segments from RTTM for {wav_path}")
        transcriber = Transcriber()
        transcriber.transcribe_from_rttm(wav_path)

    elif args.command == 'download':
        video_id: str = args.video_id

        print(f"[*] Downloading and converting YouTube ID {video_id} to WAV in {output_dir}")
        wav_path = download_and_convert_by_id(video_id, output_dir)
        print(f"[+] WAV file created at {wav_path}")

        if args.diarize:
            print(f"[*] Running diarization on {wav_path}")
            diarizer = Diarizer()
            diarizer.diarize_file(wav_path)

        if args.transcribe:
            print(f"[*] Transcribing segments from RTTM for {wav_path}")
            transcriber = Transcriber()
            transcriber.transcribe_from_rttm(wav_path)

    else:
        # Should not happen due to required=True, but just in case
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':  # pragma: no cover
    main()
