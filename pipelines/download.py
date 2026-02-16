from pathlib import Path

from src import pipeline_helper


if __name__ == '__main__':
    """
    This separates out processes so that we don't have weird coalitions,
    and it frees up models per script.
    """
    root = Path(__file__).parent.parent
    scripts = [
        root / "src" / "get_youtube.py",
        root / "src" / "get_twitch.py",
        root / "src" / "get_patreon.py",
        root / "src" / "download_youtube.py",
        root / "src" / "download_twitch.py",
        root / "src" / "download_patreon.py",
    ]

    for script in scripts:
        pipeline_helper.run_script(script)
