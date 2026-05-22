"""
generate_cc.py — Extract audio from a video file and save it as MP3
=====================================================================
Usage:
    python generate_cc.py --video <video_file>

Output:
    output/<video_stem>_audio.mp3

The extracted MP3 can then be used as input for closed-caption (CC)
generation tools or any downstream audio processing workflow.
"""

import argparse
import os
import ssl
import subprocess
import sys
from pathlib import Path

# ── SSL patch for corporate/proxy networks ─────────────────────────────────
os.environ["PYTHONHTTPSVERIFY"] = "0"
_orig_ssl = ssl.create_default_context


def _no_verify_ssl(*args, **kwargs):
    ctx = _orig_ssl(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


ssl.create_default_context = _no_verify_ssl
# ───────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output")


def extract_audio(video_path: Path, out_mp3: Path) -> None:
    """Extract the audio track from a video file and save as MP3."""
    print(f"Extracting audio from '{video_path.name}' → '{out_mp3.name}'...")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "2", "-ar", "44100",
        "-codec:a", "libmp3lame", "-q:a", "2",
        str(out_mp3),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: ffmpeg audio extraction failed:\n{result.stderr[-500:]}")
        sys.exit(1)
    print(f"  ✅ Audio MP3 saved: {out_mp3}\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="generate_cc",
        description="Extract audio from a video file as MP3 (for CC generation workflow)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input video file to extract audio from (e.g. movie.mp4)",
    )
    args = parser.parse_args()

    video_path: Path = args.video
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_mp3 = OUTPUT_DIR / f"{video_path.stem}_audio.mp3"
    extract_audio(video_path, out_mp3)


if __name__ == "__main__":
    main()
