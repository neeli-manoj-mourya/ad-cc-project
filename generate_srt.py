"""
generate_srt.py — Transcribe MP3 audio to Telugu SRT (closed captions)
=====================================================================
Usage:
    python generate_srt.py --audio output/movie_audio.mp3

Prerequisites:
    pip install faster-whisper

Output:
    output/<audio_stem>.srt

This script uses faster-whisper (large-v3 model) to transcribe Telugu audio
with accurate word-level timestamps, then writes a properly formatted SRT file.
"""

import argparse
import os
import ssl
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

# Model to use — large-v3 gives best Telugu accuracy.
# Change to "medium" if you are low on RAM (4 GB+).
WHISPER_MODEL = "large-v3"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_to_srt(audio_path: Path, output_srt: Path, model_name: str) -> None:
    """Transcribe Telugu audio using faster-whisper and write an SRT file."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        print("ERROR: faster-whisper not installed.")
        print("       Run: pip install faster-whisper")
        sys.exit(1)

    print(f"  Loading faster-whisper model '{model_name}'...")
    print("  (First run will download the model ~3 GB — please wait)\n")

    # device="auto" uses GPU if available, otherwise CPU
    # compute_type="int8" reduces RAM and speeds up CPU inference
    model = WhisperModel(model_name, device="auto", compute_type="int8")

    print(f"  Transcribing '{audio_path.name}' in Telugu...")
    segments, info = model.transcribe(
        str(audio_path),
        language="te",           # Telugu
        task="transcribe",
        beam_size=5,             # better accuracy than default beam_size=1
        vad_filter=True,         # skip silence — cleaner output
        vad_parameters={"min_silence_duration_ms": 500},
    )

    print(f"  Detected language: {info.language} "
          f"(confidence: {info.language_probability:.0%})\n")

    # consume generator now so progress is visible
    segments = list(segments)

    srt_lines: list[str] = []
    subtitle_index = 1

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        start = format_timestamp(segment.start)
        end   = format_timestamp(segment.end)
        srt_lines.append(f"{subtitle_index}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")   # blank line between entries
        subtitle_index += 1

    if subtitle_index == 1:
        print("WARNING: No subtitles were generated. Check the audio file.")
        return

    output_srt.write_text("\n".join(srt_lines), encoding="utf-8")
    print(f"  SRT generated: {output_srt}")
    print(f"  Total subtitles: {subtitle_index - 1}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="generate_srt",
        description="Transcribe MP3 audio to Telugu SRT (closed captions) using faster-whisper",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Input MP3 audio file (e.g. output/movie_audio.mp3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=WHISPER_MODEL,
        help=f"Whisper model to use (default: {WHISPER_MODEL}). "
             "Options: tiny, base, small, medium, large-v2, large-v3",
    )
    args = parser.parse_args()

    audio_path: Path = args.audio
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_srt = OUTPUT_DIR / f"{audio_path.stem}.srt"

    print(f"\n{'='*60}")
    print("  Telugu Audio → SRT (Closed Captions) Generator")
    print(f"{'='*60}\n")
    print(f"  Input : {audio_path}")
    print(f"  Model : {args.model}")
    print(f"  Output: {output_srt}\n")

    transcribe_to_srt(audio_path, output_srt, args.model)

    print(f"\n{'='*60}")
    print(f"  DONE  →  {output_srt}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
