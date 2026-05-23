"""
generate_srt.py — Transcribe MP3 audio to Telugu SRT (closed captions)
=====================================================================
Usage:
    python generate_srt.py --audio output/movie_audio.mp3

Prerequisites:
    pip install openai-whisper

Output:
    output/<audio_stem>.srt

This script uses OpenAI Whisper to transcribe Telugu audio with timestamps,
then converts the output into a properly formatted SRT subtitle file.
"""

import argparse
import json
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
TEMP_DIR = Path("temp/whisper_output")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def check_whisper_installed() -> bool:
    """Check if whisper is installed."""
    try:
        import whisper  # type: ignore
        return True
    except ImportError:
        return False


def transcribe_with_whisper(audio_path: Path) -> dict:
    """Transcribe audio using OpenAI Whisper and return result with timestamps."""
    print(f"  Loading Whisper model for Telugu transcription...")
    try:
        import whisper  # type: ignore
    except ImportError:
        print("ERROR: openai-whisper not installed.")
        print("       Install it with: pip install openai-whisper")
        sys.exit(1)

    try:
        # Load model (base is good balance of speed/accuracy)
        model = whisper.load_model("base")
        
        print(f"  Transcribing '{audio_path.name}' (Telugu)...")
        result = model.transcribe(
            str(audio_path),
            language="te",  # Telugu
            task="transcribe",
            verbose=False,
        )
        
        return result
    except Exception as exc:
        print(f"ERROR: Whisper transcription failed: {exc}")
        sys.exit(1)


def generate_srt_from_whisper(result: dict, output_srt: Path) -> None:
    """Convert Whisper JSON output to SRT format."""
    srt_content = []
    subtitle_index = 1
    
    segments = result.get("segments", [])
    if not segments:
        print("WARNING: No segments found in transcription result.")
        return
    
    for segment in segments:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        
        if not text:
            continue
        
        # SRT format: index, timecode, text, blank line
        srt_content.append(f"{subtitle_index}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")  # blank line
        subtitle_index += 1
    
    if not srt_content:
        print("WARNING: No valid subtitles generated.")
        return
    
    srt_text = "\n".join(srt_content)
    output_srt.write_text(srt_text, encoding="utf-8")
    print(f"  ✅ SRT generated: {output_srt}")
    print(f"     Subtitles: {subtitle_index - 1} entries")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="generate_srt",
        description="Transcribe MP3 audio to Telugu SRT (closed captions) using Whisper",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Input MP3 audio file to transcribe (e.g. output/movie_audio.mp3)",
    )
    args = parser.parse_args()

    audio_path: Path = args.audio
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    if not check_whisper_installed():
        print("ERROR: openai-whisper not installed.")
        print("       Install it with: pip install openai-whisper")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Telugu Audio → SRT (Closed Captions) Generator")
    print(f"{'='*60}\n")

    # Transcribe with Whisper
    print(f"  Input: {audio_path}")
    result = transcribe_with_whisper(audio_path)

    # Generate SRT from result
    output_srt = OUTPUT_DIR / f"{audio_path.stem}.srt"
    print(f"\n  Creating SRT file...")
    generate_srt_from_whisper(result, output_srt)

    print(f"\n{'='*60}")
    print(f"  ✅ DONE")
    print(f"  Output SRT: {output_srt}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
