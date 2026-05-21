"""
make_narration.py — Extract video audio + synthesize Telugu narration
=====================================================================
Usage:
    python make_narration.py                          # synthesize from narration.txt only
    python make_narration.py --video movie.mp4       # extract audio MP3 + synthesize narration

Outputs:
    - output/<video>_audio.mp3            (from --video flag)
    - output/narration_final.mp3          (from narration.txt via Edge TTS)

Format of narration.txt:
    gap_start_seconds | gap_end_seconds | Telugu text

Example:
    0.0 | 2.0 | తెరపై టైటిల్ కార్డులు కనిపిస్తాయి.
    17.0 | 38.0 | వంశీ కేఫేలో ఒంటరిగా కూర్చొని సంజనా కోసం ఎదురుచూస్తుంటాడు.
"""

import argparse
import asyncio
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

# ── Config ─────────────────────────────────────────────────────────────────
INPUT_FILE   = Path("narration.txt")          # your Telugu script file
OUTPUT_DIR   = Path("output")
TEMP_DIR     = Path("temp/tts_manual")
VOICE        = "te-IN-MohanNeural"
SPEED        = "+20%"                         # 1.2x speed
PITCH        = "-2Hz"
VOLUME       = "+5%"
# ───────────────────────────────────────────────────────────────────────────


def parse_narration_file(path: Path) -> list[dict]:
    """Parse narration.txt into list of {start, end, text} dicts."""
    entries = []
    for line_num, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            print(f"  [SKIP] Line {line_num}: wrong format (need: start | end | text)")
            continue
        try:
            start = float(parts[0])
            end   = float(parts[1])
            text  = parts[2]
        except ValueError:
            print(f"  [SKIP] Line {line_num}: start/end must be numbers")
            continue
        if not text.strip():
            print(f"  [SKIP] Line {line_num}: empty text")
            continue
        entries.append({"start": start, "end": end, "text": text, "line": line_num})
    return entries


async def synthesize_one(text: str, out_mp3: Path) -> bool:
    """Synthesize one Telugu sentence to MP3 using Edge TTS."""
    try:
        import edge_tts  # type: ignore
        communicate = edge_tts.Communicate(
            text=text,
            voice=VOICE,
            rate=SPEED,
            pitch=PITCH,
            volume=VOLUME,
        )
        await communicate.save(str(out_mp3))
        return out_mp3.exists() and out_mp3.stat().st_size > 0
    except ImportError:
        print("ERROR: edge-tts not installed. Run: pip install edge-tts")
        sys.exit(1)
    except Exception as exc:
        print(f"  [ERROR] TTS failed: {exc}")
        return False


def get_duration(mp3: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp3)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def merge_clips_to_mp3(clips: list[dict], total_duration: float, out_path: Path) -> None:
    """Place each TTS clip at its gap start time on a silent timeline, export as MP3."""
    # Build ffmpeg filter_complex to place each clip at exact timestamp
    inputs = ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={total_duration:.3f}"]
    filter_parts = ["[0:a]"]
    labels = ["base"]

    for i, clip in enumerate(clips):
        inputs += ["-i", str(clip["mp3"])]
        delay_ms = int(clip["start"] * 1000)
        filter_parts.append(
            f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[a{i}];"
        )
        labels.append(f"[a{i}]")

    # Mix base silence + all delayed clips
    mix_inputs = "[base]" + "".join(f"[a{i}]" for i in range(len(clips)))
    filter_complex = (
        "[0:a]acopy[base];"
        + "".join(
            f"[{i+1}:a]adelay={int(c['start']*1000)}|{int(c['start']*1000)}[a{i}];"
            for i, c in enumerate(clips)
        )
        + mix_inputs
        + f"amix=inputs={len(clips)+1}:normalize=0[out]"
    )

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", filter_complex, "-map", "[out]",
           "-codec:a", "libmp3lame", "-q:a", "2", str(out_path)]
    )

    print(f"\n  Merging {len(clips)} clips into {out_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg merge failed:\n{result.stderr[-500:]}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(prog="make_narration", description="Extract video audio MP3 and/or synthesize narration from narration.txt")
    parser.add_argument("--video", type=Path, help="Optional input video file to extract audio MP3")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Extract video audio if --video was provided ────────────────────────
    if args.video:
        video_path: Path = args.video
        if not video_path.exists():
            print(f"ERROR: video not found: {video_path}")
            sys.exit(1)
        
        out_mp3 = OUTPUT_DIR / f"{video_path.stem}_audio.mp3"
        print(f"Extracting audio from {video_path.name} → {out_mp3.name}...")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "2", "-ar", "44100",
            "-codec:a", "libmp3lame", "-q:a", "2",
            str(out_mp3),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: ffmpeg audio extraction failed:\n{result.stderr[-500:]}")
        else:
            print(f"  ✅ Audio MP3 saved: {out_mp3}\n")

    if not INPUT_FILE.exists():
        # Create a blank narration.txt for the user to fill in
        blank = "# Telugu Audio Description Script\n# Format: gap_start_seconds | gap_end_seconds | Telugu text\n# Lines starting with # are ignored.\n# Paste your lines below:\n"
        INPUT_FILE.write_text(blank, encoding="utf-8")
        print(f"✅ Created blank '{INPUT_FILE}'.")
        print(f"   Paste your Telugu lines into it, then run this script again.")
        return

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print("  Telugu Narration → MP3 Converter")
    print(f"{'='*55}")

    entries = parse_narration_file(INPUT_FILE)
    if not entries:
        print("ERROR: No valid entries found in narration.txt")
        sys.exit(1)

    print(f"  Found {len(entries)} narration entries.\n")

    # Synthesize each entry
    clips = []
    for entry in entries:
        mp3 = TEMP_DIR / f"clip_{entry['line']:03d}_{int(entry['start'])}.mp3"
        print(f"  [{entry['line']:02d}] {entry['start']:.1f}s → synthesizing: {entry['text'][:50]}...")

        success = asyncio.run(synthesize_one(entry["text"], mp3))
        if not success:
            print(f"       ⚠ Skipping — TTS failed")
            continue

        dur = get_duration(mp3)
        gap_dur = entry["end"] - entry["start"]
        print(f"       ✅ {dur:.2f}s TTS | {gap_dur:.1f}s gap available")

        if dur > gap_dur:
            print(f"       ⚠ WARNING: narration ({dur:.2f}s) is longer than gap ({gap_dur:.1f}s) — will overlap")

        clips.append({**entry, "mp3": mp3, "tts_duration": dur})

    if not clips:
        print("\nERROR: No clips were synthesized.")
        sys.exit(1)

    # Total duration = last gap end time + small buffer
    total_duration = max(c["end"] for c in clips) + 5.0
    out_mp3 = OUTPUT_DIR / "narration_final.mp3"

    merge_clips_to_mp3(clips, total_duration, out_mp3)

    print(f"\n{'='*55}")
    print(f"  ✅ DONE — {out_mp3}")
    print(f"  Total clips : {len(clips)}")
    print(f"  Duration    : {total_duration:.1f}s")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
 
