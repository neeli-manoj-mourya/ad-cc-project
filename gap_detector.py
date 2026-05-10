"""
gap_detector.py — STEP 2
Detects silent gaps between dialogues in a movie audio track.

Flow:
  Video → FFmpeg extract audio → Whisper transcription → silence windows
  → filter/merge → List[SilenceGap]
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from config import config
from logger import setup_logger
from models import SilenceGap

log = setup_logger("gap_detector")


# ── Public API ─────────────────────────────────────────────────────────────

def detect_silence_gaps(video_path: Path) -> list[SilenceGap]:
    """
    Main entry point.
    Returns a sorted list of SilenceGap objects safe for narration.
    """
    log.info(f"[GapDetector] Analysing audio from: {video_path.name}")

    audio_path = _extract_audio(video_path)
    dialogue_segments = _transcribe_dialogues(audio_path)
    raw_gaps = _compute_gaps(audio_path, dialogue_segments)
    filtered = _filter_and_merge(raw_gaps)

    log.info(f"[GapDetector] Found {len(filtered)} usable silence gaps.")
    return filtered


# ── Step 1: Extract audio with FFmpeg ─────────────────────────────────────

def _extract_audio(video_path: Path) -> Path:
    """Extract mono 16 kHz WAV from video using FFmpeg."""
    out_path = config.temp_dir / "extracted_audio.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz — optimal for Whisper
        "-acodec", "pcm_s16le",
        str(out_path),
    ]
    log.debug(f"[GapDetector] FFmpeg command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")
    log.info(f"[GapDetector] Audio extracted → {out_path}")
    return out_path


# ── Step 2: Transcribe with Whisper ───────────────────────────────────────

def _transcribe_dialogues(audio_path: Path) -> list[dict]:
    """
    Use OpenAI Whisper to detect dialogue timestamps.
    Falls back to FFmpeg silence detection if Whisper is unavailable.
    Returns list of {"start": float, "end": float} dicts.
    """
    try:
        import whisper  # type: ignore
        log.info("[GapDetector] Loading Whisper model (base)…")
        model = whisper.load_model("base", device="cuda" if config.gpu_enabled else "cpu")
        result = model.transcribe(str(audio_path), word_timestamps=True, verbose=False)

        segments: list[dict] = []
        for seg in result.get("segments", []):
            start = max(0.0, seg["start"] - config.safety_padding)
            end   = seg["end"] + config.safety_padding
            segments.append({"start": start, "end": end})

        log.info(f"[GapDetector] Whisper detected {len(segments)} dialogue segments.")
        return segments

    except ImportError:
        log.warning("[GapDetector] Whisper not available — using FFmpeg silence detection.")
        return _ffmpeg_silence_detection(audio_path)
    except Exception as exc:
        log.error(f"[GapDetector] Whisper error: {exc}. Falling back to FFmpeg.")
        return _ffmpeg_silence_detection(audio_path)


def _ffmpeg_silence_detection(audio_path: Path) -> list[dict]:
    """
    Use FFmpeg silencedetect filter as a fallback.
    Returns *non-silent* (sound) segments as "dialogue" regions.
    """
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-af", f"silencedetect=noise={config.silence_threshold_db}dB:d={config.min_gap_duration}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr

    silence_starts: list[float] = []
    silence_ends:   list[float] = []

    for line in output.splitlines():
        if "silence_start" in line:
            silence_starts.append(float(line.split("silence_start: ")[-1]))
        elif "silence_end" in line:
            parts = line.split("|")
            silence_ends.append(float(parts[0].split("silence_end: ")[-1].strip()))

    # Invert silences → sound regions (dialogue)
    audio_duration = _get_audio_duration(audio_path)
    dialogue: list[dict] = []

    prev = 0.0
    for s_start, s_end in zip(silence_starts, silence_ends):
        if prev < s_start:
            dialogue.append({"start": prev, "end": s_start})
        prev = s_end

    if prev < audio_duration:
        dialogue.append({"start": prev, "end": audio_duration})

    return dialogue


# ── Step 3: Compute silence gaps ──────────────────────────────────────────

def _compute_gaps(audio_path: Path, dialogue_segments: list[dict]) -> list[SilenceGap]:
    """
    Derive silence gaps as the inverse of dialogue segments.
    """
    if not dialogue_segments:
        # No dialogue detected — treat entire file as one gap
        duration = _get_audio_duration(audio_path)
        return [SilenceGap(start=0.0, end=duration, duration=duration)]

    audio_duration = _get_audio_duration(audio_path)
    sorted_segs = sorted(dialogue_segments, key=lambda s: s["start"])

    gaps: list[SilenceGap] = []
    cursor = 0.0

    for seg in sorted_segs:
        gap_start = cursor
        gap_end   = seg["start"]
        gap_dur   = gap_end - gap_start
        if gap_dur >= config.min_gap_duration:
            gaps.append(SilenceGap(start=gap_start, end=gap_end, duration=gap_dur))
        cursor = seg["end"]

    # Final gap after last dialogue
    if audio_duration - cursor >= config.min_gap_duration:
        gaps.append(SilenceGap(
            start=cursor,
            end=audio_duration,
            duration=audio_duration - cursor,
        ))

    return gaps


# ── Step 4: Filter & merge ────────────────────────────────────────────────

def _filter_and_merge(gaps: list[SilenceGap]) -> list[SilenceGap]:
    """
    - Remove gaps shorter than min_gap_duration.
    - Merge gaps that are very close together.
    - Assign sequential IDs.
    """
    if not gaps:
        return []

    # Sort by start time
    gaps = sorted(gaps, key=lambda g: g.start)

    # Merge nearby gaps
    merged: list[SilenceGap] = [gaps[0]]
    for current in gaps[1:]:
        last = merged[-1]
        if current.start - last.end <= config.merge_gap_distance:
            # Extend previous gap
            new_end = current.end
            merged[-1] = SilenceGap(
                start=last.start,
                end=new_end,
                duration=new_end - last.start,
            )
        else:
            merged.append(current)

    # Filter by minimum duration after merge
    filtered = [
        g for g in merged
        if g.usable_duration(config.safety_padding) >= config.min_gap_duration
    ]

    # Assign IDs
    for idx, gap in enumerate(filtered):
        gap.gap_id = idx + 1

    return filtered


# ── Utility ───────────────────────────────────────────────────────────────

def _get_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds using librosa."""
    try:
        return librosa.get_duration(path=str(audio_path))
    except Exception:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
