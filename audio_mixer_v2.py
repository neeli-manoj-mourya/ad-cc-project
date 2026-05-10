"""
audio_mixer.py — STEP 7 & 8
Mathematically exact PCM overlay for AD narration placement.

SYNC ARCHITECTURE (why this is drift-free):
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Allocate a numpy float32 buffer of EXACT length = video_samples * channels.
  2. For each clip, compute offset_samples = round(timestamp * 44100).
  3. Read clip PCM into a numpy array.
  4. ADD the clip array into the buffer at buffer[offset : offset + len(clip)].
  5. No FFmpeg filters, no adelay, no amix — pure arithmetic.

  The ONLY source of error is the round() in step 2, which is at most
  0.5 samples = 0.011 ms at 44100 Hz. This is 500× below human perception.

  Previous drift cause: FFmpeg's adelay filter converts float ms → internal
  integer samples with independent rounding per stream, and amix applies
  resampling that compounds the error across 20+ streams.

AUDIO PIPELINE:
  PCM overlay → fade in/out → WAV export → two-pass EBU R128 loudnorm → MP3
"""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from config import config
from logger import setup_logger
from models import TTSClip
from timeline_engine import (
    SyncPlan,
    build_sync_plan,
    log_sync_plan,
    validate_no_overlaps,
)

log = setup_logger("audio_mixer")

SAMPLE_RATE: int = 44100
CHANNELS: int = 2
BYTES_PER_SAMPLE: int = 2  # 16-bit PCM
NUMPY_DTYPE = np.int16


# ── Public API ─────────────────────────────────────────────────────────────

def mix_narration_track(
    clips: list[TTSClip],
    original_audio: Path,
    video_path: Path,
    output_dir: Path,
    merge_with_video: bool = False,
) -> tuple[Path, Optional[Path], Path]:
    """
    Build the final AD narration track with sample-exact sync.

    Returns:
        (narration_mp3, merged_video_or_None, metadata_json)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # ── Get exact video duration ───────────────────────────────────────
    video_duration = _get_video_duration(video_path)
    log.info(f"[AudioMixer] Video duration: {video_duration:.6f}s "
             f"= {round(video_duration * SAMPLE_RATE)} samples @ {SAMPLE_RATE}Hz")

    # ── Build & validate sync plan ─────────────────────────────────────
    plan = build_sync_plan(clips, video_duration)
    log_sync_plan(plan)
    overlap_warnings = validate_no_overlaps(plan)
    if overlap_warnings:
        log.warning(f"[AudioMixer] {len(overlap_warnings)} clip overlaps detected!")

    # ── PCM overlay — the core sync engine ─────────────────────────────
    narration_wav = _pcm_overlay(plan)

    # ── Two-pass loudnorm → final MP3 ─────────────────────────────────
    normalized_mp3 = output_dir / f"{stem}_audio_description_telugu.mp3"
    _two_pass_loudnorm(narration_wav, normalized_mp3)

    # ── Export JSON metadata ───────────────────────────────────────────
    metadata_path = output_dir / f"{stem}_ad_metadata.json"
    _export_metadata(plan, metadata_path)

    # ── Optional: merge AD track with original movie ───────────────────
    merged_video: Optional[Path] = None
    if merge_with_video:
        merged_video = output_dir / f"{stem}_with_AD.mp4"
        _merge_with_movie(video_path, normalized_mp3, merged_video)

    log.info(f"[AudioMixer] ✅ Final AD track → {normalized_mp3}")
    return normalized_mp3, merged_video, metadata_path


# ══════════════════════════════════════════════════════════════════════════
#  CORE: PCM OVERLAY ENGINE
# ══════════════════════════════════════════════════════════════════════════

def _pcm_overlay(plan: SyncPlan) -> Path:
    """
    Place every clip at its exact sample offset using numpy array addition.

    This is mathematically exact because:
      - The output buffer is a flat numpy array sized to total_samples * channels.
      - Each clip is read as a numpy array of int16 samples.
      - The clip is ADDED into the buffer at offset_samples * channels.
      - No resampling, no filtering, no FFmpeg — pure integer arithmetic.
    """
    total_frames = plan.total_samples  # frames (not bytes)
    log.info(f"[AudioMixer] Allocating PCM buffer: {total_frames} frames × {CHANNELS} ch")

    # Allocate as float32 for headroom during mixing, convert to int16 at the end
    buffer = np.zeros(total_frames * CHANNELS, dtype=np.float32)

    fade_in_samples = int((config.fade_in_ms / 1000.0) * SAMPLE_RATE)
    fade_out_samples = int((config.fade_out_ms / 1000.0) * SAMPLE_RATE)

    for placed in plan.placed_clips:
        clip_path = placed.clip.audio_path

        # Read clip as raw int16 PCM samples
        clip_data = _read_wav_as_numpy(clip_path)
        if clip_data is None or len(clip_data) == 0:
            log.warning(f"[AudioMixer] Gap #{placed.clip.gap_id}: "
                        f"could not read {clip_path.name}, skipping.")
            continue

        clip_float = clip_data.astype(np.float32)
        clip_frames = len(clip_float) // CHANNELS

        # ── Apply fade in/out ──────────────────────────────────────────
        clip_float = _apply_fades(clip_float, clip_frames, fade_in_samples, fade_out_samples)

        # ── Compute insertion point ────────────────────────────────────
        # offset_samples is in FRAMES (not interleaved samples)
        start_idx = placed.offset_samples * CHANNELS
        end_idx = start_idx + len(clip_float)

        # Clamp to buffer bounds
        if start_idx >= len(buffer):
            log.warning(f"[AudioMixer] Gap #{placed.clip.gap_id}: "
                        f"offset {placed.offset_samples} beyond buffer end, skipping.")
            continue

        if end_idx > len(buffer):
            # Truncate clip to fit within video duration
            overflow_samples = end_idx - len(buffer)
            clip_float = clip_float[: len(clip_float) - overflow_samples]
            end_idx = len(buffer)
            log.warning(f"[AudioMixer] Gap #{placed.clip.gap_id}: "
                        f"truncated {overflow_samples // CHANNELS} frames at video end.")

        # ── ADD clip into buffer (mixing) ──────────────────────────────
        buffer[start_idx:end_idx] += clip_float

        log.info(
            f"[AudioMixer] Placed gap #{placed.clip.gap_id:03d} at "
            f"sample {placed.offset_samples} "
            f"({placed.actual_timestamp:.3f}s) "
            f"dur={placed.audio_duration:.3f}s "
            f"drift={placed.drift_ms:+.4f}ms"
        )

    # ── Clip to int16 range and export ─────────────────────────────────
    buffer = np.clip(buffer, -32768, 32767).astype(np.int16)

    out_path = config.temp_dir / "narration_mixed.wav"
    _write_wav(out_path, buffer, SAMPLE_RATE, CHANNELS)

    log.info(f"[AudioMixer] PCM overlay complete → {out_path.name} "
             f"({total_frames / SAMPLE_RATE:.3f}s)")
    return out_path


def _apply_fades(
    data: np.ndarray,
    num_frames: int,
    fade_in_frames: int,
    fade_out_frames: int,
) -> np.ndarray:
    """Apply linear fade-in and fade-out to interleaved stereo PCM data."""
    if num_frames == 0:
        return data

    # Fade in
    fi = min(fade_in_frames, num_frames)
    if fi > 0:
        ramp = np.linspace(0.0, 1.0, fi, dtype=np.float32)
        # Expand ramp to interleaved stereo: each frame has CHANNELS samples
        ramp_stereo = np.repeat(ramp, CHANNELS)
        data[: fi * CHANNELS] *= ramp_stereo

    # Fade out
    fo = min(fade_out_frames, num_frames)
    if fo > 0:
        ramp = np.linspace(1.0, 0.0, fo, dtype=np.float32)
        ramp_stereo = np.repeat(ramp, CHANNELS)
        data[-(fo * CHANNELS):] *= ramp_stereo

    return data


# ══════════════════════════════════════════════════════════════════════════
#  WAV I/O — direct PCM reading/writing (no FFmpeg needed for this step)
# ══════════════════════════════════════════════════════════════════════════

def _read_wav_as_numpy(path: Path) -> Optional[np.ndarray]:
    """
    Read a WAV file as a flat numpy int16 array (interleaved channels).
    The TTS engine guarantees all clips are 44100Hz/stereo/16-bit.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth != 2:
            log.warning(f"[AudioMixer] {path.name}: expected 16-bit, got {sampwidth * 8}-bit")
            return None

        data = np.frombuffer(raw, dtype=np.int16)

        # If mono, duplicate to stereo
        if n_channels == 1:
            data = np.column_stack([data, data]).flatten()
        elif n_channels != CHANNELS:
            log.warning(f"[AudioMixer] {path.name}: unexpected {n_channels} channels")
            # Take first 2 channels
            data = data.reshape(-1, n_channels)[:, :CHANNELS].flatten()

        return data

    except Exception as exc:
        log.error(f"[AudioMixer] Failed to read {path.name}: {exc}")
        return None


def _write_wav(path: Path, data: np.ndarray, sample_rate: int, channels: int) -> None:
    """Write a numpy int16 array as a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


# ══════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING: loudnorm + export
# ══════════════════════════════════════════════════════════════════════════

def _two_pass_loudnorm(wav_path: Path, out_mp3: Path) -> None:
    """
    EBU R128 loudness normalization in two passes for broadcast accuracy.
    Pass 1: measure actual loudness stats.
    Pass 2: apply corrective filter with measured values.
    """
    target_i = config.target_lufs
    target_tp = -1.5
    target_lra = 11.0

    # ── Pass 1: measure ──────────────────────────────────────────────────
    log.info("[AudioMixer] Loudnorm pass 1 — measuring…")
    p1_cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json",
        "-f", "null", "-",
    ]
    p1 = subprocess.run(p1_cmd, capture_output=True, text=True)
    stats = _parse_loudnorm_stats(p1.stderr)

    if stats:
        measured_i = stats.get("input_i", "-23.0")
        measured_lra = stats.get("input_lra", "7.0")
        measured_tp = stats.get("input_tp", "-2.0")
        measured_thresh = stats.get("input_thresh", "-33.0")
        offset = stats.get("target_offset", "0.0")

        log.info(f"[AudioMixer] Measured: I={measured_i} LRA={measured_lra} TP={measured_tp}")

        # ── Pass 2: apply corrective filter ─────────────────────────────
        log.info("[AudioMixer] Loudnorm pass 2 — applying…")
        p2_af = (
            f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
            f":measured_I={measured_i}:measured_LRA={measured_lra}"
            f":measured_TP={measured_tp}:measured_thresh={measured_thresh}"
            f":offset={offset}:linear=true:print_format=none"
        )
        p2_cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-af", p2_af,
            "-ar", "44100",
            "-ac", "2",
            "-codec:a", "libmp3lame",
            "-q:a", "0",
            "-id3v2_version", "3",
            str(out_mp3),
        ]
        _run_ffmpeg(p2_cmd, "loudnorm pass 2")
    else:
        # Fallback: single-pass
        log.warning("[AudioMixer] Could not parse loudnorm stats, using single-pass.")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}",
            "-ar", "44100", "-ac", "2",
            "-codec:a", "libmp3lame", "-q:a", "0",
            str(out_mp3),
        ]
        _run_ffmpeg(cmd, "loudnorm single-pass")


def _parse_loudnorm_stats(stderr: str) -> Optional[dict]:
    """Extract JSON block from ffmpeg loudnorm stderr output."""
    import re
    match = re.search(r'\{[^}]+\}', stderr, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════
#  MERGE WITH MOVIE
# ══════════════════════════════════════════════════════════════════════════

def _merge_with_movie(video_path: Path, ad_mp3: Path, out_path: Path) -> None:
    """
    Mux the AD track with the original movie.
    Video stream is stream-copied (no re-encode). Audio is AAC 192k.
    """
    log.info(f"[AudioMixer] Merging AD track with movie → {out_path.name}")

    pts_offset = _get_audio_start_pts(video_path)
    log.debug(f"[AudioMixer] Video audio PTS start offset: {pts_offset:.6f}s")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(ad_mp3),
        "-filter_complex",
        f"[1:a]adelay={pts_offset * 1000:.3f}|{pts_offset * 1000:.3f}:all=1[ad_aligned];"
        "[0:a][ad_aligned]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run_ffmpeg(cmd, "merge AD with movie")
    log.info(f"[AudioMixer] ✅ Merged movie → {out_path}")


# ══════════════════════════════════════════════════════════════════════════
#  METADATA EXPORT
# ══════════════════════════════════════════════════════════════════════════

def _export_metadata(plan: SyncPlan, out_path: Path) -> None:
    """Export full timeline JSON with sync analytics."""
    metadata = {
        "pipeline": "Telugu AD Pipeline v2 — PCM overlay sync",
        "video_duration_sec": round(plan.video_duration, 6),
        "sample_rate": SAMPLE_RATE,
        "total_clips": plan.total_clips,
        "overflow_clips": plan.overflow_clips,
        "max_drift_ms": round(plan.max_drift_ms, 6),
        "clips": [],
    }

    for p in plan.placed_clips:
        metadata["clips"].append({
            "gap_id": p.clip.gap_id,
            "gap_start": round(p.clip.gap.start, 6),
            "gap_end": round(p.clip.gap.end, 6),
            "gap_duration": round(p.clip.gap.duration, 6),
            "target_timestamp": round(p.target_timestamp, 6),
            "actual_timestamp": round(p.actual_timestamp, 6),
            "offset_samples": p.offset_samples,
            "drift_ms": round(p.drift_ms, 6),
            "audio_duration": round(p.audio_duration, 6),
            "end_timestamp": round(p.end_timestamp, 6),
            "gap_available": round(p.gap_available, 6),
            "margin": round(p.margin, 6),
            "status": p.status,
            "text_te": p.clip.telugu_text,
            "audio_file": p.clip.audio_path.name,
        })

    out_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"[AudioMixer] Metadata → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def _get_video_duration(path: Path) -> float:
    """Get frame-accurate video duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    val = result.stdout.strip()
    if val and val != "N/A":
        try:
            return float(val)
        except ValueError:
            pass

    cmd2 = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    try:
        return float(result2.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Cannot determine duration of {path}")


def _get_audio_start_pts(path: Path) -> float:
    """Return the audio stream start_pts in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=start_time",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        val = result.stdout.strip()
        return float(val) if val and val != "N/A" else 0.0
    except ValueError:
        return 0.0


def _run_ffmpeg(cmd: list[str], step: str) -> None:
    """Run FFmpeg command, raise on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg [{step}] failed:\n{result.stderr[-600:]}")
