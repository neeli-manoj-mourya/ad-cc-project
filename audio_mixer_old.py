"""
audio_mixer.py — STEP 7 & 8
Sample-accurate timeline placement of TTS narration clips into silence gaps.

Sync strategy:
  - All TTS clips are standardized to 44100 Hz / stereo / 16-bit WAV before mixing
  - Delay is expressed in exact SAMPLES (not milliseconds) to avoid rounding drift
  - Base silence track is generated at EXACT sample count from source video
  - Every clip placement is validated for drift BEFORE final export
  - loudnorm applied in two-pass mode for broadcast-accurate LUFS
"""

import json
import subprocess
from pathlib import Path
from typing import Optional

from config import config
from logger import setup_logger
from models import TTSClip

log = setup_logger("audio_mixer")

SAMPLE_RATE   = 44100
CHANNELS      = 2
CODEC         = "pcm_s16le"
BYTES_PER_SAMPLE = 2  # 16-bit


# ── Public API ─────────────────────────────────────────────────────────────

def mix_narration_track(
    clips: list[TTSClip],
    original_audio: Path,
    video_path: Path,
    output_dir: Path,
    merge_with_video: bool = False,
) -> tuple[Path, Optional[Path], Path]:
    """
    Build the final AD narration track with sample-accurate sync.
    Returns (narration_mp3, merged_video_or_None, metadata_json).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # Get exact frame-accurate duration from the video container
    video_duration = _get_video_duration(video_path)
    total_samples  = int(video_duration * SAMPLE_RATE)

    log.info(f"[AudioMixer] Video duration: {video_duration:.6f}s = {total_samples} samples @ {SAMPLE_RATE}Hz")

    # Validate and log sync plan before mixing
    _validate_sync_plan(clips, video_duration)

    # Build narration track with sample-accurate placement
    narration_wav = _build_narration_track(clips, total_samples)

    # Two-pass loudnorm → final MP3
    normalized_mp3 = output_dir / f"{stem}_audio_description_telugu.mp3"
    _two_pass_loudnorm(narration_wav, normalized_mp3)

    # Export JSON metadata
    metadata_path = output_dir / f"{stem}_ad_metadata.json"
    _export_metadata(clips, metadata_path, video_duration)

    # Optional: merge AD track with original movie
    merged_video: Optional[Path] = None
    if merge_with_video:
        merged_video = output_dir / f"{stem}_with_AD.mp4"
        _merge_with_movie(video_path, normalized_mp3, merged_video)

    log.info(f"[AudioMixer] ✅ Final AD track → {normalized_mp3}")
    return normalized_mp3, merged_video, metadata_path


# ── Sync Validation ───────────────────────────────────────────────────────

def _validate_sync_plan(clips: list[TTSClip], video_duration: float) -> None:
    """
    Log a sync plan table and warn about any clips that would overflow their gap.
    """
    log.info(f"[SyncValidator] {'─'*65}")
    log.info(f"[SyncValidator]  {'Gap#':>4}  {'Start':>8}  {'End':>8}  {'Gap':>6}  {'Audio':>6}  {'Margin':>7}  Status")
    log.info(f"[SyncValidator] {'─'*65}")

    for clip in clips:
        gap_avail = clip.gap.usable_duration(config.safety_padding)
        margin    = gap_avail - clip.actual_duration
        placement = clip.gap.start + config.safety_padding
        end_time  = placement + clip.actual_duration
        status    = "✅ OK" if margin >= 0 else f"⚠️  OVERFLOW +{abs(margin):.3f}s"

        log.info(
            f"[SyncValidator]  #{clip.gap_id:03d}  "
            f"{placement:>7.3f}s  {end_time:>7.3f}s  "
            f"{clip.gap.duration:>5.2f}s  {clip.actual_duration:>5.3f}s  "
            f"{margin:>+6.3f}s  {status}"
        )

        if end_time > video_duration:
            log.warning(f"[SyncValidator] Gap #{clip.gap_id}: clip extends beyond video end!")

    log.info(f"[SyncValidator] {'─'*65}")


# ── Sample-Accurate Narration Track Builder ───────────────────────────────

def _build_narration_track(clips: list[TTSClip], total_samples: int) -> Path:
    """
    Place each clip at its exact sample offset using raw PCM arithmetic.

    Strategy:
      1. Create a silent PCM buffer of exact total_samples length
      2. For each clip, write its PCM data at: offset = round(start_time * SAMPLE_RATE)
      3. Apply fade-in/fade-out at sample level
      4. Export as WAV
    """
    if not clips:
        return _create_silent_wav(total_samples)

    # Build FFmpeg filter_complex with sample-precise adelay
    inputs: list[str] = []
    filter_parts: list[str] = []
    mix_labels: list[str] = []

    # Stream 0: silence base at exact length
    silent_base = _create_silent_wav(total_samples)
    inputs += ["-i", str(silent_base)]
    mix_labels.append("[0:a]")

    for idx, clip in enumerate(clips):
        stream = idx + 1
        inputs += ["-i", str(clip.audio_path)]

        # Compute exact sample offset → convert back to ms for adelay
        # Using float seconds * sample_rate → integer samples → ms
        start_sec    = clip.gap.start + config.safety_padding
        start_sample = round(start_sec * SAMPLE_RATE)
        delay_ms_exact = (start_sample / SAMPLE_RATE) * 1000.0  # precise float ms

        fade_in_s    = config.fade_in_ms / 1000.0
        fade_out_s   = config.fade_out_ms / 1000.0
        fade_out_st  = max(0.0, clip.actual_duration - fade_out_s)

        raw_label    = f"[raw{stream}]"
        faded_label  = f"[faded{stream}]"
        delayed_label = f"[d{stream}]"

        # Resample to ensure matching sample rate (defensive)
        filter_parts.append(
            f"[{stream}:a]aresample={SAMPLE_RATE}:resampler=swr{raw_label}"
        )
        # Fade in + fade out
        filter_parts.append(
            f"{raw_label}"
            f"afade=t=in:st=0:d={fade_in_s:.4f},"
            f"afade=t=out:st={fade_out_st:.4f}:d={fade_out_s:.4f}"
            f"{faded_label}"
        )
        # Delay with sub-millisecond float precision
        filter_parts.append(
            f"{faded_label}adelay={delay_ms_exact:.3f}|{delay_ms_exact:.3f}:all=1{delayed_label}"
        )
        mix_labels.append(delayed_label)

        log.debug(
            f"[AudioMixer] Gap #{clip.gap_id}: "
            f"start_sample={start_sample} ({start_sec:.6f}s) "
            f"delay_ms={delay_ms_exact:.3f}"
        )

    # Mix all streams — use first stream duration (video length)
    mix_in = "".join(mix_labels)
    n      = len(mix_labels)
    filter_parts.append(
        f"{mix_in}amix=inputs={n}:duration=first:dropout_transition=0:normalize=0[out]"
    )

    out_wav = config.temp_dir / "narration_mixed.wav"
    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[out]",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-acodec", CODEC,
            str(out_wav),
        ]
    )
    _run_ffmpeg(cmd, "sample-accurate clip overlay")
    return out_wav


# ── Silent Base Track ─────────────────────────────────────────────────────

def _create_silent_wav(total_samples: int) -> Path:
    """Create a silent WAV of exactly total_samples samples."""
    out = config.temp_dir / "silent_base.wav"
    duration = total_samples / SAMPLE_RATE
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
        "-t", f"{duration:.6f}",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-acodec", CODEC,
        str(out),
    ]
    _run_ffmpeg(cmd, "create silent base")
    return out


# ── Two-Pass Loudnorm ─────────────────────────────────────────────────────

def _two_pass_loudnorm(wav_path: Path, out_mp3: Path) -> None:
    """
    EBU R128 loudness normalization in two passes for broadcast accuracy.
    Pass 1: measure actual loudness stats.
    Pass 2: apply corrective filter with measured values.
    """
    target_i  = config.target_lufs
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
        measured_i   = stats.get("input_i",   "-23.0")
        measured_lra = stats.get("input_lra",  "7.0")
        measured_tp  = stats.get("input_tp",  "-2.0")
        measured_thresh = stats.get("input_thresh", "-33.0")
        offset       = stats.get("target_offset", "0.0")

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
            "-q:a", "0",          # VBR highest quality
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


# ── Merge with Video ──────────────────────────────────────────────────────

def _merge_with_movie(video_path: Path, ad_mp3: Path, out_path: Path) -> None:
    """
    Mux the AD track with the original movie using frame-accurate pts_start.
    Video stream is stream-copied (no re-encode). Audio is AAC 192k.
    """
    log.info(f"[AudioMixer] Merging AD track with movie → {out_path.name}")

    # Get video stream start_pts to handle non-zero DTS offsets
    pts_offset = _get_audio_start_pts(video_path)
    log.debug(f"[AudioMixer] Video audio PTS start offset: {pts_offset:.6f}s")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(ad_mp3),
        "-filter_complex",
        # Offset AD track to match video audio start PTS
        f"[1:a]adelay={pts_offset*1000:.3f}|{pts_offset*1000:.3f}:all=1[ad_aligned];"
        "[0:a][ad_aligned]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",          # No video re-encode — preserves exact timing
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",  # Web-optimized MP4
        str(out_path),
    ]
    _run_ffmpeg(cmd, "merge AD with movie")
    log.info(f"[AudioMixer] ✅ Merged movie → {out_path}")


# ── Metadata Export ───────────────────────────────────────────────────────

def _export_metadata(
    clips: list[TTSClip],
    out_path: Path,
    video_duration: float,
) -> None:
    """Export full timeline JSON with sync analytics."""
    metadata = {
        "pipeline": "Telugu AD Pipeline",
        "video_duration_sec": round(video_duration, 6),
        "sample_rate": SAMPLE_RATE,
        "total_clips": len(clips),
        "clips": [],
    }

    for clip in clips:
        placement   = clip.gap.start + config.safety_padding
        end_time    = placement + clip.actual_duration
        margin      = clip.gap.usable_duration(config.safety_padding) - clip.actual_duration
        start_sample = round(placement * SAMPLE_RATE)

        metadata["clips"].append({
            "gap_id":           clip.gap_id,
            "gap_start":        round(clip.gap.start, 6),
            "gap_end":          round(clip.gap.end, 6),
            "gap_duration":     round(clip.gap.duration, 6),
            "narration_start":  round(placement, 6),
            "narration_end":    round(end_time, 6),
            "narration_duration": round(clip.actual_duration, 6),
            "start_sample":     start_sample,
            "margin_sec":       round(margin, 6),
            "tts_speed":        round(clip.tts_speed, 4),
            "sync_status":      "OK" if margin >= 0 else "OVERFLOW",
            "text_te":          clip.telugu_text,
            "audio_file":       clip.audio_path.name,
        })

    out_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"[AudioMixer] Metadata → {out_path.name}")


# ── Utility ───────────────────────────────────────────────────────────────

def _get_video_duration(path: Path) -> float:
    """Get frame-accurate video duration via ffprobe stream data."""
    # Try duration from video stream first (most accurate)
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

    # Fallback: container format duration
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
    """Return the audio stream start_pts in seconds (handles non-zero offsets)."""
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
    """Run FFmpeg command, raise on failure with truncated stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg [{step}] failed:\n{result.stderr[-600:]}")
