"""
timeline_engine.py — Absolute timeline positioning & drift tracking.

This module handles the pure math of placing AD narration clips
onto a movie timeline. It computes exact sample offsets, tracks
cumulative drift, and produces detailed sync diagnostics.

SYNC STRATEGY:
  - Every clip is placed at an ABSOLUTE sample offset from timeline zero.
  - offset_samples = round(target_timestamp_seconds * SAMPLE_RATE)
  - No relative delays — each clip is independently positioned.
  - Drift is measured as the difference between intended and actual placement.
  - Since we use integer sample offsets into a PCM buffer, placement is
    mathematically exact (zero drift by construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import config
from logger import setup_logger
from models import TTSClip

log = setup_logger("timeline_engine")

SAMPLE_RATE: int = 44100
CHANNELS: int = 2
BYTES_PER_SAMPLE: int = 2  # 16-bit PCM


@dataclass
class PlacedClip:
    """A clip with its exact placement on the output timeline."""
    clip: TTSClip
    target_timestamp: float       # seconds — where we WANT the clip to start
    offset_samples: int           # exact sample index in the PCM buffer
    actual_timestamp: float       # seconds — the real time represented by offset_samples
    drift_ms: float               # difference (actual - target) in milliseconds
    audio_duration: float         # clip duration in seconds
    end_timestamp: float          # actual_timestamp + audio_duration
    gap_available: float          # usable gap duration
    margin: float                 # gap_available - audio_duration (negative = overflow)
    status: str                   # "OK", "OVERFLOW", "TRIMMED"


@dataclass
class SyncPlan:
    """Complete timeline plan with diagnostics."""
    placed_clips: list[PlacedClip] = field(default_factory=list)
    video_duration: float = 0.0
    total_samples: int = 0
    max_drift_ms: float = 0.0
    total_clips: int = 0
    overflow_clips: int = 0


def build_sync_plan(
    clips: list[TTSClip],
    video_duration: float,
) -> SyncPlan:
    """
    Compute exact placement for every clip on the timeline.

    Each clip's target timestamp is:
        gap.start + safety_padding

    The sample offset is:
        round(target_timestamp * SAMPLE_RATE)

    This is an ABSOLUTE position — not relative to any previous clip.
    Therefore there is no cumulative drift by construction.

    Args:
        clips: Sorted list of TTSClip objects.
        video_duration: Total video duration in seconds.

    Returns:
        A SyncPlan with all placements and diagnostics.
    """
    total_samples = round(video_duration * SAMPLE_RATE)
    plan = SyncPlan(
        video_duration=video_duration,
        total_samples=total_samples,
        total_clips=len(clips),
    )

    max_drift = 0.0

    for clip in sorted(clips, key=lambda c: c.gap.start):
        # ── Compute absolute target ────────────────────────────────────
        target_ts = clip.gap.start + config.safety_padding

        # ── Convert to exact sample offset ─────────────────────────────
        # This is the ONLY place where floating-point → integer conversion
        # happens. round() gives us the nearest sample. The maximum error
        # is 0.5 samples = 0.5/44100 ≈ 0.011 ms — imperceptible.
        offset = round(target_ts * SAMPLE_RATE)

        # Clamp to valid range
        offset = max(0, min(offset, total_samples - 1))

        # ── Compute actual timestamp from sample offset ────────────────
        actual_ts = offset / SAMPLE_RATE
        drift = (actual_ts - target_ts) * 1000.0  # ms

        # ── Gap / overflow analysis ────────────────────────────────────
        gap_available = clip.gap.usable_duration(config.safety_padding)
        margin = gap_available - clip.actual_duration
        end_ts = actual_ts + clip.actual_duration

        if margin < 0:
            status = f"OVERFLOW +{abs(margin):.3f}s"
            plan.overflow_clips += 1
        else:
            status = "OK"

        if end_ts > video_duration:
            status = "PAST_END"

        placed = PlacedClip(
            clip=clip,
            target_timestamp=target_ts,
            offset_samples=offset,
            actual_timestamp=actual_ts,
            drift_ms=drift,
            audio_duration=clip.actual_duration,
            end_timestamp=end_ts,
            gap_available=gap_available,
            margin=margin,
            status=status,
        )
        plan.placed_clips.append(placed)
        max_drift = max(max_drift, abs(drift))

    plan.max_drift_ms = max_drift
    return plan


def log_sync_plan(plan: SyncPlan) -> None:
    """Print a detailed sync plan table to the log."""
    log.info(f"[SyncPlan] {'═' * 90}")
    log.info(f"[SyncPlan]  Video: {plan.video_duration:.3f}s | "
             f"Clips: {plan.total_clips} | "
             f"Max drift: {plan.max_drift_ms:.4f}ms | "
             f"Overflows: {plan.overflow_clips}")
    log.info(f"[SyncPlan] {'─' * 90}")
    log.info(f"[SyncPlan]  {'Gap#':>4}  {'Target':>9}  {'Actual':>9}  "
             f"{'Drift':>8}  {'Duration':>8}  {'GapAvail':>8}  {'Margin':>8}  Status")
    log.info(f"[SyncPlan] {'─' * 90}")

    for p in plan.placed_clips:
        log.info(
            f"[SyncPlan]  #{p.clip.gap_id:03d}  "
            f"{p.target_timestamp:>8.3f}s  "
            f"{p.actual_timestamp:>8.3f}s  "
            f"{p.drift_ms:>+7.4f}ms  "
            f"{p.audio_duration:>7.3f}s  "
            f"{p.gap_available:>7.2f}s  "
            f"{p.margin:>+7.3f}s  "
            f"{p.status}"
        )

    log.info(f"[SyncPlan] {'═' * 90}")


def validate_no_overlaps(plan: SyncPlan) -> list[str]:
    """Check that no placed clips overlap each other."""
    warnings: list[str] = []
    placed = sorted(plan.placed_clips, key=lambda p: p.actual_timestamp)

    for i in range(1, len(placed)):
        prev_end = placed[i - 1].end_timestamp
        curr_start = placed[i].actual_timestamp
        if curr_start < prev_end:
            gap = prev_end - curr_start
            msg = (f"Overlap: clip #{placed[i - 1].clip.gap_id} ends at "
                   f"{prev_end:.3f}s but clip #{placed[i].clip.gap_id} "
                   f"starts at {curr_start:.3f}s (overlap={gap:.3f}s)")
            warnings.append(msg)
            log.warning(f"[SyncPlan] {msg}")

    return warnings
