"""
duration_optimizer.py — STEP 5
Synthesizes Telugu narration at 1.2x speed to keep narration punchy and
ensure it fits within the silence gap without overlapping dialogue.
"""

from pathlib import Path
from typing import Optional

from config import config
from logger import setup_logger
from models import TeluguNarration, TTSClip
from tts_engine import synthesize

log = setup_logger("duration_optimizer")

# Speed for all narration — 1.2x keeps it brisk and fits within gaps
TTS_SPEED = 1.2

# If clip still overflows after speed-up, trim text to this ratio of the gap
MAX_FILL_RATIO = 0.88  # use at most 88% of available gap duration


def optimize_and_synthesize(narration: TeluguNarration) -> Optional[TTSClip]:
    """
    Synthesize Telugu narration at 1.2x speed.
    If the clip overflows the gap, shorten the text and retry once.
    """
    if not narration.telugu_text.strip():
        log.warning(f"[Optimizer] Gap #{narration.gap_id}: empty text, skipping.")
        return None

    available = narration.gap.usable_duration(config.safety_padding)
    log.info(f"[Optimizer] Gap #{narration.gap_id}: available={available:.2f}s | speed={TTS_SPEED}x")

    text = narration.telugu_text.strip()

    # First attempt at 1.2x speed
    clip = synthesize(text, narration.gap, tts_speed=TTS_SPEED)
    if clip is None:
        log.error(f"[Optimizer] TTS failed for gap #{narration.gap_id}.")
        return None

    # If it overflows the gap, trim to first sentence and retry
    if clip.actual_duration > available * MAX_FILL_RATIO:
        log.warning(
            f"[Optimizer] Gap #{narration.gap_id}: {clip.actual_duration:.2f}s overflows "
            f"{available:.2f}s — trimming to first sentence."
        )
        # Keep only the first sentence (split on ।, ., or newline)
        import re
        sentences = re.split(r'[।.\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1:
            text = sentences[0] + "."
            clip = synthesize(text, narration.gap, tts_speed=TTS_SPEED)
            if clip is None:
                log.error(f"[Optimizer] Retry TTS failed for gap #{narration.gap_id}.")
                return None

    margin = available - clip.actual_duration
    status = "✓" if margin >= 0 else f"OVERFLOW {abs(margin):.2f}s"
    log.info(f"[Optimizer] Gap #{narration.gap_id}: "
             f"{clip.actual_duration:.2f}s / {available:.2f}s {status}")

    return clip
