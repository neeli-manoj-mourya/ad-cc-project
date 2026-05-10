"""
duration_optimizer.py — STEP 5
Simply synthesizes Telugu narration at natural speed. No trimming, no shortening,
no speed adjustment. Just TTS the translated text as-is.
"""

from pathlib import Path
from typing import Optional

from config import config
from logger import setup_logger
from models import TeluguNarration, TTSClip
from tts_engine import synthesize

log = setup_logger("duration_optimizer")


def optimize_and_synthesize(narration: TeluguNarration) -> Optional[TTSClip]:
    """Synthesize Telugu text at natural speed. No modifications."""
    if not narration.telugu_text.strip():
        log.warning(f"[Optimizer] Gap #{narration.gap_id}: empty text, skipping.")
        return None

    available = narration.gap.usable_duration(config.safety_padding)
    log.info(f"[Optimizer] Gap #{narration.gap_id}: available={available:.2f}s")

    clip = synthesize(narration.telugu_text, narration.gap, tts_speed=1.0)
    if clip is None:
        log.error(f"[Optimizer] TTS failed for gap #{narration.gap_id}.")
        return None

    margin = available - clip.actual_duration
    status = "✓" if margin >= 0 else f"overflow {abs(margin):.2f}s"
    log.info(f"[Optimizer] Gap #{narration.gap_id}: "
             f"{clip.actual_duration:.2f}s / {available:.2f}s {status}")

    return clip
