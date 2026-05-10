"""
translator.py — STEP 4
Translates English scene descriptions into natural, cinematic Telugu
using the Google Gemini API.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from config import config
from logger import setup_logger
from models import SceneDescription, TeluguNarration

log = setup_logger("translator")


# ── Public API ─────────────────────────────────────────────────────────────

def translate_to_telugu(description: SceneDescription) -> TeluguNarration:
    """
    Translate an English SceneDescription into Telugu and return
    a TeluguNarration with an estimated spoken duration.
    """
    if not description.english_text.strip():
        log.warning(f"[Translator] Gap #{description.gap_id}: empty description, skipping.")
        return TeluguNarration(
            gap_id=description.gap_id,
            telugu_text="",
            gap=description.gap,
            estimated_duration=0.0,
        )

    log.info(f"[Translator] Translating gap #{description.gap_id}: \"{description.english_text}\"")

    # Check translation cache
    cached = _load_cache(description)
    if cached:
        log.debug(f"[Translator] Cache hit for gap #{description.gap_id}")
        return cached

    telugu_text = _query_gemini_translate(description)
    estimated_dur = _estimate_duration(telugu_text)

    result = TeluguNarration(
        gap_id=description.gap_id,
        telugu_text=telugu_text,
        gap=description.gap,
        estimated_duration=estimated_dur,
    )
    _save_cache(description, result)
    return result


# ── Gemini Translation ────────────────────────────────────────────────────

def _query_gemini_translate(description: SceneDescription) -> str:
    """Call Gemini to produce natural, cinematic Telugu translation."""
    from google import genai  # type: ignore
    from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore

    client = genai.Client(api_key=config.gemini_api_key)

    available_seconds = description.gap.usable_duration(config.safety_padding)

    prompt = f"""You are a Telugu movie narrator. A blind person is watching a movie. Tell them what's happening on screen in Telugu.

English: "{description.english_text}"

LANGUAGE STYLE — MOST IMPORTANT:
Write in "వాడుక భాష" (spoken Telugu), NOT "గ్రాంథిక భాష" (literary Telugu).
Imagine you are sitting next to a blind friend in a movie theater in Hyderabad, whispering what's on screen. Use the exact Telugu words you would use in that real conversation.

For ANY English word that Telugu people commonly say in English in daily life, write the English word in Telugu script as-is. Telugu people say "gym" not "వ్యాయామశాల", "phone" not "చరవాణి", "office" not "కార్యాలయం", "hospital" not "వైద్యశాల", "car" not "మోటారు వాహనం", etc.

GENERAL RULE: If a word sounds like something from a Telugu textbook or Doordarshan news, DON'T use it. Use what a normal person in Hyderabad would say.

DESCRIPTION STYLE:
- Be descriptive and emotional. Include HOW characters feel, not just what they do.
- "అతను టెన్షన్‌గా ఆమె వైపు చూస్తున్నాడు" is better than "అతను చూస్తున్నాడు"
- Capture the mood: nervousness, happiness, sadness, romance, tension
- Keep it natural — like you're actually telling your friend what you see
- 1-2 sentences OK for longer gaps

Respond with ONLY the Telugu sentence(s)."""

    @retry(stop=stop_after_attempt(config.gemini_max_retries),
           wait=wait_exponential(multiplier=config.gemini_retry_delay, min=2, max=10))
    def _call() -> str:
        response = client.models.generate_content(
            model=config.gemini_model,
            contents=prompt,
        )
        return response.text.strip()

    try:
        telugu = _call()
        log.info(f"[Translator] Gap #{description.gap_id}: \"{telugu}\"")
        return telugu
    except Exception as exc:
        log.error(f"[Translator] Translation failed for gap #{description.gap_id}: {exc}")
        return ""


# ── Duration Estimation ───────────────────────────────────────────────────

def _estimate_duration(text: str, chars_per_second: float = 12.0) -> float:
    """
    Rough estimate of Telugu TTS duration based on character count.
    Telugu averages ~12 characters per second of natural speech.
    This is refined after actual TTS synthesis in duration_optimizer.py.
    """
    if not text:
        return 0.0
    char_count = len(text.strip())
    return char_count / chars_per_second


# ── Cache Helpers ─────────────────────────────────────────────────────────

def _cache_key(description: SceneDescription) -> str:
    raw = description.english_text.strip().lower()
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_path(description: SceneDescription) -> Path:
    return config.temp_dir / "cache" / f"telugu_{_cache_key(description)}.json"


def _load_cache(description: SceneDescription) -> Optional[TeluguNarration]:
    if not config.cache_gemini:
        return None
    path = _cache_path(description)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TeluguNarration(
                gap_id=description.gap_id,
                telugu_text=data["telugu_text"],
                gap=description.gap,
                estimated_duration=data.get("estimated_duration", 0.0),
            )
        except Exception:
            return None
    return None


def _save_cache(description: SceneDescription, narration: TeluguNarration) -> None:
    path = _cache_path(description)
    path.write_text(
        json.dumps({
            "telugu_text": narration.telugu_text,
            "estimated_duration": narration.estimated_duration,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
