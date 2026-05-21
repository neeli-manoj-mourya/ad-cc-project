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

    prompt = f"""You are a Telugu movie audio describer for blind viewers. Your job: describe what's on screen in natural, spoken Hyderabadi Telugu — SHORT, VIVID, CONVERSATIONAL.

Study these examples carefully and match this exact style:

English: "A man walks into a dark room and nervously looks around"
Telugu: "అతను చీకటి గదిలోకి వెళ్ళి, టెన్షన్‌గా చుట్టూ చూశాడు"

English: "Two friends laugh and eat street food together"
Telugu: "ఇద్దరు ఫ్రెండ్స్ స్ట్రీట్ ఫుడ్ తింటూ నవ్వుతున్నారు"

English: "A woman cries alone sitting by a window at night"
Telugu: "రాత్రిపూట ఆమె ఒంటరిగా కూర్చుని ఏడుస్తోంది"

English: "A car speeds through empty streets in heavy rain"
Telugu: "కార్ వాన లో వేగంగా ఖాళీ రోడ్డు మీద పోతోంది"

English: "The hero stares at the villain with cold anger"
Telugu: "హీరో చల్లటి కోపంతో విలన్‌ని చూస్తున్నాడు"

English: "A child runs excitedly toward a playground"
Telugu: "పిల్లాడు సంతోషంగా పరిగెత్తుకుంటూ పార్క్‌కి వెళ్తున్నాడు"

---
Now describe this scene:
English: "{description.english_text}"

Gap duration: {available_seconds:.1f} seconds (audio plays at 1.2x speed, so write even shorter)

STRICT RULES:
1. MAX 1 sentence. Under 4 seconds → just a short phrase.
2. Hyderabad spoken Telugu ("వాడుక భాష") — NOT formal/textbook Telugu.
3. English loanwords in Telugu script: car→కార్, phone→ఫోన్, office→ఆఫీస్, tension→టెన్షన్
4. No filler, no repetition. Every word must earn its place.
5. Target {max(4, int(available_seconds * 10))} Telugu characters max.

Reply with ONLY the Telugu sentence. Nothing else."""

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
    Telugu averages ~12 characters per second at natural speed.
    We divide by 1.2 since all clips are played at 1.2x speed.
    """
    if not text:
        return 0.0
    char_count = len(text.strip())
    return (char_count / chars_per_second) / 1.2


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
