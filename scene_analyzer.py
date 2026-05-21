"""
scene_analyzer.py — STEP 3
Extracts video frames around each silence gap and sends them to
Gemini Vision to generate a concise English scene description.
"""

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import subprocess

from config import config
from logger import setup_logger
from models import SilenceGap, SceneDescription

log = setup_logger("scene_analyzer")


# ── Public API ─────────────────────────────────────────────────────────────

def analyze_scene(gap: SilenceGap, video_path: Path,
                  previous_description: Optional[str] = None) -> SceneDescription:
    """
    For a given SilenceGap, extract surrounding frames and ask
    Gemini Vision to generate a concise accessibility description.
    """
    log.info(f"[SceneAnalyzer] Analyzing gap #{gap.gap_id} ({gap.start:.1f}s–{gap.end:.1f}s)")

    # Check cache first
    cached = _load_from_cache(gap)
    if cached:
        log.debug(f"[SceneAnalyzer] Cache hit for gap #{gap.gap_id}")
        return cached

    frames = _extract_frames(video_path, gap)
    description_text = _query_gemini_vision(gap, frames, previous_description)

    result = SceneDescription(
        gap_id=gap.gap_id,
        english_text=description_text,
        gap=gap,
        frames_used=[str(f) for f in frames],
    )

    if config.cache_gemini:
        _save_to_cache(gap, result)

    return result


# ── Frame Extraction ──────────────────────────────────────────────────────

def _extract_frames(video_path: Path, gap: SilenceGap) -> list[Path]:
    """
    Extract JPEG frames from the context window surrounding the gap.
    Returns a list of frame file paths.
    """
    frame_dir = config.temp_dir / "frames" / f"gap_{gap.gap_id:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Context window: [gap.start - context_before, gap.end + context_after]
    t_start = max(0.0, gap.start - config.context_window_before)
    t_end   = gap.end + config.context_window_after
    fps     = config.frame_sample_fps

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(t_start),
        "-to", str(t_end),
        "-i", str(video_path),
        "-vf", f"fps={fps},scale=640:-1",
        "-q:v", "3",
        str(frame_dir / "frame_%04d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning(f"[SceneAnalyzer] Frame extraction warning: {result.stderr[-200:]}")

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    log.debug(f"[SceneAnalyzer] Extracted {len(frames)} frames for gap #{gap.gap_id}")
    return frames


# ── Gemini Vision Query ───────────────────────────────────────────────────

def _query_gemini_vision(gap: SilenceGap,
                          frames: list[Path],
                          previous_description: Optional[str]) -> str:
    """
    Send frames to Gemini Vision and return a concise English
    audio description sentence.
    """
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore

    client = genai.Client(api_key=config.gemini_api_key)

    # Build image parts (max 8 frames to stay within token budget)
    selected_frames = _select_keyframes(frames, max_frames=8)
    image_parts = []
    for frame_path in selected_frames:
        with open(frame_path, "rb") as f:
            image_data = f.read()
        image_parts.append(types.Part.from_bytes(data=image_data, mime_type="image/jpeg"))

    context_note = ""
    if previous_description:
        context_note = f"\nPrevious narration context: {previous_description}"

    prompt = f"""You are an audio describer for a blind person watching a Telugu movie. Look at these frames and describe ONLY what you literally see.

Frames: {gap.start:.1f}s to {gap.end:.1f}s.{context_note}

SPECIAL CASES — handle these first before anything else:
1. TEXT ON SCREEN (title cards, quote cards, dedication cards):
   → Say exactly: 'Title card reads: "<the exact text shown>"'
   → Do NOT describe the background color or font. Just read the text.

2. CHARACTER NAME CARDS (e.g. "Prudhvi Kamepalli as Arjun"):
   → Say exactly: 'Character card: <Name> as <Role>'
   → If it says "Story/Direction by <Name>", say: 'Credits: <Name> — Story and Direction'

3. END CREDITS / SUBSCRIBE screens:
   → Say: 'End credits rolling' or 'Subscribe screen displayed'
   → Do NOT list crew names.

4. PRODUCTION LOGOS / STUDIO LOGOS:
   → Say: 'Production logo displayed'

FOR NORMAL SCENES (people, locations, actions):
- Use character names if they appeared in a name card earlier. Otherwise say "the man" or "the woman" — do NOT make up names.
- Describe ONLY what is VISIBLE. Do NOT invent objects, locations, or actions.
- Focus on: who is present, what they are doing, their expression/emotion, the setting.
- Present tense. Keep it factual and concise. Gap is {gap.duration:.1f}s long.

Respond with ONLY the description. No explanation, no quotes around your answer."""

    @retry(stop=stop_after_attempt(config.gemini_max_retries),
           wait=wait_exponential(multiplier=config.gemini_retry_delay, min=2, max=10))
    def _call_gemini() -> str:
        contents = [prompt] + image_parts
        response = client.models.generate_content(
            model=config.gemini_vision_model,
            contents=contents,
        )
        return response.text.strip()

    try:
        description = _call_gemini()
        log.info(f"[SceneAnalyzer] Gap #{gap.gap_id}: \"{description}\"")
        return description
    except Exception as exc:
        log.error(f"[SceneAnalyzer] Gemini Vision failed for gap #{gap.gap_id}: {exc}")
        return ""


def _select_keyframes(frames: list[Path], max_frames: int) -> list[Path]:
    """Evenly sample up to max_frames from the full frame list."""
    if len(frames) <= max_frames:
        return frames
    indices = [int(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frames[i] for i in indices]


# ── Cache Helpers ─────────────────────────────────────────────────────────

def _cache_key(gap: SilenceGap) -> str:
    raw = f"{gap.start:.3f}_{gap.end:.3f}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(gap: SilenceGap) -> Path:
    return config.temp_dir / "cache" / f"scene_{_cache_key(gap)}.json"


def _load_from_cache(gap: SilenceGap) -> Optional[SceneDescription]:
    if not config.cache_gemini:
        return None
    path = _cache_path(gap)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SceneDescription(
                gap_id=gap.gap_id,
                english_text=data["english_text"],
                gap=gap,
                frames_used=data.get("frames_used", []),
            )
        except Exception:
            return None
    return None


def _save_to_cache(gap: SilenceGap, desc: SceneDescription) -> None:
    path = _cache_path(gap)
    path.write_text(
        json.dumps({"english_text": desc.english_text, "frames_used": desc.frames_used},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
