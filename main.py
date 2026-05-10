"""
main.py — Telugu Audio Description Pipeline
Entry point for the complete AD generation workflow.

Usage:
    python main.py movie.mp4
    python main.py movie.mp4 --merge
    python main.py movie.mp4 --tts-provider elevenlabs --merge
"""

import argparse
import json
import os
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ── SSL patch — must run before any network imports ────────────────────────
# Disables SSL verification for corporate/proxy networks (affects httpx too)
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ.setdefault("SSL_CERT_FILE", "")
_orig_ssl = ssl.create_default_context
def _no_verify_ssl(*args, **kwargs):
    ctx = _orig_ssl(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = _no_verify_ssl

# Patch httpx at import time so google-genai uses unverified SSL
try:
    import httpx as _httpx
    _orig_init = _httpx.Client.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        _orig_init(self, *args, **kwargs)
    _httpx.Client.__init__ = _patched_init

    _orig_async_init = _httpx.AsyncClient.__init__
    def _patched_async_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        _orig_async_init(self, *args, **kwargs)
    _httpx.AsyncClient.__init__ = _patched_async_init
except ImportError:
    pass
# ──────────────────────────────────────────────────────────────────────────

from config import config, PipelineConfig
from logger import setup_logger
from models import (
    SilenceGap,
    SceneDescription,
    TeluguNarration,
    TTSClip,
    PipelineResult,
)
from gap_detector import detect_silence_gaps
from scene_analyzer import analyze_scene
from translator import translate_to_telugu
from duration_optimizer import optimize_and_synthesize
from audio_mixer import mix_narration_track

log = setup_logger("main", config.log_level)

# ── Supported input formats ────────────────────────────────────────────────
SUPPORTED_FORMATS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ad_pipeline",
        description="Generate synchronized Telugu Audio Description for a movie.",
    )
    parser.add_argument("video", type=Path, help="Input video file (.mp4/.mkv/.mov)")
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge AD narration track with original movie audio",
    )
    parser.add_argument(
        "--tts-provider", choices=["gtts", "elevenlabs", "azure"],
        default=None, help="Override TTS provider from .env",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker threads",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable Gemini response caching",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Output directory (default: output/)",
    )
    return parser.parse_args()


# ── Pipeline ───────────────────────────────────────────────────────────────

def run_pipeline(video_path: Path, merge: bool = False) -> PipelineResult:
    """
    Orchestrate the full 8-step AD generation pipeline.

    Steps:
      1. Validate input
      2. Detect silence gaps
      3. Scene understanding (Gemini Vision)
      4. Telugu translation (Gemini)
      5. Duration optimization + TTS synthesis
      6. Audio mixing + export
    """
    start_time = time.time()

    # ── Validate input ─────────────────────────────────────────────────────
    if not video_path.exists():
        log.error(f"Input file not found: {video_path}")
        sys.exit(1)
    if video_path.suffix.lower() not in SUPPORTED_FORMATS:
        log.error(f"Unsupported format: {video_path.suffix}")
        sys.exit(1)
    if not config.gemini_api_key:
        log.error("GEMINI_API_KEY not set. Copy .env.example → .env and add your key.")
        sys.exit(1)

    config.ensure_dirs()
    log.info(f"{'='*60}")
    log.info(f"  Telugu AD Pipeline — {video_path.name}")
    log.info(f"  TTS: {config.tts_provider} | Workers: {config.max_workers}")
    log.info(f"{'='*60}")

    # ── STEP 2: Detect silence gaps ────────────────────────────────────────
    log.info("STEP 2 ▶ Detecting silence gaps…")
    gaps: list[SilenceGap] = detect_silence_gaps(video_path)

    if not gaps:
        log.warning("No usable silence gaps found. The movie may have continuous dialogue.")
        sys.exit(0)

    log.info(f"  → {len(gaps)} gaps detected.")

    # ── STEP 3: Scene analysis (parallel) ─────────────────────────────────
    log.info("STEP 3 ▶ Analyzing scenes with Gemini Vision…")
    descriptions: list[SceneDescription] = _analyze_scenes_parallel(gaps, video_path)

    # ── STEP 4: Telugu translation ─────────────────────────────────────────
    log.info("STEP 4 ▶ Translating to Telugu…")
    narrations: list[TeluguNarration] = _translate_parallel(descriptions)

    # ── STEP 5: Duration optimization + TTS ───────────────────────────────
    log.info("STEP 5+6 ▶ Optimizing duration & synthesizing speech…")
    clips: list[TTSClip] = _synthesize_all(narrations)

    if not clips:
        log.error("No TTS clips were generated. Check API keys and dependencies.")
        sys.exit(1)

    # ── STEP 7+8: Audio mixing + export ───────────────────────────────────
    log.info("STEP 7+8 ▶ Mixing and exporting…")
    original_audio = _extract_audio_for_mixing(video_path)

    narration_mp3, merged_video, metadata_json = mix_narration_track(
        clips=clips,
        original_audio=original_audio,
        video_path=video_path,
        output_dir=config.output_dir,
        merge_with_video=merge,
    )

    elapsed = time.time() - start_time

    result = PipelineResult(
        input_video=video_path,
        narration_track=narration_mp3,
        merged_video=merged_video,
        metadata_json=metadata_json,
        total_gaps_detected=len(gaps),
        total_gaps_narrated=len(clips),
        total_gaps_skipped=len(gaps) - len(clips),
        processing_time_seconds=elapsed,
    )

    _print_summary(result)
    return result


# ── Parallel Helpers ───────────────────────────────────────────────────────

def _analyze_scenes_parallel(
    gaps: list[SilenceGap], video_path: Path
) -> list[SceneDescription]:
    """Run scene analysis for all gaps using a thread pool."""
    descriptions: list[Optional[SceneDescription]] = [None] * len(gaps)
    previous_text: Optional[str] = None

    # Process sequentially to maintain context window awareness
    # (parallel would scramble the previous_description context)
    for idx, gap in enumerate(gaps):
        desc = analyze_scene(gap, video_path, previous_description=previous_text)
        descriptions[idx] = desc
        if desc.english_text:
            previous_text = desc.english_text

    return [d for d in descriptions if d is not None and d.english_text.strip()]


def _translate_parallel(
    descriptions: list[SceneDescription],
) -> list[TeluguNarration]:
    """Translate all descriptions to Telugu in parallel."""
    narrations: list[TeluguNarration] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(translate_to_telugu, desc): desc
            for desc in descriptions
        }
        for future in as_completed(futures):
            try:
                narration = future.result()
                if narration.telugu_text.strip():
                    narrations.append(narration)
            except Exception as exc:
                log.error(f"[main] Translation error: {exc}")

    return sorted(narrations, key=lambda n: n.gap.start)


def _synthesize_all(narrations: list[TeluguNarration]) -> list[TTSClip]:
    """Optimize duration and synthesize TTS for all narrations."""
    clips: list[TTSClip] = []
    for narration in narrations:
        clip = optimize_and_synthesize(narration)
        if clip:
            clips.append(clip)
        else:
            log.warning(f"[main] Gap #{narration.gap_id}: synthesis failed, skipping.")
    return sorted(clips, key=lambda c: c.gap.start)


def _extract_audio_for_mixing(video_path: Path) -> Path:
    """Extract audio track needed by the mixer for duration reference."""
    import subprocess
    out = config.temp_dir / "original_audio_mix.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "2", "-ar", "44100",
        "-acodec", "pcm_s16le", str(out),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out


# ── Summary ────────────────────────────────────────────────────────────────

def _print_summary(result: PipelineResult) -> None:
    log.info(f"\n{'='*60}")
    log.info("  ✅  PIPELINE COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"  Input           : {result.input_video.name}")
    log.info(f"  AD Track        : {result.narration_track}")
    if result.merged_video:
        log.info(f"  Merged Movie    : {result.merged_video}")
    log.info(f"  Metadata        : {result.metadata_json}")
    log.info(f"  Gaps detected   : {result.total_gaps_detected}")
    log.info(f"  Gaps narrated   : {result.total_gaps_narrated}")
    log.info(f"  Gaps skipped    : {result.total_gaps_skipped}")
    log.info(f"  Processing time : {result.processing_time_seconds:.1f}s")
    log.info(f"{'='*60}\n")


# ── Entry Point ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Apply CLI overrides to config
    if args.tts_provider:
        config.tts_provider = args.tts_provider
    if args.workers:
        config.max_workers = args.workers
    if args.no_cache:
        config.cache_gemini = False
    if args.output_dir:
        config.output_dir = args.output_dir

    run_pipeline(video_path=args.video, merge=args.merge)


if __name__ == "__main__":
    main()

