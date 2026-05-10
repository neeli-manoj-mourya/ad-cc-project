"""
config.py — Central configuration for the AD pipeline.
All tunable parameters live here. Edit this file to adjust behaviour.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    temp_dir: Path = Path("temp")

    # ── Gap Detection ──────────────────────────────────────────────────────
    min_gap_duration: float = 1.5       # seconds — ignore shorter silences
    safety_padding: float = 0.2         # seconds — keep clear of dialogue edges
    silence_threshold_db: float = -40.0 # dBFS threshold for silence detection
    merge_gap_distance: float = 0.3     # merge gaps closer than this (seconds)

    # ── Scene Analysis ─────────────────────────────────────────────────────
    frames_before_gap: int = 8          # frames sampled before the gap
    frames_after_gap: int = 4           # frames sampled after the gap
    frame_sample_fps: float = 1.0       # frames per second to extract
    context_window_before: float = 6.0  # seconds of video before gap
    context_window_after: float = 3.0   # seconds of video after gap

    # ── Gemini ─────────────────────────────────────────────────────────────
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = "gemini-2.5-flash"
    gemini_vision_model: str = "gemini-2.5-flash"
    cache_gemini: bool = True
    gemini_max_retries: int = 3
    gemini_retry_delay: float = 2.0     # seconds between retries

    # ── TTS ────────────────────────────────────────────────────────────────
    tts_provider: str = field(default_factory=lambda: os.getenv("TTS_PROVIDER", "edge"))
    tts_language: str = "te"            # Telugu BCP-47 code
    tts_speed_range: tuple = (0.85, 1.15)  # safe speed adjustment range

    # ElevenLabs
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", ""))

    # Azure
    azure_speech_key: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_KEY", ""))
    azure_speech_region: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_REGION", "eastus"))

    # ── Timing ─────────────────────────────────────────────────────────────
    target_fill_ratio: float = 0.92     # narration should use ≤92% of gap duration
    max_optimization_attempts: int = 5  # max rewrites to fit narration in gap

    # ── Audio Mixing ───────────────────────────────────────────────────────
    narration_volume_db: float = 0.0    # relative to original; 0 = no change
    fade_in_ms: int = 80
    fade_out_ms: int = 80
    target_lufs: float = -18.0          # loudness normalization target

    # ── Performance ────────────────────────────────────────────────────────
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "4")))
    gpu_enabled: bool = field(default_factory=lambda: os.getenv("GPU_ENABLED", "false").lower() == "true")

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def ensure_dirs(self) -> None:
        """Create all required working directories."""
        for d in [self.input_dir, self.output_dir, self.temp_dir,
                  self.temp_dir / "frames", self.temp_dir / "clips",
                  self.temp_dir / "tts", self.temp_dir / "cache"]:
            Path(d).mkdir(parents=True, exist_ok=True)


# Singleton used across all modules
config = PipelineConfig()
