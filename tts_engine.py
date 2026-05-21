"""
tts_engine.py — STEP 6
Synthesizes Telugu narration text into speech audio clips.

Supports:
  - edge  (default — Microsoft Edge Neural TTS, free, natural male voice)
  - gtts  (Google TTS, free fallback)
  - elevenlabs (high quality, requires API key)
  - azure (requires API key)

ALL providers output a standardized 44100 Hz / stereo / 16-bit WAV
so the audio mixer can do sample-accurate timeline placement.
"""

import asyncio
import os
import ssl
import subprocess
from pathlib import Path
from typing import Optional

# ── Fix SSL for corporate/proxy networks ──────────────────────────────────
# Monkey-patch ssl.create_default_context to skip verification
_original_create_default_context = ssl.create_default_context

def _create_unverified_context(*args, **kwargs):
    ctx = _original_create_default_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

ssl.create_default_context = _create_unverified_context

from config import config
from logger import setup_logger
from models import SilenceGap, TTSClip

log = setup_logger("tts_engine")

# Edge TTS — te-IN-MohanNeural is the native male Telugu neural voice
EDGE_VOICE_MALE_TE   = "te-IN-MohanNeural"
EDGE_VOICE_FEMALE_TE = "te-IN-ShrutiNeural"

# Target audio spec — every provider must output this before returning
TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS    = 2           # stereo
TARGET_FORMAT      = "pcm_s16le" # 16-bit WAV


# ── Public API ─────────────────────────────────────────────────────────────

def synthesize(text: str, gap: SilenceGap, tts_speed: float = 1.0) -> Optional[TTSClip]:
    """
    Synthesize Telugu text to speech.
    Returns a TTSClip with path to a normalized WAV file and its actual duration.
    """
    if not text.strip():
        return None

    # Final output is always WAV for mixer accuracy
    out_path = config.temp_dir / "tts" / f"clip_{gap.gap_id:04d}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    provider = config.tts_provider.lower()
    log.info(f"[TTS] Gap #{gap.gap_id} | provider={provider} | speed={tts_speed:.2f}x")

    success = False

    if provider == "edge":
        success = _synthesize_edge(text, out_path, tts_speed)
    elif provider == "gtts":
        success = _synthesize_gtts(text, out_path, tts_speed)
    elif provider == "elevenlabs":
        success = _synthesize_elevenlabs(text, out_path, tts_speed)
    elif provider == "azure":
        success = _synthesize_azure(text, out_path, tts_speed)
    else:
        log.warning(f"[TTS] Unknown provider '{provider}', falling back to Edge TTS.")
        success = _synthesize_edge(text, out_path, tts_speed)

    if not success or not out_path.exists():
        log.error(f"[TTS] Synthesis failed for gap #{gap.gap_id}.")
        return None

    # Ensure standard spec (resample / remap if needed)
    out_path = _normalize_to_standard(out_path)
    actual_duration = _measure_duration_accurate(out_path)

    log.info(f"[TTS] Gap #{gap.gap_id}: {actual_duration:.3f}s | "
             f"gap_avail={gap.usable_duration(config.safety_padding):.3f}s → {out_path.name}")

    return TTSClip(
        gap_id=gap.gap_id,
        audio_path=out_path,
        actual_duration=actual_duration,
        gap=gap,
        telugu_text=text,
        tts_speed=tts_speed,
    )


# ── Edge TTS — Microsoft Neural (Free, High Quality) ──────────────────────

def _synthesize_edge(text: str, out_path: Path, speed: float) -> bool:
    """
    Use edge-tts with te-IN-MohanNeural (natural male Telugu neural voice).
    Prosody is controlled via edge-tts's built-in rate/pitch/volume params.
    """
    try:
        import edge_tts  # type: ignore

        # Convert speed multiplier to edge-tts rate percentage
        # e.g. 1.2 → +20%, 1.0 → +0%, 0.9 → -10%
        rate_pct = int((speed - 1.0) * 100)
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

        # Slightly lower pitch for warm male narrator tone
        pitch_str = "-2Hz"

        # edge-tts outputs MP3 — we convert to WAV afterwards
        mp3_path = out_path.with_suffix(".mp3")

        async def _run() -> None:
            communicate = edge_tts.Communicate(
                text=text,
                voice=EDGE_VOICE_MALE_TE,
                rate=rate_str,
                pitch=pitch_str,
                volume="+5%",
            )
            await communicate.save(str(mp3_path))

        asyncio.run(_run())

        if mp3_path.exists():
            # Convert MP3 → standardized WAV
            _ffmpeg_convert(mp3_path, out_path)
            mp3_path.unlink(missing_ok=True)
            return out_path.exists()

        return False

    except ImportError:
        log.error("[TTS] edge-tts not installed. Run: pip install edge-tts")
        return False
    except Exception as exc:
        log.error(f"[TTS] Edge TTS failed: {exc}")
        return False


# ── gTTS (Google TTS — free fallback) ─────────────────────────────────────

def _synthesize_gtts(text: str, out_path: Path, speed: float) -> bool:
    """Use gTTS for Telugu — natural but female voice."""
    try:
        from gtts import gTTS  # type: ignore

        mp3_path = out_path.with_suffix(".mp3")
        tts = gTTS(text=text, lang="te", slow=(speed < 0.9))
        tts.save(str(mp3_path))

        # Convert + apply speed
        _ffmpeg_convert(mp3_path, out_path, speed=speed if abs(speed - 1.0) > 0.05 else None)
        mp3_path.unlink(missing_ok=True)
        return out_path.exists()

    except ImportError:
        log.error("[TTS] gTTS not installed. Run: pip install gtts")
        return False
    except Exception as exc:
        log.error(f"[TTS] gTTS failed: {exc}")
        return False


# ── ElevenLabs ────────────────────────────────────────────────────────────

def _synthesize_elevenlabs(text: str, out_path: Path, speed: float) -> bool:
    """Use ElevenLabs API for ultra-realistic TTS."""
    try:
        import requests

        headers = {
            "xi-api-key": config.elevenlabs_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.65, "similarity_boost": 0.80},
        }
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.elevenlabs_voice_id}"
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            mp3_path = out_path.with_suffix(".mp3")
            mp3_path.write_bytes(response.content)
            _ffmpeg_convert(mp3_path, out_path, speed=speed if abs(speed - 1.0) > 0.05 else None)
            mp3_path.unlink(missing_ok=True)
            return out_path.exists()
        else:
            log.error(f"[TTS] ElevenLabs {response.status_code}: {response.text[:200]}")
            return False
    except Exception as exc:
        log.error(f"[TTS] ElevenLabs failed: {exc}")
        return False


# ── Azure Neural TTS ──────────────────────────────────────────────────────

def _synthesize_azure(text: str, out_path: Path, speed: float) -> bool:
    """Use Azure Cognitive Services Neural TTS — te-IN-MohanNeural male voice."""
    try:
        import azure.cognitiveservices.speech as speechsdk  # type: ignore

        speech_config = speechsdk.SpeechConfig(
            subscription=config.azure_speech_key,
            region=config.azure_speech_region,
        )
        # Use male neural voice on Azure too
        speech_config.speech_synthesis_voice_name = "te-IN-MohanNeural"
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff44100Hz16BitMonoPcm
        )

        rate_pct = int((speed - 1.0) * 100)
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
        ssml = (
            f"<speak version='1.0' xml:lang='te-IN'>"
            f"<voice name='te-IN-MohanNeural'>"
            f"<prosody rate='{rate_str}'>{text}</prosody>"
            f"</voice></speak>"
        )

        wav_path = out_path.with_suffix(".azure.wav")
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(wav_path))
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )
        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            _ffmpeg_convert(wav_path, out_path)
            wav_path.unlink(missing_ok=True)
            return out_path.exists()
        else:
            log.error(f"[TTS] Azure TTS error: {result.reason}")
            return False
    except ImportError:
        log.error("[TTS] Azure SDK not installed.")
        return False
    except Exception as exc:
        log.error(f"[TTS] Azure failed: {exc}")
        return False


# ── Audio Standardization ─────────────────────────────────────────────────

def _ffmpeg_convert(
    src: Path,
    dst: Path,
    speed: Optional[float] = None,
) -> None:
    """
    Convert any audio file to standard 44100 Hz / stereo / 16-bit WAV.
    Optionally apply atempo speed adjustment.
    """
    filters = [f"aresample={TARGET_SAMPLE_RATE}"]
    if speed is not None and abs(speed - 1.0) > 0.01:
        # atempo range: 0.5–2.0; chain for values outside that range
        speed = max(0.5, min(2.0, speed))
        filters.append(f"atempo={speed:.4f}")

    af = ",".join(filters)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-af", af,
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-acodec", TARGET_FORMAT,
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"[TTS] ffmpeg convert failed:\n{result.stderr[-300:]}")


def _normalize_to_standard(path: Path) -> Path:
    """
    Ensure audio is 44100 Hz / stereo / 16-bit WAV.
    Returns a new path if conversion was needed, else the original.
    """
    # Probe current spec
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    info = result.stdout.strip()

    needs_convert = (
        str(TARGET_SAMPLE_RATE) not in info
        or str(TARGET_CHANNELS) not in info
        or "pcm_s16le" not in info
    )

    if needs_convert:
        std_path = path.with_stem(path.stem + "_std")
        _ffmpeg_convert(path, std_path)
        path.unlink(missing_ok=True)
        return std_path

    return path


# ── Accurate Duration Measurement ─────────────────────────────────────────

def _measure_duration_accurate(audio_path: Path) -> float:
    """
    Measure audio duration by computing exact sample count ÷ sample rate.
    More accurate than format-level duration metadata.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=nb_samples,sample_rate",
        "-of", "csv=p=0",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        parts = result.stdout.strip().split(",")
        if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
            nb_samples = int(parts[0].strip())
            sample_rate = int(parts[1].strip())
            return nb_samples / sample_rate
    except Exception:
        pass

    # Fallback: container duration
    cmd2 = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(audio_path),
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    try:
        return float(result2.stdout.strip())
    except ValueError:
        log.warning(f"[TTS] Could not measure duration of {audio_path.name}")
        return 0.0
