# Adcc-python — Telugu Audio Description Pipeline

An AI-powered pipeline that automatically generates synchronized **Telugu Audio Description (AD)** tracks for movies using the **Google Gemini API**.

---

## 🎬 What It Does

```
Input: movie.mp4
  → Detects silence gaps between dialogues
  → Extracts video frames around each gap
  → Gemini Vision analyses the scene visually
  → Generates concise English scene description
  → Translates to natural, cinematic Telugu
  → Optimizes text to fit precisely inside the gap
  → Synthesizes Telugu speech (TTS)
  → Mixes narration into a synchronized MP3 track
Output: audio_description_telugu.mp3
```

---

## 📁 Project Structure

```
Adcc-python/
├── main.py                  ← Entry point + pipeline orchestrator
├── config.py                ← All configuration (env + defaults)
├── logger.py                ← Coloured logging to console + file
├── models.py                ← Shared data classes
├── gap_detector.py          ← STEP 2: Silence gap detection (Whisper + FFmpeg)
├── scene_analyzer.py        ← STEP 3: Scene understanding (Gemini Vision)
├── translator.py            ← STEP 4: English → Telugu (Gemini)
├── duration_optimizer.py    ← STEP 5: Fit narration within gap duration
├── tts_engine.py            ← STEP 6: Telugu TTS (gTTS / ElevenLabs / Azure)
├── audio_mixer.py           ← STEP 7+8: Timeline mixing + export
├── requirements.txt
├── .env.example             ← Copy to .env and fill in API keys
└── README.md
```

---

## ⚙️ Setup

### 1. Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on PATH
- Google Gemini API key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 4. Run

```bash
# Basic — generate AD track only
python main.py movie.mp4

# Generate AD track + merge into final movie
python main.py movie.mp4 --merge

# Use ElevenLabs for higher quality TTS
python main.py movie.mp4 --tts-provider elevenlabs --merge

# Custom output directory
python main.py movie.mp4 --output-dir results/
```

---

## 📤 Outputs

| File | Description |
|---|---|
| `output/<name>_audio_description_telugu.mp3` | Synchronized Telugu AD track |
| `output/<name>_with_AD.mp4` | Merged movie with AD (if `--merge`) |
| `output/<name>_ad_metadata.json` | Timeline JSON with all narration data |
| `output/logs/pipeline_*.log` | Full processing log |

---

## 🔧 TTS Providers

| Provider | Quality | Cost | Setup |
|---|---|---|---|
| `gtts` (default) | Good | Free | None |
| `elevenlabs` | Excellent | Paid | `ELEVENLABS_API_KEY` in `.env` |
| `azure` | Excellent | Paid | `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION` |

---

## 🧪 CLI Options

```
python main.py video.mp4 [options]

Options:
  --merge               Merge AD track with original movie audio
  --tts-provider        gtts | elevenlabs | azure
  --workers N           Parallel worker threads (default: 4)
  --no-cache            Disable Gemini response caching
  --output-dir PATH     Output directory (default: output/)
```
