"""Quick test: verify Edge TTS produces correct-length Telugu audio."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, ".")
from pathlib import Path
from tts_engine import _synthesize_edge, _ffmpeg_convert, _measure_duration_accurate

Path("temp/tts").mkdir(parents=True, exist_ok=True)
out = Path("temp/tts/test_fix.wav")

text = "చిరునవ్వుతో ఆమె అతడిని దాటి వెళ్తుంది."
print(f"Text length: {len(text)} chars")

ok = _synthesize_edge(text, out, speed=1.0)
print(f"Synthesis OK: {ok}")

if ok and out.exists():
    dur = _measure_duration_accurate(out)
    print(f"Duration: {dur:.3f}s")
    print(f"File size: {out.stat().st_size} bytes")
    if dur < 10:
        print("PASS - duration is reasonable")
    else:
        print("FAIL - duration too long")
else:
    print("FAIL - synthesis failed")
