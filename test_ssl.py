"""Quick SSL test for edge-tts"""
import ssl

# Patch before importing edge_tts
_orig = ssl.create_default_context
def _unverified(*a, **kw):
    ctx = _orig(*a, **kw)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = _unverified

import asyncio
import edge_tts
from pathlib import Path

Path("temp/tts").mkdir(parents=True, exist_ok=True)

async def main():
    c = edge_tts.Communicate(text="హలో ప్రపంచం", voice="te-IN-MohanNeural", rate="+0%")
    await c.save("temp/tts/test.mp3")
    print(f"OK: {Path('temp/tts/test.mp3').stat().st_size} bytes")

asyncio.run(main())
