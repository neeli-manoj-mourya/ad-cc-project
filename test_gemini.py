"""Quick test to diagnose Gemini API connection issues."""
import os
import ssl
import traceback

os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ.setdefault("SSL_CERT_FILE", "")
_orig = ssl.create_default_context
def _nv(*a, **k):
    c = _orig(*a, **k)
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c
ssl.create_default_context = _nv

import httpx
_oi = httpx.Client.__init__
def _pi(self, *a, **k):
    k.setdefault("verify", False)
    _oi(self, *a, **k)
httpx.Client.__init__ = _pi

from dotenv import load_dotenv
load_dotenv()

from google import genai

api_key = os.getenv("GEMINI_API_KEY", "")
print(f"API Key: {api_key[:10]}...{api_key[-4:]} (len={len(api_key)})")

client = genai.Client(api_key=api_key)
try:
    r = client.models.generate_content(model="gemini-2.5-flash", contents="Say hi in one word")
    print(f"SUCCESS: {r.text}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
