import os
import requests
from typing import Optional

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://172.25.69.74:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:latest")
OLLAMA_API = f"{OLLAMA_HOST}/api/generate"

def ask_ollama(prompt: str, model: Optional[str] = None) -> str:
    m = model or OLLAMA_MODEL
    payload = {"model": m, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_API, json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
    data = r.json()
    out = data.get("response", "")
    if isinstance(out, dict) and "content" in out:
        out = out["content"]
    if isinstance(out, list):
        parts = []
        for p in out:
            if isinstance(p, dict):
                parts.append(p.get("text") or str(p))
            else:
                parts.append(str(p))
        out = "".join(parts)
    return str(out)

def test_connection():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False
