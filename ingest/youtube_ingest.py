import subprocess
import json
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VENVDIR = PROJECT_ROOT / ".venv"
YTDLP = VENVDIR / "bin" / "yt-dlp"

if not YTDLP.exists():
    raise RuntimeError(
        f"ERROR: yt-dlp not found in venv.\n"
        f"Expected at: {YTDLP}\n"
        f"Install with:\n"
        f"   /workspaces/eng-ai-agents/.venv/bin/pip install yt-dlp"
    )

def video_id_from_url(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def download_transcript(url: str) -> str:
    print(f"\nUsing yt-dlp at: {YTDLP}\n")

    TEMP_DIR = PROJECT_ROOT / "data" / "tmp_subs"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for f in TEMP_DIR.glob("*"):
        f.unlink()

    output_template = str(TEMP_DIR / "subs")

    cmd = [
        str(YTDLP),
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "-o", output_template,
        url
    ]

    print("Running yt-dlp...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("\n=== STDERR ===")
    print(result.stderr)

    vtt_files = list(TEMP_DIR.glob("*.vtt"))

    print("\nLooking for subtitles in:", TEMP_DIR)
    print("Found:", vtt_files)

    if not vtt_files:
        print(f" No subtitles found for {url}")
        return ""

    vtt_file = vtt_files[0]

    lines = []
    for line in vtt_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("WEBVTT"): continue
        if "-->" in line: continue
        if line.strip().isdigit(): continue
        if not line.strip(): continue
        lines.append(line.strip())

    return " ".join(lines)


def save_transcript(url: str, text: str):
    out_path = OUT_DIR / f"{hash(url)}.json"
    data = {
        "url": url,
        "title": "YouTube Transcript",
        "clean_text": text
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_path

def ingest_youtube_video(url: str):
    text = download_transcript(url)
    out_path = save_transcript(url, text)
    return out_path
