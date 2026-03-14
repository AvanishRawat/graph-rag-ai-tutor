import os
import json
from bs4 import BeautifulSoup
from pathlib import Path
from project.database.store_clean import store_clean_document

# ROOT/eng-ai-agents
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "website"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def process_one_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)

    url = record.get("url", "")
    html = record.get("html", "")

    cleaned_text = clean_html_to_text(html)

    clean_record = {
        "url": url,
        "title": record.get("title", ""),
        "text": cleaned_text,
    }

    # Save to data/clean/
    out_path = CLEAN_DIR / path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_record, f, ensure_ascii=False, indent=2)

    # Save to MongoDB
    store_clean_document(clean_record)

    print(f"[CLEANED] {path.name}")


def run_cleaning():
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".json"):
            continue

        process_one_file(RAW_DIR / filename)

    print(" Cleaning completed")
