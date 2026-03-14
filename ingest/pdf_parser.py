import json
import pdfplumber
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text.append(txt)
    return "\n".join(text)

def main():
    filepath = input("PDF path: ").strip()
    pdf_path = Path(filepath)
    if not pdf_path.exists():
        print("File not found.")
        return

    text = parse_pdf(pdf_path)
    out = {
        "url": f"file:{pdf_path.name}",
        "title": pdf_path.name,
        "clean_text": text
    }

    fname = f"{hash(pdf_path)}.json"
    with open(OUT_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK]Saved to {OUT_DIR / fname}")

if __name__ == "__main__":
    main()
