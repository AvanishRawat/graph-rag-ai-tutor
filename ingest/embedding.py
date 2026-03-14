import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from project.database.store_embeddings import save_embedding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def run_embeddings():
    print("Embedding cleaned files...")

    files = sorted(list(CLEAN_DIR.glob("*.json")))
    print(f"Found {len(files)} cleaned files.")

    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Use filename as ID
        doc_id = f.stem

        text = data.get("text", "").strip()

        if not text:
            print(f"[SKIP] {doc_id}: empty text")
            continue

        embedding = model.encode(text).tolist()

        # Save into MongoDB
        save_embedding(
            doc_id=doc_id,
            url=data.get("url", ""),
            text=text,
            embedding=embedding,
        )

        print(f"[EMBEDDED] {doc_id}")

    print("Done embedding all documents!")
