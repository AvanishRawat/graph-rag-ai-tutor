import os
import json
import time
import requests
from pymongo import MongoClient
from typing import List, Dict

from project.graph.schema import Concept, Resource, Example

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("GRAPH_DB", "eng_ai_rag")
CLEAN_COLL = os.environ.get("CLEAN_COLL", "clean_text")

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:latest")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]


def ask_ollama_json(prompt: str) -> dict:

    payload = {
        "model": OLLAMA_MODEL,
        "format": "json",
        "prompt": prompt,
        "stream": False
    }

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=120
        )
        r.raise_for_status()
    except Exception as e:
        print("Ollama request failed:", e)
        return None

    raw = r.json().get("response", "")

    try:
        return json.loads(raw)
    except:
        try:
            fixed = raw.strip()
            fixed = fixed[: fixed.rfind("}")] + "}"
            return json.loads(fixed)
        except:
            return None


def make_batch_prompt(batch_docs: List[Dict]) -> str:
    """
    Larger context window + richer schema
    """

    items = []
    for d in batch_docs:
        items.append({
            "url": d.get("url", "unknown"),
            "text": d.get("text", "")[:3000]
        })

    return f"""
        You are extracting structured knowledge from course materials (PDF, slides, websites, notes).
        Return ONLY valid JSON matching the exact schema below.

        SCHEMA TO RETURN:
        {{
        "items": [
            {{
            "url": "",
            "concepts": [
                {{
                "title": "",
                "aliases": [],
                "difficulty": "easy|medium|hard",
                "definitions": [],
                "formulas": []
                }}
            ],
            "examples": [
                {{
                "text": ""
                }}
            ],
            "resource": {{
                "type": "pdf|web|slide|video",
                "title": "",
                "url": ""
            }}
            }}
        ]
        }}

        RULES:
        - Extract ALL concepts present in the text.
        - Extract ALL definitions for each concept.
        - If formulas (LaTeX, inline math like x^2 + y^2, or multi-line equations) appear, include them in "formulas".
        - Definitions should be split into small meaningful segments.
        - "examples" should contain short illustrative examples or snippets from the text.
        - JSON MUST be strictly valid.

        DOCUMENTS:
        {json.dumps(items, indent=2)}
        """

def run_extraction(batch_size=1):
    cleaned = list(db[CLEAN_COLL].find({}))
    total = len(cleaned)

    print(f"\nTotal cleaned docs: {total}")
    print(f"Using batch size: {batch_size}\n")

    concepts_coll = db.concepts
    examples_coll = db.examples
    resources_coll = db.resources

    batches = (total + batch_size - 1) // batch_size

    for bi in range(batches):
        print(f"=== Processing batch {bi+1} / {batches} ===")

        start = bi * batch_size
        end = min(start + batch_size, total)
        batch_docs = cleaned[start:end]

        prompt = make_batch_prompt(batch_docs)

        result = ask_ollama_json(prompt)

        # Retry once if failed
        if not result:
            print("Retrying batch once...")
            time.sleep(1.0)
            result = ask_ollama_json(prompt)

        if not result or "items" not in result:
            print(" Skipping batch — still invalid JSON\n")
            continue

        for item in result["items"]:
            url = item.get("url")
            if not url:
                continue

            rid = f"res-{abs(hash(url)) % (10**9)}"
            res_raw = item.get("resource", {})

            res = Resource(
                id=rid,
                type=res_raw.get("type", "web"),
                url=url,
                title=res_raw.get("title", url)
            )

            resources_coll.update_one(
                {"id": res.id},
                {"$set": res.__dict__},
                upsert=True
            )

            for c in item.get("concepts", []):
                if not c.get("title"):
                    continue

                cid = f"c-{abs(hash(c['title'])) % (10**9)}"

                concept = Concept(
                    id=cid,
                    title=c["title"],
                    aliases=c.get("aliases", []),
                    difficulty=c.get("difficulty", "unknown"),
                    definitions=c.get("definitions", []),
                    formulas=c.get("formulas", [])
                )

                concepts_coll.update_one(
                    {"id": concept.id},
                    {"$set": concept.__dict__},
                    upsert=True
                )

            for ex in item.get("examples", []):
                txt = ex.get("text", "").strip()
                if not txt:
                    continue

                exid = f"ex-{abs(hash(txt)) % (10**9)}"

                example = Example(
                    id=exid,
                    text=txt,
                    source_url=url,
                    related_concepts=[]
                )

                examples_coll.update_one(
                    {"id": example.id},
                    {"$set": example.__dict__},
                    upsert=True
                )

        print("Batch written to Mongo\n")
        time.sleep(5)

    print("\nDONE — all batches processed.\n")
