import os
import json
import pickle
from pathlib import Path
from pymongo import MongoClient
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from project.graph.schema import Concept, Resource, Example

# Ollama client
from ollama import Client as OllamaClient

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://172.25.69.74:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:latest")

ollama_client = OllamaClient(host=OLLAMA_URL)


GRAPH_DIR = Path(__file__).resolve().parents[0]
GRAPH_PATH = GRAPH_DIR / "kg.gpickle"
GRAPH_JSON = GRAPH_DIR / "kg.json"


MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("GRAPH_DB", "eng_ai_rag")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]


CONCEPTS_COLL = "concepts"
RESOURCES_COLL = "resources"
EXAMPLES_COLL = "examples"

# Embedding model
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

NEAR_TRANSFER_THRESHOLD = float(os.environ.get("NEAR_TRANSFER_THRESH", 0.45))


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))



def extract_prereqs_llm(concept_list):
    titles = [c["title"] for c in concept_list]

    prompt = f"""
        You are an AI expert constructing prerequisite chains.

        Given the concepts:

        {json.dumps(titles, indent=2)}

        Return ONLY lines in this exact format:

        prereq: A -> B

        Meaning "A must be learned before B".

        Use ONLY concepts from the list.
        Return nothing else.
    """

    response = ollama_client.generate(model=OLLAMA_MODEL, prompt=prompt)
    text = response["response"]

    prereqs = []

    for line in text.split("\n"):
        if line.startswith("prereq:"):
            try:
                body = line.replace("prereq:", "").strip()
                a, b = body.split("->")
                prereqs.append((a.strip(), b.strip()))
            except:
                pass

    return prereqs

def build_graph(limit_concepts: int = None):

    print("Loading data from MongoDB...")

    concepts_cursor = db[CONCEPTS_COLL].find({})
    if limit_concepts:
        concepts_cursor = concepts_cursor.limit(limit_concepts)
    concepts = list(concepts_cursor)

    resources = list(db[RESOURCES_COLL].find({}))
    examples = list(db[EXAMPLES_COLL].find({}))

    if not concepts:
        raise RuntimeError("No concepts in DB — run extract_concepts.py first.")

    G = nx.DiGraph()

    print("Adding concepts...")
    for c in concepts:
        G.add_node(
            c["id"],
            type="concept",
            title=c["title"],
            definitions=c.get("definitions", []),
            aliases=c.get("aliases", []),
            difficulty=c.get("difficulty", "unknown"),
        )

    print("Adding resources...")
    for r in resources:
        rid = r["id"]
        G.add_node(rid, type="resource", title=r.get("title", ""), url=r.get("url", ""))

        r_text = f"{r.get('title', '')} {r.get('url', '')}".lower()
        for c in concepts:
            if c["title"].lower() in r_text:
                G.add_edge(rid, c["id"], relation="explains")

    print("Adding examples...")
    for e in examples:
        eid = e["id"]
        G.add_node(eid, type="example", text=e.get("text", ""), source_url=e.get("source_url", ""))

        ex_text = e.get("text", "").lower()
        for c in concepts:
            if c["title"].lower() in ex_text:
                G.add_edge(eid, c["id"], relation="exemplifies")

    print("Computing embeddings...")
    concept_ids = [c["id"] for c in concepts]
    concept_texts = [
        c["title"] + " " + " ".join(c.get("definitions", []))
        for c in concepts
    ]

    embeddings = embedder.encode(concept_texts, convert_to_numpy=True)

    print("Adding near_transfer edges...")
    n = len(concept_ids)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine(embeddings[i], embeddings[j])
            if sim >= NEAR_TRANSFER_THRESHOLD:
                a = concept_ids[i]
                b = concept_ids[j]
                G.add_edge(a, b, relation="near_transfer", similarity=float(sim))
                G.add_edge(b, a, relation="near_transfer", similarity=float(sim))

    print("Extracting prereqs via Qwen2.5...")

    prereq_pairs = extract_prereqs_llm(concepts)

    title_to_id = {c["title"].lower(): c["id"] for c in concepts}

    count = 0
    for a, b in prereq_pairs:
        a_id = title_to_id.get(a.lower())
        b_id = title_to_id.get(b.lower())
        if a_id and b_id and a_id != b_id:
            G.add_edge(a_id, b_id, relation="prereq_of")
            count += 1

    print("Added prereq_of edges:", count)

    print("Saving...")
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)

    with open(GRAPH_JSON, "w") as f:
        json.dump(nx.node_link_data(G), f, indent=2)

    print("Graph built.")
    print("Nodes:", G.number_of_nodes(), "| Edges:", G.number_of_edges())
    return G
