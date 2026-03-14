import os
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import Client as OllamaClient


ROOT = Path(__file__).resolve().parents[1]
KG_PATH = ROOT / "graph" / "kg.gpickle"

print("[pipeline] Loading Knowledge Graph...")

with open(KG_PATH, "rb") as f:
    KG = pickle.load(f)

print("[pipeline] KG loaded:",
      "Nodes =", KG.number_of_nodes(),
      "| Edges =", KG.number_of_edges())

print("[pipeline] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("[pipeline] Precomputing concept embeddings...")

CONCEPT_EMBEDS = {}
CONCEPT_TEXTS = {}

for nid, data in KG.nodes(data=True):
    if data.get("type") != "concept":
        continue

    text = data.get("title", "")
    defs = " ".join(data.get("definitions", []))
    full = text + " " + defs

    emb = embedder.encode([full])[0]

    CONCEPT_TEXTS[nid] = full
    CONCEPT_EMBEDS[nid] = emb

print("[pipeline] Concept embeddings ready:", len(CONCEPT_EMBEDS))

def retrieve_concepts(query, top_k=5, min_sim=0.40):
    print("[pipeline] Retrieving concepts...")

    q_emb = embedder.encode([query])[0]

    scores = []
    for cid, c_emb in CONCEPT_EMBEDS.items():
        sim = cosine(q_emb, c_emb)
        if sim >= min_sim:
            scores.append((cid, sim))

    if not scores:
        print("[pipeline] No concepts above similarity threshold.")
        return []

    scores.sort(key=lambda x: x[1], reverse=True)

    final = [cid for cid, _ in scores[:top_k]]
    print("[pipeline] Retrieved:", final)
    return final

def expand_subgraph(concept_ids, prereq_depth=2):
    print("[pipeline] Expanding subgraph...")

    final = set(concept_ids)

    #prerequisite expansion
    frontier = set(concept_ids)
    for _ in range(prereq_depth):
        new = set()
        for c in frontier:
            for a, b, d in KG.in_edges(c, data=True):
                if d.get("relation") == "prereq_of":
                    new.add(a)
        frontier = new
        final |= new

    #near-transfer
    for c in list(final):
        for _, b, d in KG.edges(c, data=True):
            if d.get("relation") == "near_transfer":
                final.add(b)

    for c in list(final):
        # incoming
        for a, _, d in KG.in_edges(c, data=True):
            if d.get("relation") in ("explains", "exemplifies"):
                final.add(a)
        # outgoing
        for _, b, d in KG.out_edges(c, data=True):
            if d.get("relation") in ("explains", "exemplifies"):
                final.add(b)

    print("[pipeline] Subgraph size:", len(final), "nodes")
    return KG.subgraph(final).copy()

def graph_to_context(G_sub):
    print("[pipeline] Preparing context...")
    lines = []

    for nid, data in G_sub.nodes(data=True):

        if data["type"] == "concept":
            lines.append(f"CONCEPT: {data['title']}")
            for d in data.get("definitions", []):
                lines.append(f"  - {d}")

        elif data["type"] == "example":
            lines.append(f"EXAMPLE: {data.get('text','')}")

        elif data["type"] == "resource":
            lines.append(f"RESOURCE: {data.get('title','')} ({data.get('url','')})")

    for a, b, d in G_sub.edges(data=True):
        if d.get("relation") == "prereq_of":
            lines.append(f"PREREQ: {KG.nodes[a]['title']} -> {KG.nodes[b]['title']}")

    return "\n".join(lines)


OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://172.25.69.74:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:latest")
ollama = OllamaClient(host=OLLAMA_URL)

SYSTEM_PROMPT = """
        You are an AI Tutor for an Artificial Intelligence course.

        RULES FOR ALL ANSWERS:
        1. Start with an intuitive explanation.
        2. Progress to a formal mathematical explanation if the context provides formulas.
        3. Use definitions, equations, and notation exactly as they appear in the provided context.
        4. If the question involves a mechanism (e.g., an algorithm, architecture, or loss function), include its step-by-step computation.
        5. If the question asks for code, provide a minimal, correct example consistent with the concepts in the context.
        6. Reference only the concepts, resources, and examples found in the context block.
        7. At the end of every answer, include the section headers:
        === KNOWLEDGE GRAPH CONCEPTS USED ===
        === RESOURCES USED ===
    """

def answer_query(query: str) -> str:

    print("\n==============================")
    print("[pipeline] Received question:")
    print(query)
    print("==============================\n")


    top_concepts = retrieve_concepts(query)
    G_sub = expand_subgraph(top_concepts)
    context = graph_to_context(G_sub)

    # LLM call
    prompt = f"""{SYSTEM_PROMPT}

        ### USER QUESTION
        {query}

        ### KNOWLEDGE GRAPH CONTEXT
        {context}

        ### FINAL ANSWER
    """

    print("[pipeline] Ollama thinking... (sending prompt)")
    print("-----------------------------------")

    result = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)

    print("[pipeline] Ollama responded.")
    print("-----------------------------------")

    return result["response"]
