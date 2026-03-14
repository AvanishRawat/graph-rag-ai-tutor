# project/rag/prompt.py
"""
Prompt builder for Graph RAG. Provides:
 - SYSTEM_PROMPT (string)
 - build_generation_prompt(question, subgraph, seed_concepts, max_context_chars=2500)
"""

import os
from typing import List
import networkx as nx
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("GRAPH_DB", "eng_ai_rag")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

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

def fetch_resource_excerpt(url: str, max_chars: int = 2000):
    rec = db.resources.find_one({"url": url}) or {}
    text = ""
    if rec.get("text"):
        text = rec.get("text")[:max_chars]
    else:
        doc = db.clean_text.find_one({"url": url}) or {}
        text = (doc.get("text") or "")[:max_chars]
    return text

def build_generation_prompt(question: str, subgraph: nx.Graph, seed_concepts: List[str], max_context_chars: int = 2500):
    concept_blocks = []
    for n, d in subgraph.nodes(data=True):
        if d.get("type") == "concept":
            title = d.get("title","")
            defs = " ".join(d.get("definitions",[]))[:max_context_chars]
            aliases = d.get("aliases", [])
            difficulty = d.get("difficulty", "")
            concept_blocks.append(f"[CONCEPT] {title}\nDefinitions: {defs}\nAliases: {aliases}\nDifficulty: {difficulty}\n")

    resource_blocks = []
    for n, d in subgraph.nodes(data=True):
        if d.get("type") == "resource":
            url = d.get("url", "")
            title = d.get("title", "") or url
            excerpt = fetch_resource_excerpt(url, max_chars=1000)
            resource_blocks.append(f"[RESOURCE] {title}\nURL: {url}\nExcerpt: {excerpt}\n")

    example_blocks = []
    for n, d in subgraph.nodes(data=True):
        if d.get("type") == "example":
            text = d.get("text","")[:1000]
            source = d.get("source_url","")
            example_blocks.append(f"[EXAMPLE] Source: {source}\nText: {text}\n")

    context_parts = ["CONTEXT (use ONLY these):", "\n-- Concepts --"]
    context_parts.extend(concept_blocks)
    context_parts.append("\n-- Resources --")
    context_parts.extend(resource_blocks)
    context_parts.append("\n-- Examples --")
    context_parts.extend(example_blocks)

    context_text = "\n".join(context_parts)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Below is the context extracted from the knowledge graph.\n"
        "USE ONLY THIS CONTEXT to answer the student question.\n\n"
        f"{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        "BEGIN ANSWER:\n"
    )
    return prompt
