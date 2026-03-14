import os
from typing import List, Set, Tuple, Dict
import numpy as np
import networkx as nx
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("GRAPH_DB", "eng_ai_rag")
GRAPH_PATH = os.environ.get("GRAPH_PATH", "project/graph/kg.gpickle")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

_G = None
def _load_graph():
    global _G
    if _G is None:
        import pickle, pathlib
        p = pathlib.Path(GRAPH_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Graph file not found at {p}")
        with open(p, "rb") as f:
            _G = pickle.load(f)
    return _G

_EMBED = None
def _embedder():
    global _EMBED
    if _EMBED is None:
        _EMBED = SentenceTransformer(EMBED_MODEL)
    return _EMBED

def embed_text(text: str):
    if not text:
        return np.zeros(_embedder().get_sentence_embedding_dimension(), dtype=float)
    return _embedder().encode([text], convert_to_numpy=True)[0]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _precompute_concept_embeddings(G: nx.Graph):
    concepts = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "concept"]
    ids = [n for n, _ in concepts]
    texts = [ (d.get("title","") + " " + " ".join(d.get("definitions",[]))).strip() for _, d in concepts ]
    embs = _embedder().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return ids, texts, embs

def rank_concepts_by_query(query: str, G: nx.Graph, concept_ids: List[str], concept_embs: np.ndarray,
                           top_k: int = 10, alpha: float = 0.7):
    qv = embed_text(query)
    sims = [cosine(qv, e) for e in concept_embs]

    q_tokens = set(query.lower().split())
    overlap_scores = []
    for cid in concept_ids:
        title = (G.nodes[cid].get("title","") or "").lower()
        title_tokens = set(title.split())
        overlap_scores.append(len(q_tokens & title_tokens) / max(1, len(title_tokens)) if title_tokens else 0.0)

    combined = []
    for i, cid in enumerate(concept_ids):
        score = alpha * sims[i] + (1-alpha) * overlap_scores[i]
        combined.append((cid, float(score), {"title": G.nodes[cid].get("title",""), "sim": float(sims[i]), "overlap": float(overlap_scores[i])}))
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]

def expand_prereq_chain(G: nx.Graph, seeds: List[str], max_depth: int = 2):
    result = set(seeds)

    frontier = set(seeds)
    for _ in range(max_depth):
        parents = set()
        for node in frontier:
            for u, v, d in G.in_edges(node, data=True):
                if d.get("relation") == "prereq_of":
                    parents.add(u)
        if not parents:
            break
        result.update(parents)
        frontier = parents

    frontier = set(seeds)
    for _ in range(max_depth):
        children = set()
        for node in frontier:
            for u, v, d in G.out_edges(node, data=True):
                if d.get("relation") == "prereq_of":
                    children.add(v)
        if not children:
            break
        result.update(children)
        frontier = children
    return result

def get_near_transfer_neighbors(G: nx.Graph, seeds: List[str], top_per: int = 6):
    out = set()
    for c in seeds:
        neigh = []
        for _, v, d in G.out_edges(c, data=True):
            if d.get("relation") == "near_transfer":
                neigh.append((v, d.get("similarity", 0.0)))
        neigh.sort(key=lambda x: x[1], reverse=True)
        out.update([n for n,_ in neigh[:top_per]])
    return out

def include_resource_example_nodes(G: nx.Graph, concept_ids: List[str]):
    out = set()
    for c in concept_ids:
        for u, _, d in G.in_edges(c, data=True):
            if d.get("relation") in ("explains", "exemplifies"):
                out.add(u)
    return out

def retrieve_subgraph_for_query(question: str, top_k: int = 3, prereq_depth: int = 2):
    G = _load_graph()
    concept_ids, _, concept_embs = _precompute_concept_embeddings(G)
    ranked = rank_concepts_by_query(question, G, concept_ids, concept_embs, top_k=top_k)
    seed_concepts = [r[0] for r in ranked]

    prereqs = expand_prereq_chain(G, seed_concepts, max_depth=prereq_depth)
    siblings = get_near_transfer_neighbors(G, seed_concepts, top_per=6)
    related_res_ex = include_resource_example_nodes(G, list(prereqs | siblings | set(seed_concepts)))

    node_set = set(seed_concepts) | prereqs | siblings | related_res_ex
    sub = G.subgraph(node_set).copy()

    used_concepts = [n for n, d in sub.nodes(data=True) if d.get("type") == "concept"]
    used_resources = [n for n, d in sub.nodes(data=True) if d.get("type") == "resource"]
    used_examples = [n for n, d in sub.nodes(data=True) if d.get("type") == "example"]

    return {
        "seed_concepts": seed_concepts,
        "ranked": ranked,
        "subgraph": sub,
        "used_concepts": used_concepts,
        "used_resources": used_resources,
        "used_examples": used_examples
    }
