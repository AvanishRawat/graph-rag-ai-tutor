# 🧠 Graph RAG AI Tutor

An end-to-end **Retrieval-Augmented Generation (RAG) system** that ingests course materials (websites, PDFs, YouTube transcripts), builds a **knowledge graph**, and answers student questions through an intelligent chat interface — grounding every answer in structured curriculum context.

Built for NJIT CS370: Engineering AI Agents.

---

## What It Does

Students ask questions like *"What is attention in transformers?"* or *"Explain the variational lower bound"* and receive:

- **Structured answers** grounded in course-specific concepts, not generic internet knowledge
- **Step-by-step explanations** that progress from intuition → math → code
- **Citation of source materials** (lecture pages, papers, notebooks) used to generate the answer
- **Prerequisite-aware context** — the system traces concept dependency chains so answers build on what you already know

The system is entirely self-contained: it crawls your course website, extracts knowledge, builds a graph, and serves a chat UI — no external APIs required.

---

## How It Works

### Architecture Overview

```
Course Website / PDFs / YouTube
          │
          ▼
   [ Ingestion Pipeline ]
    crawler.py → cleaning.py
    pdf_parser.py → youtube_ingest.py
          │
          ▼
   [ Knowledge Extraction ]   ← Qwen2.5 (Ollama, local LLM)
    extract_concepts.py
    → Concepts, Resources, Examples → MongoDB
          │
          ▼
   [ Knowledge Graph Builder ]
    build_graph.py
    → NetworkX DiGraph (kg.gpickle)
    → Node types: concept / resource / example
    → Edge types: prereq_of / near_transfer / explains / exemplifies
          │
          ▼
   [ Graph RAG Retrieval ]
    retriever.py
    → Embed query with all-MiniLM-L6-v2
    → Rank concepts by cosine similarity + token overlap
    → Expand: prereq chains (depth=2) + near-transfer neighbors
    → Include linked resources and examples
          │
          ▼
   [ Answer Generation ]       ← Qwen2.5 (Ollama, local LLM)
    prompt.py → generator.py
    → Context-grounded prompt with concepts + resources + examples
          │
          ▼
   [ Flask Chat UI ]
    app.py → http://localhost:5001
```

### Key Design Decisions

**Hybrid retrieval scoring**: Concepts are ranked by a weighted combination of semantic similarity (α=0.7) and token overlap with the query title, reducing false positives from pure embedding search.

**Prerequisite chain expansion**: The retriever walks `prereq_of` edges in both directions (depth=2) so answers automatically include foundational concepts students need to understand the answer.

**Near-transfer neighbors**: Concepts connected by `near_transfer` edges (cosine similarity ≥ 0.45 between their embeddings) are included to surface related ideas — like pulling "RNN Language Models" when asking about "Transformers."

**LLM-extracted prerequisites**: `extract_prereqs_llm()` in `build_graph.py` uses Qwen2.5 to infer prerequisite chains between concepts, producing directed `prereq_of` edges that go beyond what keyword matching can find.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Scraping** | `requests`, `BeautifulSoup4` |
| **PDF Parsing** | `pdfplumber` |
| **YouTube Transcripts** | `yt-dlp` (VTT subtitle extraction) |
| **Database** | MongoDB (raw docs, clean text, embeddings, concepts) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Knowledge Graph** | `NetworkX` DiGraph (persisted as `.gpickle`) |
| **Local LLM** | Qwen2.5 via [Ollama](https://ollama.ai) |
| **Graph Visualization** | `pyvis` (interactive HTML) |
| **Backend** | Flask |

---

## Results / Demo

The system was evaluated on 3 representative questions. Key findings from `m4_results.json`:

**Q: "What is attention in transformers and can you provide a Python example?"**
- Retrieved 30-node subgraph including: *Transformers and Self-Attention*, *Self-Attention Mechanism*, *Single-head self-attention*, *Vision Transformer*, *Encoder-Decoder Architecture*, and 5 course resource pages
- Top seed concept scored **0.618** combined relevance (sim=0.598, overlap=0.667)
- Generated correct `Attention(Q,K,V) = softmax(QK^T / √d_k)V` formula with working PyTorch `SelfAttention` class

**Q: "What is CLIP and how is it used in computer vision?"**
- Correctly surfaced *Vision Language Models (VLMs)* as the relevant concept cluster
- Retrieved 34-node subgraph spanning VLMs, Vision Transformers, Object Detection, Instance Segmentation
- Answer included correct CLIP contrastive loss formulation and working Python example using the `clip` library

**Q: "Explain the variational lower bound and Jensen's inequality"**
- Gracefully handled out-of-distribution query — concept retrieval returned low-relevance concepts, and the system transparently noted the context gap while still providing a correct general explanation
- Demonstrates robustness: system doesn't hallucinate course-specific grounding when it doesn't exist

**Graph Stats:**
- Edge types: `prereq_of`, `near_transfer`, `explains`, `exemplifies`
- Embedding similarity threshold for `near_transfer`: **0.45**
- Hybrid retrieval weight: **α=0.7** semantic, **0.3** token overlap

---

## Project Structure

```
graph-rag-ai-tutor/
├── ingest/
│   ├── crawler.py          # BFS web crawler with allowed-prefix filtering
│   ├── cleaning.py         # HTML → clean text (BeautifulSoup)
│   ├── pdf_parser.py       # PDF text extraction (pdfplumber)
│   └── youtube_ingest.py   # YouTube VTT transcript downloader
├── graph/
│   ├── schema.py           # Concept, Resource, Example dataclasses
│   ├── extract_concepts.py # Ollama LLM → structured concept extraction
│   ├── build_graph.py      # NetworkX graph builder (embeddings + LLM prereqs)
│   └── visualize.py        # pyvis interactive graph export
├── rag/
│   ├── retriever.py        # Query → subgraph retrieval (embedding + graph walk)
│   ├── prompt.py           # Prompt builder (concept + resource + example blocks)
│   ├── generator.py        # Ollama LLM inference wrapper
│   ├── pipeline.py         # End-to-end query answering pipeline
│   └── ai_tutor.py         # AITutor class (top-level interface)
├── app.py                  # Flask chat UI (http://localhost:5001)
├── requirements.txt
└── ingestion_pipeline.py   # Orchestrates full ingest: crawl → clean → PDF → YT
```

---

## How to Run

### Prerequisites
- Python 3.11+
- MongoDB running locally
- [Ollama](https://ollama.ai) installed with Qwen2.5 pulled:
```bash
ollama pull qwen2.5
```

### Setup
```bash
git clone https://github.com/AvanishRawat/graph-rag-ai-tutor.git
cd graph-rag-ai-tutor
pip install -r requirements.txt
```

### 1. Run the Ingestion Pipeline
```bash
python ingestion_pipeline.py
```
Crawls the course website, cleans HTML, extracts PDFs and YouTube transcripts, and stores everything in MongoDB.

### 2. Build the Knowledge Graph
```bash
python graph/build_graph.py
```
Builds the NetworkX graph with semantic `near_transfer` edges, LLM-inferred `prereq_of` edges, and `explains`/`exemplifies` edges. Output saved to `graph/kg.gpickle`.

### 3. Start the Chat UI
```bash
python app.py
```
Then open **http://localhost:5001** and start asking questions.

---

## Requirements

```
flask
pymongo
networkx
numpy
sentence-transformers
torch
pdfplumber
beautifulsoup4
requests
pyvis
ollama
yt-dlp
```

---

## License

MIT
