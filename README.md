---
title: RAG Knowledge Agent
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# ⬡ Multimodal RAG Knowledge Base Agent

A production-grade, fully documented **Multimodal Retrieval-Augmented Generation** system for startup internal knowledge bases. Processes text PDFs, embedded images, diagrams, and scanned forms — all searchable through a single conversational interface.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    OFFLINE PHASE (ingest.py)                       │
│                                                                    │
│  PDFs  →  PyMuPDF  →  Text Chunks ──────────────────────────────┐ │
│                    ↘                                             │ │
│                      Images  →  GPT-4o-mini Vision  →  Captions ┤ │
│                                                                  ↓ │
│                                         HuggingFace Embeddings   │ │
│                                                  ↓               │ │
│                                            ChromaDB (local)      │ │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                    ONLINE PHASE (query.py + app.py)                │
│                                                                    │
│  User Query + Optional Image                                       │
│        ↓                                                           │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │              LangGraph Agent Pipeline                    │      │
│  │                                                          │      │
│  │  [1] Query Analyser  →  Intent classification +         │      │
│  │      Node               Query rewriting/expansion       │      │
│  │          ↓                                              │      │
│  │  [2] Retrieval Node  →  Embed query → ChromaDB top-K   │      │
│  │                          + Vision describe upload       │      │
│  │          ↓                                              │      │
│  │  [3] Generation Node →  Build grounded prompt →        │      │
│  │                          GPT-4o-mini → Cited Answer    │      │
│  └─────────────────────────────────────────────────────────┘      │
│        ↓                                                           │
│  Gradio Chat UI (app.py)                                           │
└────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- An OpenAI API key (`OPENAI_API_KEY`)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

### 4. Add your documents

```bash
mkdir docs
cp /path/to/your/documents/*.pdf docs/
```

### 5. Run ingestion (one-time)

```bash
python ingest.py --docs_dir ./docs --chroma_dir ./chroma_db
```

Expected output:
```
Processing: company_handbook.pdf
  → Extracted text from 42 pages
  → Extracted 8 images
  → Describing image on page 5 (org chart, 120KB)
  → Describing image on page 17 (revenue chart, 89KB)
  ✔ Upserted 187 records into ChromaDB

Ingestion complete. Total docs in collection: 187
```

### 6. Launch the UI

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

---

## File Structure

```
multimodal_rag/
├── ingest.py          # PDF parsing, image captioning, ChromaDB ingestion
├── query.py           # LangGraph agent, retrieval, LLM generation
├── app.py             # Gradio web UI
├── requirements.txt   # Python dependencies
├── docs/              # Put your PDF files here (created by you)
└── chroma_db/         # Auto-created by ingest.py — persistent vector store
```

---

## Key Design Decisions (Learning Notes)

### Why PyMuPDF over LangChain's PDF loaders?
LangChain's `PyPDFLoader` extracts text only. We need byte-level access to embedded images, which requires `fitz.Document.extract_image(xref)`. Using PyMuPDF directly gives us full control and keeps the ingestion pipeline transparent.

### Why convert images to text descriptions?
Our embedding model (`all-MiniLM-L6-v2`) is text-only. By asking GPT-4o-mini Vision to describe each image in structured detail, we project visual content into the same semantic vector space as text — enabling unified retrieval without a separate multimodal embedding model.

### Why LangGraph over a plain function chain?
LangGraph provides:
- **Explicit typed state** — every node receives and returns the same `AgentState` TypedDict
- **Easy branching** — add conditional edges (e.g. "retry if no results found") without restructuring code
- **Built-in streaming** — swap `.invoke()` for `.stream()` for streaming responses
- **Observability** — each node's inputs/outputs can be logged, traced, or cached independently

### Why cosine similarity?
Sentence-transformer embeddings are unit-normalised (L2 norm = 1). In this case, cosine similarity = dot product. ChromaDB's `hnsw:space=cosine` uses this for fast approximate nearest-neighbour search (HNSW = Hierarchical Navigable Small World graph).

### Why the 2-step Gradio pattern?
```python
send_btn.click(fn=_submit_message, ...).then(fn=_generate_response, ...)
```
Step 1 runs instantly (no I/O), making the user's message appear immediately. Step 2 runs the 3-5 second LLM call. This prevents the "frozen UI" experience where the user waits with no feedback.

---

## Extending the System

### Add keyword (BM25) retrieval for hybrid search
```python
from rank_bm25 import BM25Okapi
# Combine BM25 ranks + vector ranks with Reciprocal Rank Fusion (RRF)
```

### Add streaming responses
```python
# In generation_node, replace client.chat.completions.create with:
stream = client.chat.completions.create(..., stream=True)
for chunk in stream:
    yield chunk.choices[0].delta.content
```

### Add a hallucination guard node
```python
def hallucination_check_node(state):
    # Ask LLM: "Does the answer contradict any of the source chunks?"
    # If yes → regenerate with stricter grounding instruction
```

### Scale to cloud
- Replace `chromadb.PersistentClient` with `chromadb.HttpClient` pointing at a Chroma server
- Replace local HuggingFace model with OpenAI `text-embedding-3-small` for managed infrastructure
- Deploy Gradio app to Hugging Face Spaces or any Docker-compatible host

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for GPT-4o-mini |
| `CHROMA_DIR` | `./chroma_db` | Path to ChromaDB storage |
| `APP_PORT` | `7860` | Gradio server port |

---

## Cost Estimate

| Operation | Model | Approx. Cost |
|---|---|---|
| Image captioning (per image) | gpt-4o-mini vision | ~$0.001–0.003 |
| Query analysis (per query) | gpt-4o-mini | ~$0.0001 |
| Answer generation (per query) | gpt-4o-mini | ~$0.001–0.003 |
| Text embeddings | all-MiniLM-L6-v2 (local) | **Free** |

A 100-page document with 20 images costs roughly **$0.05–0.10** to ingest. Each user query costs roughly **$0.002–0.005**.
