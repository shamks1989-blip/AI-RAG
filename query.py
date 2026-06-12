"""
=============================================================================
query.py — Hybrid Agentic Retrieval + Contextual Generation Engine
=============================================================================
PURPOSE:
    This module is the "online" (real-time) phase of the RAG system. It:
      1. Analyses the user query with a lightweight LangGraph agent to
         decide the best retrieval strategy and rewrite/expand the query
         when needed.
      2. Performs semantic similarity retrieval from ChromaDB to fetch
         the top-K most relevant text chunks AND image descriptions.
      3. Optionally accepts an external image uploaded by the user via
         Gradio — this image is also described by Vision and mixed into
         the retrieval context.
      4. Constructs a rich, structured system prompt combining all context.
      5. Calls GPT-4o-mini to generate a grounded, cited answer.

ARCHITECTURE — LangGraph Agent Flow:
    ┌──────────────┐     ┌───────────────────┐     ┌──────────────────┐
    │  query_node  │────▶│  retrieval_node   │────▶│  generation_node │
    │  (analyser)  │     │  (ChromaDB fetch) │     │  (GPT-4o-mini)   │
    └──────────────┘     └───────────────────┘     └──────────────────┘
    Each node receives and mutates a shared AgentState TypedDict.
    LangGraph handles the state passing — no global variables needed.

USAGE (standalone):
    from query import run_query
    answer = run_query("What are our Q3 revenue targets?")
    print(answer["answer"])

DEPENDENCIES:
    openai, chromadb, sentence-transformers, langgraph, langchain-core
=============================================================================
"""

import os
import base64
import logging
from typing import Optional, TypedDict, Annotated

# ── LangGraph — lightweight stateful agent framework ─────────────────────────
from langgraph.graph import StateGraph, END

# ── Embeddings & vector store ─────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ── LLM ───────────────────────────────────────────────────────────────────────
from openai import OpenAI

log = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

COLLECTION_NAME  = "knowledge_base"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL        = "openai/gpt-4o-mini"
TOP_K            = 10         # number of chunks to retrieve per query
SCORE_THRESHOLD  = 1.6        # cosine *distance* threshold (lower = more similar)
                               # ChromaDB returns distances, not similarities.
                               # For cosine space: distance = 1 - similarity.
                               # distance < 0.3  → very relevant
                               # distance < 0.5  → relevant
                               # distance > 1.4  → likely noise, filter out

MAX_CONTEXT_CHARS = 6000      # Trim total context pasted into system prompt


# =============================================================================
# MODULE-LEVEL SINGLETONS (lazy-initialised once per process)
# =============================================================================
# These are initialised on first call to run_query() so that importing
# this module (e.g. in app.py) doesn't trigger slow model loading.

_embed_model: Optional[SentenceTransformer] = None
_chroma_collection = None
_openai_client: Optional[OpenAI] = None


def _get_resources(chroma_dir: str = "./chroma_db"):
    """
    Lazy-initialise and cache the embedding model, ChromaDB collection,
    and OpenAI client. Safe to call multiple times — returns cached objects.
    """
    global _embed_model, _chroma_collection, _openai_client

    if _embed_model is None:
        log.info("Loading embedding model (first call only)…")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    if _chroma_collection is None:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            f"Connected to ChromaDB collection '{COLLECTION_NAME}' "
            f"({_chroma_collection.count()} docs)"
        )

    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        _openai_client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    return _embed_model, _chroma_collection, _openai_client


def get_indexed_documents(chroma_dir: str = "./chroma_db") -> list[dict]:
    """
    Return sorted unique documents indexed in ChromaDB.
    Each entry: {"source": filename, "title": human-readable title}
    """
    try:
        _, collection, _ = _get_resources(chroma_dir)
        result = collection.get(include=["metadatas"])
        seen: dict[str, str] = {}   # source → title
        for meta in result.get("metadatas") or []:
            src = meta.get("source", "")
            if src and src not in seen:
                title = meta.get("title", "").strip()
                if not title:
                    # Fallback for docs ingested before title was added
                    stem = src.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
                    title = " ".join(w.capitalize() for w in stem.split())
                seen[src] = title
        return sorted(seen.items(), key=lambda x: x[1])   # sort by title
    except Exception as e:
        log.warning(f"get_indexed_documents failed: {e}")
        return []


# =============================================================================
# LANGGRAPH AGENT STATE
# =============================================================================

class AgentState(TypedDict):
    """
    Shared mutable state passed through every node in the LangGraph graph.

    Using a TypedDict makes the state fully introspectable and type-safe.
    LangGraph automatically routes it from node to node.
    """
    # ── Inputs ────────────────────────────────────────────────────────────────
    user_query: str                   # raw user question
    chat_history: list[dict]          # prior turns [{role, content}, …]
    uploaded_image_bytes: Optional[bytes]  # optional image from UI upload
    chroma_dir: str                   # path to ChromaDB storage

    # ── Intermediate state (populated by agent nodes) ─────────────────────────
    rewritten_query: str              # query after expansion/rewriting
    query_intent: str                 # "factual" | "visual" | "comparison" | "summary"
    retrieved_chunks: list[dict]      # [{text, source, page, score, type}, …]
    uploaded_image_description: str  # Vision description of user-uploaded image

    # ── Final output ──────────────────────────────────────────────────────────
    answer: str                       # final LLM response
    sources: list[dict]               # [{source, page, type}, …] for citations


# =============================================================================
# NODE 1 — QUERY ANALYSER
# =============================================================================

def query_analyser_node(state: AgentState) -> AgentState:
    """
    WHAT IT DOES:
        - Classifies the user's intent (factual lookup, visual query, etc.)
        - Rewrites/expands the query to maximise retrieval recall.
        - Incorporates recent chat history for follow-up questions
          ("What about the Q4 numbers?" needs prior context to make sense).

    WHY REWRITE?
        User questions are often terse or ambiguous. Expanding them with
        synonyms and related terms dramatically improves vector search recall
        without hurting precision (the LLM still filters in generation).

    This uses a cheap, fast LLM call (the query itself is tiny).
    """
    _, _, client = _get_resources(state["chroma_dir"])

    # Build a compact context window from recent history (last 4 turns)
    recent_history = state.get("chat_history", [])[-4:]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent_history
    ) or "No prior conversation."

    system = (
        "You are a query analysis assistant. Given a user question and recent "
        "conversation history, respond with a JSON object containing:\n"
        '  "intent": one of ["factual", "visual", "comparison", "summary", "procedural"],\n'
        '  "rewritten_query": an expanded, self-contained version of the question '
        "enriched with synonyms, related terms, and enough context from history "
        "so it can be understood without the history.\n\n"
        "Rules:\n"
        "- If the question is a follow-up, resolve the pronoun/reference.\n"
        "- Add domain-relevant synonyms (e.g. revenue → earnings, sales, income).\n"
        "- Keep the rewritten query under 60 words.\n"
        "- Return ONLY the JSON object, no markdown fences."
    )

    user_msg = (
        f"RECENT HISTORY:\n{history_text}\n\n"
        f"CURRENT QUESTION: {state['user_query']}"
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=200,
            temperature=0.0,  # deterministic for analysis tasks
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        import json
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        state["query_intent"]   = parsed.get("intent", "factual")
        state["rewritten_query"] = parsed.get("rewritten_query", state["user_query"])
        log.info(f"Query intent: {state['query_intent']}")
        log.info(f"Rewritten:    {state['rewritten_query']}")
    except Exception as e:
        # Graceful fallback — never block the user due to analysis failure
        log.warning(f"Query analyser failed ({e}), using raw query.")
        state["query_intent"]    = "factual"
        state["rewritten_query"] = state["user_query"]

    return state


# =============================================================================
# NODE 2 — RETRIEVAL
# =============================================================================

def retrieval_node(state: AgentState) -> AgentState:
    """
    WHAT IT DOES:
        1. Embeds the rewritten query with the same model used during ingestion
           (all-MiniLM-L6-v2) — MUST be the same model for the vector space
           to be compatible.
        2. Queries ChromaDB for top-K nearest neighbours by cosine distance.
        3. Filters results below the quality threshold.
        4. If the user uploaded an image, describes it with Vision and prepends
           that description to the context.

    DISTANCE vs SIMILARITY:
        ChromaDB's cosine space returns *distances* (0 = identical, 2 = opposite).
        We convert to a pseudo-similarity score for display: score = 1 - distance/2.

    HYBRID RETRIEVAL NOTE:
        A fully hybrid system would combine dense (vector) + sparse (BM25/keyword)
        retrieval and merge results with Reciprocal Rank Fusion (RRF). For clarity
        we implement dense-only retrieval here, which covers 90%+ of real use-cases.
        You can extend this by adding a BM25Retriever and merging the result lists.
    """
    embed_model, collection, client = _get_resources(state["chroma_dir"])

    # ── 2a. Embed the query ────────────────────────────────────────────────────
    query_vec = embed_model.encode(state["rewritten_query"]).tolist()

    # ── 2b. ChromaDB nearest-neighbour search ─────────────────────────────────
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(TOP_K, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs       = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    for doc, meta, dist in zip(docs, metadatas, distances):
        # Filter out low-quality matches
        if dist > SCORE_THRESHOLD:
            continue
        # Convert distance to a 0→1 confidence score for display
        similarity_score = round(1.0 - dist / 2.0, 3)
        chunks.append({
            "text":   doc,
            "source": meta.get("source", "unknown"),
            "page":   meta.get("page", "?"),
            "type":   meta.get("chunk_type", "text"),
            "score":  similarity_score,
        })

    log.info(f"Retrieved {len(chunks)} chunks above quality threshold.")
    state["retrieved_chunks"] = chunks

    # ── 2c. Handle user-uploaded image (optional) ─────────────────────────────
    state["uploaded_image_description"] = ""

    if state.get("uploaded_image_bytes"):
        log.info("Describing user-uploaded image via Vision…")
        try:
            img_bytes = state["uploaded_image_bytes"]
            b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
            # Try to detect format; default to png
            ext = "png"
            if img_bytes[:3] == b"\xff\xd8\xff":
                ext = "jpeg"
            elif img_bytes[:4] == b"\x89PNG":
                ext = "png"

            vision_resp = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{ext};base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image in detail, focusing on any "
                                    "data, text, charts, diagrams, or business-relevant "
                                    "content visible. The user will ask a question about it."
                                ),
                            },
                        ],
                    }
                ],
            )
            state["uploaded_image_description"] = (
                vision_resp.choices[0].message.content.strip()
            )
            log.info("User image description generated.")
        except Exception as e:
            log.warning(f"Could not describe uploaded image: {e}")
            state["uploaded_image_description"] = "[Could not process uploaded image]"

    return state


# =============================================================================
# NODE 3 — CONTEXTUAL GENERATION
# =============================================================================

def generation_node(state: AgentState) -> AgentState:
    """
    WHAT IT DOES:
        Assembles a structured system prompt from all retrieved context and
        calls GPT-4o-mini to generate the final answer.

    PROMPT ENGINEERING STRATEGY:
        We use a "grounding prompt" pattern:
          1. Role assignment — tells the model WHO it is.
          2. Strict grounding instruction — "only use the provided context".
          3. Structured context blocks — clearly labelled and separated.
          4. Output format instruction — ensures citations and confidence.
          5. User question — always last.

    CONFIDENCE SCORE:
        We ask the model to self-assess confidence. This is a heuristic
        (models are not perfectly calibrated) but gives users a useful signal.
    """
    _, _, client = _get_resources(state["chroma_dir"])

    chunks        = state.get("retrieved_chunks", [])
    img_desc      = state.get("uploaded_image_description", "")
    query_intent  = state.get("query_intent", "factual")

    # ── 3a. Short-circuit if nothing was retrieved ────────────────────────────
    if not chunks and not img_desc:
        try:
            import chromadb
            col = chromadb.PersistentClient(path=state["chroma_dir"]).get_or_create_collection(COLLECTION_NAME)
            all_meta = col.get(include=["metadatas"])["metadatas"] or []
            docs = sorted({m.get("source", "") for m in all_meta if m.get("source")})
            doc_list = "\n".join(f"  • {d}" for d in docs) if docs else "  (none indexed yet)"
        except Exception:
            doc_list = "  (could not read document list)"
        state["answer"] = (
            "I couldn't find relevant information in the knowledge base for that question.\n\n"
            f"**Documents currently indexed:**\n{doc_list}\n\n"
            "Try rephrasing your question or ask about a topic covered by one of the documents above."
        )
        state["sources"] = []
        return state

    # ── 3b. Build the context section ─────────────────────────────────────────
    context_parts = []
    sources_seen  = set()
    final_sources = []

    for i, chunk in enumerate(chunks):
        # Truncate very long chunks to respect context window limits
        text_preview = chunk["text"][:1200]
        label = "📄 TEXT" if chunk["type"] == "text" else "🖼 IMAGE DESCRIPTION"
        context_parts.append(
            f"[{i+1}] {label} | Source: {chunk['source']} | "
            f"Page: {chunk['page']} | Relevance: {chunk['score']:.0%}\n"
            f"{text_preview}"
        )
        # Deduplicate source citations
        src_key = f"{chunk['source']}|{chunk['page']}"
        if src_key not in sources_seen:
            sources_seen.add(src_key)
            final_sources.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "type": chunk["type"],
                "score": chunk["score"],
            })

    # Optionally inject user-uploaded image context
    if img_desc:
        context_parts.insert(
            0,
            f"[0] 📷 USER-UPLOADED IMAGE DESCRIPTION\n{img_desc}",
        )

    # Assemble and trim total context to avoid exceeding token budget
    full_context = "\n\n---\n\n".join(context_parts)
    if len(full_context) > MAX_CONTEXT_CHARS:
        full_context = full_context[:MAX_CONTEXT_CHARS] + "\n\n[…context trimmed…]"

    # ── 3c. Select tone based on intent ───────────────────────────────────────
    tone_instructions = {
        "factual":    "Give a precise, concise, factual answer.",
        "visual":     "Describe the visual content clearly and explain what it means.",
        "comparison": "Structure your answer as a clear comparison with pros/cons or a table if helpful.",
        "summary":    "Provide a structured executive summary with key bullets.",
        "procedural": "Provide a numbered step-by-step guide.",
    }.get(query_intent, "Give a precise, concise answer.")

    # ── 3d. System prompt ──────────────────────────────────────────────────────
    system_prompt = f"""You are an expert internal knowledge assistant for a startup company.
Your role is to provide accurate, grounded answers strictly based on the provided context.

STRICT RULES:
1. Base your answer primarily on the context blocks below. You may use general knowledge only to clarify terminology or fill minor gaps — always make clear what came from the documents vs general knowledge.
2. Answer as fully as you can from the available context. If the context only partially covers the question, answer what you can and briefly note what is missing.
3. Always cite your sources using the format [Source: filename, Page X].
4. After your answer, provide a CONFIDENCE SCORE from 0–100% based on how well the context supports your answer.

TONE INSTRUCTION: {tone_instructions}

RESPONSE FORMAT:
---
[Your answer here, with inline citations like: [Source: file.pdf, Page 3]]

**Sources Used:**
[List each source: filename, page, content type]

**Confidence Score:** [X]%
**Reasoning:** [One sentence explaining why you gave this confidence score]
---

CONTEXT:
{full_context}
"""

    # ── 3e. Build message list (include chat history for follow-ups) ───────────
    messages = [{"role": "system", "content": system_prompt}]

    # Include last 6 turns for conversational continuity
    for turn in state.get("chat_history", [])[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Current user question
    messages.append({"role": "user", "content": state["user_query"]})

    # ── 3f. LLM call ──────────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1200,
            temperature=0.2,   # low temperature for factual grounding
            messages=messages,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"LLM generation failed: {e}")
        answer = f"⚠️ Generation error: {e}"

    state["answer"]  = answer
    state["sources"] = final_sources
    return state


# =============================================================================
# LANGGRAPH GRAPH ASSEMBLY
# =============================================================================

def build_agent_graph() -> StateGraph:
    """
    Assemble the three nodes into a linear LangGraph pipeline.

    Graph structure:
        START → query_analyser → retrieval → generation → END

    LangGraph is used here instead of a plain function chain because:
      - State is explicitly typed and immutable between nodes
      - Easy to add conditional branching later (e.g. "if no results, ask
        for clarification" → loop back to query_analyser)
      - Built-in streaming and async support
      - Clear visualisation of the agent flow

    EXTENDING THIS GRAPH (examples):
      - Add a "hallucination checker" node after generation
      - Add conditional edge: if len(chunks) == 0 → "clarification_node"
      - Add a "tool_use" node that queries external APIs for live data
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("query_analyser", query_analyser_node)
    graph.add_node("retrieval",      retrieval_node)
    graph.add_node("generation",     generation_node)

    # Connect edges (linear flow)
    graph.set_entry_point("query_analyser")
    graph.add_edge("query_analyser", "retrieval")
    graph.add_edge("retrieval",      "generation")
    graph.add_edge("generation",     END)

    return graph.compile()


# Cache the compiled graph (expensive to build; cheap to reuse)
_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
        log.info("LangGraph agent compiled and ready.")
    return _agent_graph


# =============================================================================
# PUBLIC API — called by app.py
# =============================================================================

def run_query(
    user_query: str,
    chat_history: Optional[list[dict]] = None,
    uploaded_image_bytes: Optional[bytes] = None,
    chroma_dir: str = "./chroma_db",
) -> dict:
    """
    Main entry point. Runs the full agent pipeline and returns a dict:
      {
        "answer":  str,          — LLM-generated answer with citations
        "sources": list[dict],   — [{source, page, type, score}, …]
        "intent":  str,          — detected query intent
        "query":   str,          — rewritten query
      }

    Args:
        user_query:            Raw question from the user.
        chat_history:          Prior conversation turns for context.
        uploaded_image_bytes:  Raw bytes of an image uploaded via the UI.
        chroma_dir:            Path to ChromaDB storage directory.
    """
    # Pre-load resources to avoid first-query latency on subsequent calls
    _get_resources(chroma_dir)

    initial_state: AgentState = {
        "user_query":                user_query,
        "chat_history":              chat_history or [],
        "uploaded_image_bytes":      uploaded_image_bytes,
        "chroma_dir":                chroma_dir,
        # These will be populated by the nodes:
        "rewritten_query":           "",
        "query_intent":              "",
        "retrieved_chunks":          [],
        "uploaded_image_description": "",
        "answer":                    "",
        "sources":                   [],
    }

    graph  = get_agent_graph()
    result = graph.invoke(initial_state)

    return {
        "answer":  result["answer"],
        "sources": result["sources"],
        "intent":  result["query_intent"],
        "query":   result["rewritten_query"],
    }


# =============================================================================
# CLI ENTRYPOINT (for testing without the UI)
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Summarise the key documents in the knowledge base."
    print(f"\n🔍 Query: {question}\n")

    result = run_query(question)
    print("=" * 70)
    print(result["answer"])
    print("=" * 70)
    print(f"Intent: {result['intent']} | Rewritten: {result['query']}")
    print(f"Sources: {result['sources']}")
