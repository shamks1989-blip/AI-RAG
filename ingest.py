"""
=============================================================================
ingest.py — Multimodal RAG Ingestion Pipeline
=============================================================================
PURPOSE:
    This module is the "offline" phase of the RAG system. You run it once
    (or whenever your knowledge-base documents change). It:
      1. Walks a directory of PDFs.
      2. Extracts plain text and splits it into overlapping chunks.
      3. Extracts every embedded image from each PDF page.
      4. Calls GPT-4o-mini Vision to generate a rich textual description of
         each image (so images become semantically searchable).
      5. Embeds all text chunks + image descriptions with a HuggingFace model.
      6. Persists everything into a local ChromaDB collection with rich
         metadata (source file, page, chunk type, etc.).

ARCHITECTURE NOTE:
    We deliberately avoid LangChain's high-level document loaders here so
    you can see exactly what is happening at each step. PyMuPDF (fitz) gives
    us byte-level control over PDF content.

USAGE:
    python ingest.py --docs_dir ./docs --chroma_dir ./chroma_db

DEPENDENCIES (install via requirements.txt):
    pymupdf, openai, chromadb, sentence-transformers, langchain-text-splitters
=============================================================================
"""

import os
import io
import base64
import argparse
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ── PDF parsing ──────────────────────────────────────────────────────────────
import fitz  # PyMuPDF — gives us per-page text + raw image bytes

# ── Text splitting (only component we borrow from LangChain) ─────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Embeddings ────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── Vector store ──────────────────────────────────────────────────────────────
import chromadb
from chromadb.config import Settings

# ── Vision LLM ────────────────────────────────────────────────────────────────
from openai import OpenAI

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS  — tweak these to balance quality vs. cost/speed
# =============================================================================

COLLECTION_NAME    = "knowledge_base"   # ChromaDB collection name
EMBED_MODEL_NAME   = "all-MiniLM-L6-v2" # Fast, 384-dim sentence embeddings
VISION_MODEL       = "openai/gpt-4o-mini" # model for image captioning (via OpenRouter)
CHUNK_SIZE         = 800                 # Characters per text chunk
CHUNK_OVERLAP      = 150                 # Overlap to preserve sentence context
MIN_IMAGE_BYTES    = 5_000              # Skip tiny icons / decorative images
MAX_IMAGE_CAPTION_TOKENS = 400          # Max tokens for image summary


# =============================================================================
# SECTION 1 — PDF TEXT EXTRACTION
# =============================================================================

_PLACEHOLDER_TITLES = {
    "(anonymous)", "untitled", "document", "pdf", "microsoft word",
    "word document", "unknown", "no title", "none", "",
}

def extract_pdf_title(pdf_path: str) -> str:
    """
    Best-effort title extraction for a PDF, tried in order:
      1. PDF document metadata 'title' field (skips common placeholders)
      2. Largest-font text on page 1 (likely the heading)
      3. First non-trivial plain-text line on page 1
      4. Cleaned-up filename as last resort
    """
    try:
        doc = fitz.open(pdf_path)

        # 1. PDF metadata title — skip placeholder values
        title = (doc.metadata or {}).get("title", "").strip()
        if title and title.lower() not in _PLACEHOLDER_TITLES and 4 < len(title) < 150:
            doc.close()
            return title

        if len(doc) > 0:
            page = doc[0]

            # 2. Largest-font span on page 1 (strongest heading signal)
            try:
                blocks = page.get_text("dict")["blocks"]
                best_text, best_size = "", 0.0
                for block in blocks:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            size = span.get("size", 0)
                            if size > best_size and 4 < len(text) < 120:
                                best_size, best_text = size, text
                if best_text:
                    doc.close()
                    return best_text
            except Exception:
                pass

            # 3. First meaningful plain-text line
            for line in page.get_text("text").splitlines():
                line = line.strip()
                if 4 < len(line) < 120:
                    doc.close()
                    return line

        doc.close()
    except Exception:
        pass

    # 4. Filename fallback
    stem = Path(pdf_path).stem.replace("_", " ").replace("-", " ")
    return " ".join(w.capitalize() for w in stem.split())


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Open a PDF and return a list of page records, each containing:
      { "page": int, "text": str }

    We iterate pages manually so we can attach page numbers to every chunk
    later (crucial for source citations in the final answer).
    """
    pages = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")          # plain UTF-8 text
        text = text.strip()
        if text:                              # skip blank/scanned-only pages
            pages.append({"page": page_num + 1, "text": text})

    doc.close()
    log.info(f"  → Extracted text from {len(pages)} pages in {Path(pdf_path).name}")
    return pages


# =============================================================================
# SECTION 2 — IMAGE EXTRACTION FROM PDF
# =============================================================================

def extract_images_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract all raster images embedded in a PDF.
    Returns a list of records:
      { "page": int, "image_index": int, "image_bytes": bytes, "ext": str }

    PyMuPDF's get_images() returns a list of (xref, smask, width, height,
    bpc, colorspace, alt. colorspace, name, filter, referencer) tuples.
    We use the xref to extract the raw pixel data.
    """
    images = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        found_on_page = 0

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]               # cross-reference number in PDF
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]      # "png", "jpeg", etc.

                # Skip tiny images (icons, bullets, decorative elements)
                if len(image_bytes) < MIN_IMAGE_BYTES:
                    continue

                images.append({
                    "page": page_num + 1,
                    "image_index": img_idx,
                    "image_bytes": image_bytes,
                    "ext": ext,
                })
                found_on_page += 1
            except Exception as e:
                log.warning(f"  ⚠ Could not extract image xref={xref}: {e}")

        # Fallback: if get_images() found nothing, render the whole page as a
        # pixmap. This catches images inside Form XObjects (common in PDFs
        # exported from Canva, PowerPoint, and design tools) as well as vector
        # graphics that have no embedded raster XObject.
        if found_on_page == 0:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                image_bytes = pix.tobytes("png")

                # Skip if the rendered page is almost entirely white/blank
                # (indicates a text-only page with no meaningful visual content)
                if len(image_bytes) >= MIN_IMAGE_BYTES and _page_has_visual_content(pix):
                    images.append({
                        "page": page_num + 1,
                        "image_index": 0,
                        "image_bytes": image_bytes,
                        "ext": "png",
                    })
                    log.info(f"  → Page {page_num + 1}: no XObject images found, using full-page render")
            except Exception as e:
                log.warning(f"  ⚠ Could not render page {page_num + 1}: {e}")

    doc.close()
    log.info(f"  → Extracted {len(images)} images from {Path(pdf_path).name}")
    return images


def _page_has_visual_content(pix) -> bool:
    """
    Return True if the rendered page pixmap contains meaningful visual content
    (i.e. is not a near-blank white page).  Samples pixel brightness variance —
    a page that is all white has zero variance; photos/charts have high variance.
    """
    try:
        import struct
        samples = pix.samples          # raw bytes: R,G,B per pixel
        step = max(1, len(samples) // (3 * 2000))   # sample ~2000 pixels
        brightness = [
            (samples[i] + samples[i + 1] + samples[i + 2]) / 3
            for i in range(0, len(samples) - 2, step * 3)
        ]
        avg = sum(brightness) / len(brightness)
        variance = sum((b - avg) ** 2 for b in brightness) / len(brightness)
        # Pages with variance > 100 have meaningful colour/contrast variation
        return variance > 100
    except Exception:
        return True   # if we can't check, include it anyway


# =============================================================================
# SECTION 3 — VISION LLM: IMAGE → TEXT DESCRIPTION
# =============================================================================

def describe_image_with_vision(
    client: OpenAI,
    image_bytes: bytes,
    ext: str,
    source_filename: str,
    page_num: int,
) -> str:
    """
    Send an image to GPT-4o-mini and ask for a structured description.

    Why do this?
        Our embedding model (all-MiniLM-L6-v2) is text-only. By converting
        images into rich textual descriptions, we make them semantically
        retrievable alongside plain text chunks — a unified latent space.

    The prompt is carefully crafted to elicit structured output that captures:
      - What type of visual this is (chart, form, diagram, photo, table…)
      - The key data / information it conveys
      - Any visible text, labels, axes, legends
      - Business relevance cues
    """
    # Encode image to base64 (required by OpenAI Vision API)
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
    mime_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    system_prompt = (
        "You are an expert document analyst specialising in business intelligence. "
        "Your role is to analyse images extracted from corporate documents and produce "
        "detailed, factual descriptions that will be stored in a knowledge base and "
        "searched by employees. Be specific, thorough, and structured."
    )

    user_prompt = (
        f"This image was extracted from page {page_num} of '{source_filename}'. "
        "Please analyse it and provide:\n"
        "1. IMAGE TYPE: (e.g., bar chart, org chart, scanned form, floor plan, screenshot, table, diagram)\n"
        "2. MAIN CONTENT: What information does it convey? Include specific numbers, labels, and key data points.\n"
        "3. VISIBLE TEXT: Transcribe any text, axis labels, legend entries, or annotations verbatim.\n"
        "4. BUSINESS CONTEXT: What decision or question might this image help answer?\n"
        "5. SUMMARY: A single paragraph synthesis for search purposes.\n\n"
        "Be precise and thorough — this description is the only way a text-based search "
        "will ever find this image's contents."
    )

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=MAX_IMAGE_CAPTION_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}",
                                "detail": "high",  # high fidelity for charts/forms
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        )
        description = response.choices[0].message.content.strip()
        return description

    except Exception as e:
        log.error(f"  ✗ Vision API call failed: {e}")
        return f"[Image description unavailable: {e}]"


# =============================================================================
# SECTION 4 — TEXT CHUNKING
# =============================================================================

def chunk_page_text(
    page_text: str,
    source: str,
    page_num: int,
    splitter: RecursiveCharacterTextSplitter,
    title: str = "",
) -> list[dict]:
    """
    Split a single page's text into overlapping chunks and attach metadata.
    """
    chunks = splitter.split_text(page_text)
    records = []
    for idx, chunk in enumerate(chunks):
        records.append({
            "text": chunk,
            "metadata": {
                "source": source,
                "title": title,
                "page": page_num,
                "chunk_type": "text",
                "chunk_index": idx,
            },
        })
    return records


# =============================================================================
# SECTION 5 — CHROMADB PERSISTENCE
# =============================================================================

def delete_document(collection, source_name: str) -> int:
    """
    Remove every ChromaDB record whose 'source' metadata matches source_name.
    Returns the number of records deleted.
    Called before re-ingesting a PDF so stale/orphaned chunks don't linger.
    """
    result = collection.get(where={"source": source_name}, include=[])
    ids_to_delete = result.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        log.info(f"  🗑  Deleted {len(ids_to_delete)} existing records for '{source_name}'")
    return len(ids_to_delete)


def get_or_create_collection(chroma_dir: str) -> tuple:
    """
    Initialise a persistent ChromaDB client and return (client, collection).

    ChromaDB stores its data in 'chroma_dir' as SQLite + binary files,
    so the index survives between runs. On subsequent calls, existing
    embeddings are NOT re-computed (we check by document ID).

    We use cosine similarity because sentence-transformer embeddings are
    unit-normalised — cosine is equivalent to dot-product and works well
    for semantic similarity tasks.
    """
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity index
    )
    log.info(
        f"ChromaDB collection '{COLLECTION_NAME}' ready "
        f"({collection.count()} existing docs) at '{chroma_dir}'"
    )
    return client, collection


def add_records_to_chroma(
    collection,
    records: list[dict],
    embed_model: SentenceTransformer,
    batch_size: int = 64,
):
    """
    Embed a list of text records and upsert them into ChromaDB.

    We use 'upsert' (not 'add') so re-running ingest.py on the same files
    is idempotent — existing embeddings are updated, not duplicated.

    Document IDs are deterministic hashes of (source + page + chunk_index +
    chunk_type) so the same content always maps to the same ID.
    """
    import hashlib

    texts     = [r["text"] for r in records]
    metadatas = [r["metadata"] for r in records]

    # Build stable, collision-resistant IDs
    ids = [
        hashlib.md5(
            f"{m['source']}|{m['page']}|{m.get('chunk_index', 0)}|{m['chunk_type']}"
            .encode()
        ).hexdigest()
        for m in metadatas
    ]

    # Embed in batches to avoid OOM on large corpora
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embed_model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(vecs)

    collection.upsert(
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    log.info(f"  ✔ Upserted {len(records)} records into ChromaDB")


# =============================================================================
# SECTION 6 — MAIN ORCHESTRATOR
# =============================================================================

def ingest_directory(docs_dir: str, chroma_dir: str):
    """
    Top-level pipeline:
      For each PDF in docs_dir →
        extract text chunks +
        extract & describe images →
        embed & upsert into ChromaDB
    """
    # ── Sanity checks ────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before running: export OPENAI_API_KEY=sk-..."
        )

    pdf_files = list(Path(docs_dir).glob("**/*.pdf"))
    if not pdf_files:
        log.warning(f"No PDF files found in '{docs_dir}'. Exiting.")
        return

    log.info(f"Found {len(pdf_files)} PDF(s) to process.")

    # ── Initialise models & stores ───────────────────────────────────────────
    base_url = os.environ.get("OPENAI_BASE_URL")
    openai_client  = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    embed_model    = SentenceTransformer(EMBED_MODEL_NAME)
    _, collection  = get_or_create_collection(chroma_dir)

    # One splitter instance is reused for all documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
    )

    # ── Process each PDF ─────────────────────────────────────────────────────
    for pdf_path in pdf_files:
        source_name = pdf_path.name
        log.info(f"\n{'─'*60}")
        log.info(f"Processing: {source_name}")
        log.info(f"{'─'*60}")

        all_records: list[dict] = []
        title = extract_pdf_title(str(pdf_path))
        log.info(f"  → Title: {title}")

        # ── 6a. Text extraction & chunking ───────────────────────────────────
        pages = extract_text_from_pdf(str(pdf_path))
        for page_info in pages:
            chunks = chunk_page_text(
                page_text=page_info["text"],
                source=source_name,
                page_num=page_info["page"],
                splitter=splitter,
                title=title,
            )
            all_records.extend(chunks)

        log.info(f"  → {len(all_records)} text chunks produced")

        # ── 6b. Image extraction & Vision captioning ─────────────────────────
        images = extract_images_from_pdf(str(pdf_path))
        for img_info in images:
            log.info(
                f"  → Describing image on page {img_info['page']} "
                f"(index {img_info['image_index']}, {len(img_info['image_bytes'])//1024}KB)"
            )
            description = describe_image_with_vision(
                client=openai_client,
                image_bytes=img_info["image_bytes"],
                ext=img_info["ext"],
                source_filename=source_name,
                page_num=img_info["page"],
            )

            prefixed_desc = (
                f"[IMAGE DESCRIPTION — {source_name}, Page {img_info['page']}]\n"
                f"{description}"
            )

            all_records.append({
                "text": prefixed_desc,
                "metadata": {
                    "source": source_name,
                    "title": title,
                    "page": img_info["page"],
                    "chunk_type": "image_description",
                    "chunk_index": img_info["image_index"],
                },
            })

        # ── 6c. Purge old records then embed & persist ────────────────────────
        delete_document(collection, source_name)
        if all_records:
            add_records_to_chroma(collection, all_records, embed_model)
        else:
            log.warning(f"  ⚠ No extractable content found in {source_name}")

    log.info(f"\n{'='*60}")
    log.info(
        f"Ingestion complete. "
        f"Total docs in collection: {collection.count()}"
    )
    log.info(f"{'='*60}")


# =============================================================================
# SINGLE-FILE PUBLIC API  (called from app.py UI upload handler)
# =============================================================================

def ingest_single_file(pdf_path: str, chroma_dir: str) -> dict:
    """
    Ingest one PDF into ChromaDB.  Existing records for the same filename
    are deleted first, so this safely handles both new uploads and updates.

    Returns a dict: {"source": str, "chunks": int, "deleted": int, "error": str|None}
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"source": "", "chunks": 0, "deleted": 0,
                "error": "OPENAI_API_KEY is not set."}

    pdf_path_obj = Path(pdf_path)
    source_name  = pdf_path_obj.name

    try:
        base_url      = os.environ.get("OPENAI_BASE_URL")
        openai_client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        embed_model   = SentenceTransformer(EMBED_MODEL_NAME)
        _, collection = get_or_create_collection(chroma_dir)
        splitter      = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_records: list[dict] = []
        title = extract_pdf_title(str(pdf_path_obj))

        # Text extraction
        for page_info in extract_text_from_pdf(str(pdf_path_obj)):
            all_records.extend(
                chunk_page_text(page_info["text"], source_name, page_info["page"], splitter, title)
            )

        # Image extraction + Vision captioning
        for img_info in extract_images_from_pdf(str(pdf_path_obj)):
            description = describe_image_with_vision(
                client=openai_client,
                image_bytes=img_info["image_bytes"],
                ext=img_info["ext"],
                source_filename=source_name,
                page_num=img_info["page"],
            )
            all_records.append({
                "text": f"[IMAGE DESCRIPTION — {source_name}, Page {img_info['page']}]\n{description}",
                "metadata": {
                    "source": source_name,
                    "title": title,
                    "page": img_info["page"],
                    "chunk_type": "image_description",
                    "chunk_index": img_info["image_index"],
                },
            })

        deleted = delete_document(collection, source_name)
        if all_records:
            add_records_to_chroma(collection, all_records, embed_model)

        return {"source": source_name, "chunks": len(all_records),
                "deleted": deleted, "error": None}

    except Exception as exc:
        log.error(f"ingest_single_file failed for {source_name}: {exc}")
        return {"source": source_name, "chunks": 0, "deleted": 0, "error": str(exc)}


# =============================================================================
# METADATA PATCH  (update titles without re-embedding)
# =============================================================================

def patch_titles(docs_dir: str, chroma_dir: str):
    """
    Re-extract titles from PDFs on disk and update ChromaDB metadata in-place.
    Does NOT re-embed or re-process images — cheap to run.
    """
    _, collection = get_or_create_collection(chroma_dir)
    pdf_files = list(Path(docs_dir).glob("**/*.pdf"))
    if not pdf_files:
        log.warning(f"No PDFs found in {docs_dir}")
        return

    for pdf_path in pdf_files:
        source_name = pdf_path.name
        title = extract_pdf_title(str(pdf_path))
        log.info(f"  {source_name} → {title}")

        result = collection.get(where={"source": source_name}, include=["metadatas"])
        ids      = result.get("ids", [])
        metas    = result.get("metadatas", [])
        if not ids:
            log.warning(f"  (no records found for {source_name})")
            continue

        updated = [dict(m, title=title) for m in metas]
        collection.update(ids=ids, metadatas=updated)
        log.info(f"  ✔ Updated {len(ids)} records")

    log.info("Title patch complete.")


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the Multimodal RAG knowledge base."
    )
    parser.add_argument(
        "--docs_dir",
        default="./docs",
        help="Directory of PDFs to ingest (default: ./docs)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Ingest a single PDF file instead of a whole directory",
    )
    parser.add_argument(
        "--patch_titles",
        action="store_true",
        help="Re-extract titles and update ChromaDB metadata only (no re-embedding)",
    )
    parser.add_argument(
        "--chroma_dir",
        default="./chroma_db",
        help="Path for persistent ChromaDB storage (default: ./chroma_db)",
    )
    args = parser.parse_args()

    if args.patch_titles:
        patch_titles(docs_dir=args.docs_dir, chroma_dir=args.chroma_dir)
    elif args.file:
        result = ingest_single_file(pdf_path=args.file, chroma_dir=args.chroma_dir)
        if result["error"]:
            log.error(f"Failed: {result['error']}")
        else:
            log.info(
                f"Done. {result['source']} — "
                f"{result['chunks']} chunks indexed"
                + (f", replaced {result['deleted']} old chunks" if result["deleted"] else "")
            )
    else:
        ingest_directory(docs_dir=args.docs_dir, chroma_dir=args.chroma_dir)
