# Retrieval-Augmented Generation (RAG) — Full Notes and Chunk Planner

This document consolidates all the core principles we discussed about building a robust RAG pipeline. It covers **retrieval, reranking, parsing, chunking, and generation** with open-source components.

---

## RAG Basics

1. **Ingest**
   - Parse documents (PDF, HTML, MD).
   - Split into chunks with semantic boundaries.
   - Embed chunks into vectors.
   - Store vectors + metadata in a vector store (FAISS, Qdrant, Milvus, pgvector, etc.).

2. **Retrieve**
   - Encode the query into a vector.
   - ANN search retrieves candidate chunks.
   - Retrieve more (top-50) to ensure recall.

3. **Rerank (optional but recommended)**
   - Cross-encoder (CE) or LLM reranker reorders candidate chunks.
   - Select top-5/10 for final context.

4. **Generate**
   - Send query + top chunks to the generator LLM.
   - Prompt model to cite sources (page, section).

---

## Retrieval Engines (OSS)

- **FAISS** — library only, blazing fast ANN, minimal dependencies. Good for embedded/single-node.
- **Qdrant** — vector DB with REST/gRPC, metadata filtering, snapshots. Great production choice.
- **Milvus** — distributed vector DB, high scale.
- **pgvector** — Postgres extension, good if you already run Postgres.
- **OpenSearch/Elasticsearch** — hybrid BM25 + ANN search, heavy but feature-rich.
- **Lightweight** (LanceDB, SQLite-vec, USearch) — local/offline workloads.

---

## Reranking

- **Cross-Encoder reranker** — Fast, accurate, cheap. Best default.
- **LLM rerank** — Slower, costlier, better at reasoning-heavy queries.
- **Embedding filter** — Threshold drop-in, less effective.

**Prod pattern**: Retrieve top-50 → rerank to top-10 → generator.

---

## PDF Parsing (OSS)

- **PyMuPDF (fitz)** — fast, structured text + images.
- **pdfminer.six** — low-level text extraction.
- **pdfplumber** — tables + text, built on pdfminer.
- **Unstructured** — splits into semantic blocks (headings, lists, tables).
- **Docling (IBM)** — modern, layout-aware, outputs structured JSON.

**Rule**: Always preserve atomic blocks (tables, code, formulas). Chunk after parsing.

---

## Chunking Strategy

### General recommendations
- Chunk size: **800–1,200 tokens**.
- Overlap: **10–15%**.
- Use **tokens not characters**.
- Adjust to LLM context (leave ~30% free for prompt + answer).

### By document type
- **Prose/manuals:** 1,200–1,600 tokens, 10% overlap.
- **Math/proofs:** 1,000–1,400 tokens, 15–20% overlap.
- **Tables/code:** 300–600 tokens, 15–25% overlap.
- **OCR/noisy scans:** 500–900 tokens, 20% overlap.
- **Mixed:** 900–1,100 tokens, 15% overlap.

### Why these numbers
- Retrieval recall plateaus ~1k tokens.
- Too small = context fragmentation.
- Too large = wasted context and fewer retrievable units.
- Overlap <10% loses boundary context; >25% duplicates too much.

---

## Auto Chunk Planner (Token-Aware)

```python
"""
RAG Chunk Planner
Automates chunk size + overlap using doc type + structure + token stats.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple
import re

Tokenizer = Callable[[str], List[int]]
Classifier = Callable[[str], str]

@dataclass
class ChunkPlan:
    doc_type: str
    chunk_size: int
    overlap: int
    top_k: int
    notes: str

BASE = {
    "prose": (1100, 0.12),
    "tables": (500, 0.22),
    "code": (500, 0.22),
    "math": (1300, 0.18),
    "ocr_noisy": (750, 0.20),
    "mixed": (1000, 0.15),
}

def estimate_structure(first_page: str) -> dict:
    lines = [l for l in first_page.splitlines() if l.strip()]
    avg_line_len = sum(len(l) for l in lines)/max(1,len(lines))
    n_tables = sum(1 for l in lines if "|" in l and "-" in l)
    n_code = sum(1 for l in lines if l.strip().startswith(("def","class","import","#")) or "```" in l)
    n_math = sum(1 for l in lines if re.search(r"[=+\-*/^]|\\(frac|sum|int)", l))
    return dict(avg_line_len=avg_line_len, n_tables=n_tables, n_code=n_code, n_math=n_math)

def clamp(v, lo, hi): return max(lo,min(hi,v))

def auto_top_k(ctx: int, size: int, safety=0.7) -> int:
    return max(5, min(30, int((ctx*safety)//size)))

def plan_chunks(full_text_sample: str, first_page_text: str,
                tokenizer: Tokenizer, classify: Classifier,
                gen_context_budget: int=32000) -> ChunkPlan:
    doc_type = classify(first_page_text)
    size, ov = BASE.get(doc_type, BASE["mixed"])
    S = estimate_structure(first_page_text)
    if S["n_tables"]+S["n_code"]>0: size-=200; ov+=0.03
    if S["avg_line_len"]>90: size+=200
    if S["n_math"]>8: size+=100; ov+=0.02
    paras=[p for p in re.split(r"\n\s*\n", full_text_sample) if p.strip()]
    if paras:
        lens=[len(tokenizer(p)) for p in paras[:50]]
        med=sorted(lens)[len(lens)//2]
        target=int(clamp(2.5*med,300,2400))
        size=int(0.5*size+0.5*target)
    size=clamp(size,300,2400); ov=clamp(ov,0.10,0.25)
    overlap=int(size*ov)
    top_k=auto_top_k(gen_context_budget,size)
    return ChunkPlan(doc_type,size,overlap,top_k,
        f"Fitted for {gen_context_budget} ctx; {top_k} chunks at ~{size} tokens each.")
```

---

Generator LLM Choice

Qwen3 8B-Instruct — fast, good reasoning, 32k+ context. Great default.

Llama-3.1 8B Instruct — strong reasoning, slightly slower.

Mistral 7B Instruct — fastest, weaker reasoning.


Settings: temperature=0.2, max_tokens=800–1200, top_p=0.9.


---

Best Practices

Always use reranking in prod (CE default).

Always preserve structure (tables, code, headings).

Parse PDFs with layout-aware tools (PyMuPDF + Unstructured/Docling).

Evaluate with Recall@k and Context hit rate before shipping.

Leave ~30% of context free for system prompt + answer space.



---

## Putting It All Together — RAG Pipeline Flow

1. **Document ingestion**
   - Parse PDF/HTML/Markdown.
   - Extract text + metadata (page, section, heading).
   - Apply chunk planner → token-aware spans.
   - Embed each chunk with your embedding model (e.g., bge-large, e5).
   - Store in FAISS/Qdrant with metadata.

2. **Query flow**
   - Encode query → embedding.
   - **Retrieve**: FAISS/Qdrant top-50.
   - **Rerank**:
     - Cross-Encoder by default.
     - Escalate to LLM rerank if query is complex (multi-constraint, temporal, reasoning-heavy).
   - Select top-5/10 chunks.

3. **Answer generation**
   - Build prompt with query + context.
   - Feed into generator LLM (Qwen3-8B, Llama3-8B, etc.).
   - Constrain with instructions: “Answer concisely and cite page/section.”

4. **Output**
   - Return answer with citations.
   - Optionally display supporting chunks.

---

## Example Prompt for Generator LLM

You are a helpful assistant. Answer the question using ONLY the provided context. If the answer cannot be found, say "Not in the provided document."

Question: {query}

Context: {context_chunks}

Answer (cite page numbers when present):


---

## Key Trade-offs

- **Chunk size**
  - Too small → fragmented meaning, high recall but low precision.
  - Too big → wasted context, lower retrievability.

- **Reranking**
  - Adds latency but increases precision.
  - Cross-encoder = fast + accurate.
  - LLM rerank = reasoning but slower.

- **Vector store**
  - FAISS = lightweight, local.
  - Qdrant/Milvus = production DB with filtering.
  - pgvector = SQL-friendly, slower on scale.

---

## Operational Considerations

- **Evaluation**
  - Build a gold Q&A set.
  - Measure Recall@k, MRR, Exact match.
  - Check hallucination rate.

- **Monitoring**
  - Log query → retrieved chunks → final answer.
  - Track reranker decisions (CE vs LLM).
  - Watch latency and embedding drift.

- **Scaling**
  - Batch embeddings at ingest.
  - Use HNSW index for FAISS/Qdrant.
  - Warm-up generator LLM at startup.

---

## Takeaways

- Chunking is **document-type dependent**.
- Reranking should be **on by default**.
- Final answer LLM: **8B models** (Qwen3/Llama3) are sweet spot for quality vs latency.
- Parsing is as important as retrieval: **preserve structure**.

**Core mantra:**  
> *Parse clean, chunk smart, rerank always, generate carefully.*