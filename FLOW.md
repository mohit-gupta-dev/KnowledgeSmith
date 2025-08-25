# MODEL REFERENCE FOR RAG PIPELINE

This document defines the models used in the RAG pipeline, both defaults and available alternatives. Notes include justifications for defaults and reasons why other options were not selected as primary.

---

## 1. PDF Text Extraction Models

* **Primary: PDFPlumber** – Default tool for extracting text, tables, and structure. Chosen for its balance of accuracy and robustness.
* **Fallback: PyPDF2** – Used if PDFPlumber fails. Simpler, but less reliable in preserving structure.
* **Tabula (optional)** – Enhances table extraction if installed. Not default due to heavier dependency footprint.
* **OCR (optional)** – If Tesseract + PIL are available, images are processed with OCR. Not default since most documents are text-based.

---

## 2. Chunking Models

Chunking splits documents into smaller units for embedding.

* **Recursive (default)** – Token-based splitting (512 tokens, 128 overlap). Default because it consistently balances precision and recall across document types.
* **Semantic** – Uses spaCy `en_core_web_sm` for sentence segmentation (falls back to regex if spaCy unavailable). Not default since NLP-based segmentation increases processing time.
* **Hierarchical** – Detects headers/sections (Markdown, numbers, chapters). Useful for technical docs but inconsistent with unstructured PDFs.
* **Custom** – Preserves special structures like code blocks and tables. Reserved for specialized cases.

---

## 3. Embedding Models

Convert text into vectors for retrieval.

* **Default: `nomic-embed-text`** – Chosen for high semantic quality and efficiency in retrieval tasks.
* **Alternatives:**

  * Ollama embeddings (local models)
  * SentenceTransformers (e.g., `all-MiniLM-L6-v2`) – Strong, but slower for large-scale production.
  * OpenAI embeddings (e.g., `text-embedding-ada-002`) – High quality but not default due to API cost and dependency.
  * Cohere embeddings (e.g., `embed-multilingual-v2.0`) – Multilingual support, not required for current scope.

---

## 4. Vector Stores

Databases that store and search embeddings.

* **Default: FAISS** – Local, efficient similarity search. Selected for sub-50ms latency with 100K+ vectors.
* **Alternatives (if installed):**

  * Chroma
  * Qdrant
  * Pinecone

Alternatives not chosen as defaults due to external dependencies or cost factors.

---

## 5. Re-ranking Models

Improve retrieval quality by reordering results.

* **Default: Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`** – Default due to strong accuracy-speed trade-off (re-ranks 20 docs <100ms).
* **Alternatives:**

  * Relevance-based reranking (content similarity, metadata, quality, recency) – Faster, but less accurate.
  * Diversity-based reranking (balance relevance with variety) – Adds coverage but weaker relevance precision.

---

## 6. Large Language Model (LLM)

Used to generate responses with retrieved context.

* **Default tuning:** Mistral-type LLM, 4096-token context, temperature = 0.1 (low randomness). Chosen for factual stability and efficient inference.
* **Supported runtimes:**

  * Ollama local models
  * OpenAI Chat models (needs configuration)

1024-token chunk setups were considered but rejected as they reduced retrieval precision and context diversity.

---

## 7. Evaluation Models

For testing embedding, retrieval, and reranking quality.

* **Embedding Evaluation:**

  * Basic: 2 models, 10 samples
  * Standard: 3 models, 20 samples
  * Comprehensive: all available models, 50 samples
  * Metrics: similarity, isotropy, clustering, alignment, uniformity

* **Retrieval Evaluation:**

  * Test queries (easy, medium, hard)
  * Metrics: `Precision\@K`, `Recall\@K`, `NDCG`, `MRR`

* **Re-ranking Evaluation:**

  * Compare cross-encoder vs. relevance vs. diversity methods

---

## Default Model Stack Summary

* **PDF Extraction:** PDFPlumber
* **Chunking:** Recursive (512 tokens, 128 overlap)
* **Embedding:** `nomic-embed-text`
* **Vector Store:** FAISS
* **Re-ranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **LLM:** Mistral-style model (low temp, long context)

Optional modules (spaCy, Tabula, OCR, SentenceTransformers, OpenAI, Cohere, Chroma, Qdrant, Pinecone) can extend functionality when available. Defaults were selected for performance, reliability, and production cost efficiency.
