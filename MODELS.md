# RAG Pipeline Models and Algorithms Overview

This extended document provides detailed descriptions of the models, algorithms, and strategies used in the Retrieval-Augmented Generation (RAG) pipeline. It is designed to help with interview preparation where you need to explain both the theoretical and practical aspects of the codebase and PoC.

---

## 1. PDF Text Extraction Models

* **PDFPlumber (Default)**
  A Python library that accurately extracts text, tables, and layouts from PDFs. It is preferred because it retains structural fidelity, which is important for downstream processing like chunking and embeddings. It handles multi-column layouts, font information, and whitespace more reliably than other libraries.

* **PyPDF2 (Fallback)**
  A simpler, lighter library for PDF parsing. It can extract text but does not preserve formatting or table structures well. Used as a backup when PDFPlumber encounters errors.

* **Tabula (Optional)**
  Built on top of Apache PDFBox and Java, Tabula specializes in table extraction. It is useful when the primary focus is data tables within PDFs, such as financial or retail reports. Not default because it introduces a Java dependency footprint.

* **Tesseract OCR (Optional, with PIL)**
  Optical Character Recognition (OCR) engine for image-based PDFs (e.g., scanned documents). Converts rasterized text into machine-readable form. Not the default since most documents in the pipeline are digitally generated PDFs with embedded text.

---

## 2. Chunking Models

Chunking divides documents into smaller units to be embedded and retrieved.

* **Recursive Chunking (Default)**
  Token-based splitting with 512 tokens per chunk and 128 tokens overlap. This ensures continuity across boundaries and provides a balance between recall (capturing enough context) and precision (minimizing noise). Overlap helps capture concepts that span chunk boundaries.

* **Semantic Chunking**
  Relies on linguistic segmentation using spaCy (with regex fallback). Produces more natural and semantically coherent chunks, especially in narrative or descriptive documents. However, it is computationally heavier.

* **Hierarchical Chunking**
  Preserves document structure by detecting headers, numbered lists, and chapters. Ideal for technical documents, manuals, or academic papers where logical hierarchy matters.

* **Custom Chunking**
  Specialized strategy for preserving atomic blocks such as tables, code snippets, or lists. This avoids fragmenting structured data which could reduce semantic meaning.

---

## 3. Embedding Models

Convert text into dense vector representations for similarity search.

* **nomic-embed-text (Default)**
  Produces 768-dimensional embeddings optimized for semantic search tasks. Balances performance and computational efficiency. Strong isotropy properties ensure high-quality clustering in vector space.

* **Alternatives**:

  * **SentenceTransformers (all-MiniLM-L6-v2)**: Transformer-based embeddings known for semantic precision. Slower at scale.
  * **OpenAI Embeddings (text-embedding-ada-002)**: Very high accuracy embeddings with strong generalization. Excluded as default due to API costs.
  * **Cohere Embeddings (embed-multilingual-v2.0)**: Multilingual support, good for global use cases.
  * **Ollama Embeddings**: On-device models that provide privacy and cost control.

---

## 4. Vector Stores

Databases optimized for similarity search.

* **FAISS (Default)**
  Developed by Facebook AI Research, FAISS supports approximate nearest neighbor (ANN) search. Efficiently scales to millions of embeddings while maintaining sub-50ms retrieval latency.

* **Alternatives**:

  * **Chroma**: Developer-friendly, supports persistence and filtering.
  * **Qdrant**: Built with Rust, provides production-grade features and hybrid search.
  * **Pinecone**: Fully managed cloud-native vector database. Avoided as default due to recurring costs.

---

## 5. Re-ranking Models

Refine retrieval candidates by reordering based on relevance.

* **Cross-encoder (ms-marco-MiniLM-L-6-v2, Default)**
  A transformer-based model fine-tuned on MS MARCO dataset. Evaluates query-document pairs directly to produce relevance scores. Balances speed and accuracy by handling \~20 documents within 100ms. Improves Mean Reciprocal Rank (MRR) by \~40%.

* **Relevance-based reranking**
  Uses similarity scores, metadata, or recency to reorder results. Fast but less semantically accurate.

* **Diversity-based reranking**
  Ensures coverage of different aspects of the query. Useful for exploratory tasks but sacrifices precision.

---

## 6. Large Language Models (LLMs)

Generate the final response by integrating retrieved context.

* **Mistral-type Model (Default)**
  An open-weight transformer LLM with 4096-token context window. Low temperature (0.1) is used for deterministic and factual responses. Efficient inference makes it suitable for production.

* **Alternatives**:

  * **Ollama Local Models**: Privacy-friendly and cost-efficient but limited in model variety.
  * **OpenAI Chat Models**: API-based, high performance but expensive and dependent on external service.

---

## 7. Evaluation Models

Used for testing and validating the pipeline.

* **Embedding Evaluation**:
  Metrics include isotropy (vector space uniformity), clustering (semantic grouping), and similarity (cosine/dot product performance).

* **Retrieval Evaluation**:
  Precision\@K, Recall\@K, NDCG (Normalized Discounted Cumulative Gain), and MRR (Mean Reciprocal Rank) used to measure retrieval effectiveness.

* **Re-ranking Evaluation**:
  A/B testing cross-encoder reranking vs. relevance/diversity methods. Provides insights into latency vs. accuracy trade-offs.

---

## 8. Justified Defaults

* **Chunk size:** 512 tokens. Smaller than 1024 for better precision and diversity of included context.
* **Overlap:** 128 tokens (\~25%) to prevent boundary issues while limiting redundancy.
* **Vector Store:** FAISS chosen for its sub-50ms latency at scale without added costs.
* **Reranker:** Cross-encoder MiniLM for the best balance of performance and accuracy.
* **LLM:** Mistral-type with low randomness for factual reliability.

---

## Default Pipeline Summary

* **Text Extraction:** PDFPlumber
* **Chunking:** Recursive (512 tokens, 128 overlap)
* **Embeddings:** nomic-embed-text
* **Vector Store:** FAISS
* **Re-ranking:** ms-marco-MiniLM-L-6-v2
* **LLM:** Mistral-type (low temperature, long context)

This stack ensures reliable, cost-efficient, and production-ready performance while keeping the architecture modular enough to swap in alternatives when needed.
