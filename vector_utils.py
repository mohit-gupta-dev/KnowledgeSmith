import os
import shutil
import time
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import logger, FAISS_INDEX_PATH, EMBEDDING_MODEL, BATCH_SIZE, MAX_BATCH_SIZE

# Optional imports for different vector databases
try:
    import chromadb
    from langchain_community.vectorstores import Chroma

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    Chroma = None

try:
    import qdrant_client
    from langchain_community.vectorstores import Qdrant

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    Qdrant = None

try:
    import pinecone
    from langchain_community.vectorstores import Pinecone

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    Pinecone = None


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def create_vector_store(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> bool:
        """Create vector store from texts and metadata."""
        pass

    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> bool:
        """Add texts to existing vector store."""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Any]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete_vector_store(self) -> bool:
        """Delete vector store."""
        pass

    @abstractmethod
    def get_vector_store(self):
        """Get vector store instance."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation with enhanced features."""

    def __init__(self, embeddings, index_path: str = FAISS_INDEX_PATH):
        self.embeddings = embeddings
        self.index_path = index_path
        self.vector_store = None
        self.metadata_store = {}  # Enhanced metadata storage

    def create_vector_store(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> bool:
        """Create FAISS vector store with batch processing."""
        try:
            if not texts:
                logger.error("No texts provided for vector store creation")
                return False

            # Process in batches for large datasets
            total_texts = len(texts)
            batch_size = min(BATCH_SIZE, total_texts)

            logger.info(f"Creating FAISS vector store with {total_texts} texts in batches of {batch_size}")

            # Create initial vector store with first batch
            start_idx = 0
            end_idx = min(batch_size, total_texts)
            batch_texts = texts[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx] if metadatas else None

            self.vector_store = FAISS.from_texts(
                batch_texts,
                embedding=self.embeddings,
                metadatas=batch_metadatas
            )

            # Store enhanced metadata
            if batch_metadatas:
                for i, metadata in enumerate(batch_metadatas):
                    self.metadata_store[start_idx + i] = metadata

            # Add remaining batches
            start_idx = end_idx
            while start_idx < total_texts:
                end_idx = min(start_idx + batch_size, total_texts)
                batch_texts = texts[start_idx:end_idx]
                batch_metadatas = metadatas[start_idx:end_idx] if metadatas else None

                logger.info(f"Processing batch {start_idx // batch_size + 1}: texts {start_idx} to {end_idx - 1}")

                # Add batch to vector store
                ids = self.vector_store.add_texts(batch_texts, metadatas=batch_metadatas)

                # Store enhanced metadata
                if batch_metadatas:
                    for i, metadata in enumerate(batch_metadatas):
                        self.metadata_store[start_idx + i] = metadata

                start_idx = end_idx

            # Save to disk
            self.vector_store.save_local(self.index_path)
            self._save_metadata_store()

            logger.info(f"FAISS vector store created successfully with {total_texts} documents")
            return True

        except Exception as e:
            logger.error(f"Error creating FAISS vector store: {e}")
            return False

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> bool:
        """Add texts to existing FAISS vector store."""
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return False

            # Process in batches
            total_texts = len(texts)
            batch_size = min(BATCH_SIZE, total_texts)
            current_size = len(self.metadata_store)

            start_idx = 0
            while start_idx < total_texts:
                end_idx = min(start_idx + batch_size, total_texts)
                batch_texts = texts[start_idx:end_idx]
                batch_metadatas = metadatas[start_idx:end_idx] if metadatas else None

                # Add batch to vector store
                self.vector_store.add_texts(batch_texts, metadatas=batch_metadatas)

                # Update metadata store
                if batch_metadatas:
                    for i, metadata in enumerate(batch_metadatas):
                        self.metadata_store[current_size + start_idx + i] = metadata

                start_idx = end_idx

            # Save updated vector store
            self.vector_store.save_local(self.index_path)
            self._save_metadata_store()

            logger.info(f"Added {total_texts} texts to FAISS vector store")
            return True

        except Exception as e:
            logger.error(f"Error adding texts to FAISS vector store: {e}")
            return False

    def similarity_search(self, query: str, k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Any]:
        """Enhanced similarity search with metadata filtering."""
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return []

            # Perform similarity search
            docs = self.vector_store.similarity_search(query, k=k * 2)  # Get more docs for filtering

            # Apply metadata filtering if specified
            if filter_dict:
                filtered_docs = []
                for doc in docs:
                    doc_metadata = getattr(doc, 'metadata', {})
                    if self._matches_filter(doc_metadata, filter_dict):
                        filtered_docs.append(doc)
                docs = filtered_docs[:k]  # Limit to requested k
            else:
                docs = docs[:k]

            return docs

        except Exception as e:
            logger.error(f"Error in FAISS similarity search: {e}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 5, filter_dict: Dict[str, Any] = None) -> List[tuple]:
        """Similarity search with relevance scores."""
        try:
            if not self.vector_store:
                return []

            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k * 2)

            # Apply metadata filtering if specified
            if filter_dict:
                filtered_docs = []
                for doc, score in docs_with_scores:
                    doc_metadata = getattr(doc, 'metadata', {})
                    if self._matches_filter(doc_metadata, filter_dict):
                        filtered_docs.append((doc, score))
                docs_with_scores = filtered_docs[:k]
            else:
                docs_with_scores = docs_with_scores[:k]

            return docs_with_scores

        except Exception as e:
            logger.error(f"Error in FAISS similarity search with score: {e}")
            return []

    def delete_vector_store(self) -> bool:
        """Delete FAISS vector store and metadata."""
        try:
            self.vector_store = None
            self.metadata_store = {}

            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
                logger.info("FAISS index deleted from disk")

            metadata_path = f"{self.index_path}_metadata.json"
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                logger.info("FAISS metadata deleted from disk")

            return True
        except Exception as e:
            logger.error(f"Error deleting FAISS vector store: {e}")
            return False

    def get_vector_store(self):
        """Get FAISS vector store instance."""
        return self.vector_store

    def load_vector_store(self) -> bool:
        """Load existing FAISS vector store."""
        try:
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._load_metadata_store()
                logger.info("FAISS vector store loaded from disk")
                return True
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}")
        return False

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False

            if isinstance(value, dict):
                # Handle range queries, e.g., {"chunk_size_tokens": {"$gte": 100, "$lte": 1000}}
                for op, val in value.items():
                    if op == "$gte" and metadata[key] < val:
                        return False
                    elif op == "$lte" and metadata[key] > val:
                        return False
                    elif op == "$eq" and metadata[key] != val:
                        return False
                    elif op == "$ne" and metadata[key] == val:
                        return False
            elif isinstance(value, list):
                # Handle "in" queries
                if metadata[key] not in value:
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False

        return True

    def _save_metadata_store(self):
        """Save enhanced metadata to disk."""
        import json
        metadata_path = f"{self.index_path}_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata_store, f)
        except Exception as e:
            logger.error(f"Error saving metadata store: {e}")

    def _load_metadata_store(self):
        """Load enhanced metadata from disk."""
        import json
        metadata_path = f"{self.index_path}_metadata.json"
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata_store = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata store: {e}")
            self.metadata_store = {}


class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store implementation."""

    def __init__(self, embeddings, collection_name: str = "rag_collection"):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")

        self.embeddings = embeddings
        self.collection_name = collection_name
        self.vector_store = None
        self.client = None

    def create_vector_store(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> bool:
        """Create Chroma vector store."""
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")

            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                collection_name=self.collection_name,
                client=self.client
            )

            logger.info(f"Chroma vector store created with {len(texts)} documents")
            return True

        except Exception as e:
            logger.error(f"Error creating Chroma vector store: {e}")
            return False

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> bool:
        """Add texts to Chroma vector store."""
        try:
            if not self.vector_store:
                return False

            self.vector_store.add_texts(texts, metadatas=metadatas)
            logger.info(f"Added {len(texts)} texts to Chroma vector store")
            return True

        except Exception as e:
            logger.error(f"Error adding texts to Chroma vector store: {e}")
            return False

    def similarity_search(self, query: str, k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Any]:
        """Chroma similarity search with filtering."""
        try:
            if not self.vector_store:
                return []

            # Chroma has built-in filtering support
            return self.vector_store.similarity_search(query, k=k, filter=filter_dict)

        except Exception as e:
            logger.error(f"Error in Chroma similarity search: {e}")
            return []

    def delete_vector_store(self) -> bool:
        """Delete Chroma vector store."""
        try:
            if self.client:
                self.client.delete_collection(self.collection_name)

            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")

            self.vector_store = None
            self.client = None
            logger.info("Chroma vector store deleted")
            return True

        except Exception as e:
            logger.error(f"Error deleting Chroma vector store: {e}")
            return False

    def get_vector_store(self):
        """Get Chroma vector store instance."""
        return self.vector_store


class VectorStoreManager:
    """Enhanced vector store manager supporting multiple backends."""

    def __init__(self, vector_db_type: str = "faiss"):
        self.vector_db_type = vector_db_type.lower()
        self.embeddings = None
        self.vector_store_impl = None
        self.performance_metrics = {}

        # Validate vector DB type
        available_types = ["faiss"]
        if CHROMA_AVAILABLE:
            available_types.append("chroma")
        if QDRANT_AVAILABLE:
            available_types.append("qdrant")
        if PINECONE_AVAILABLE:
            available_types.append("pinecone")

        if self.vector_db_type not in available_types:
            logger.warning(f"Vector DB type '{vector_db_type}' not available. Using FAISS.")
            self.vector_db_type = "faiss"

        logger.info(f"Initialized VectorStoreManager with {self.vector_db_type} backend")

    def get_embeddings(self) -> OllamaEmbeddings:
        """Get or create Ollama embeddings instance."""
        if self.embeddings is None:
            logger.info("Creating Ollama Embeddings")
            self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        return self.embeddings

    def _initialize_vector_store_impl(self):
        """Initialize the appropriate vector store implementation."""
        if self.vector_store_impl is None:
            embeddings = self.get_embeddings()

            if self.vector_db_type == "faiss":
                self.vector_store_impl = FAISSVectorStore(embeddings)
            elif self.vector_db_type == "chroma" and CHROMA_AVAILABLE:
                self.vector_store_impl = ChromaVectorStore(embeddings)
            else:
                # Fallback to FAISS
                self.vector_store_impl = FAISSVectorStore(embeddings)
                logger.warning(f"Falling back to FAISS for vector storage")

    def create_vector_store(self, text_chunks: List[str], chunk_metadata: List[Dict[str, Any]] = None) -> Optional[Any]:
        """Create vector store with enhanced metadata and performance tracking."""
        try:
            start_time = time.time()

            if not text_chunks:
                logger.error("No text chunks provided")
                return None

            # Initialize vector store implementation
            self._initialize_vector_store_impl()

            # Prepare metadata if not provided
            if chunk_metadata is None:
                chunk_metadata = [{"chunk_id": i} for i in range(len(text_chunks))]

            # Enhance metadata with additional information
            enhanced_metadata = []
            for i, (text, metadata) in enumerate(zip(text_chunks, chunk_metadata)):
                enhanced_meta = {
                    **metadata,
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "created_at": time.time(),
                    "vector_db_type": self.vector_db_type
                }
                enhanced_metadata.append(enhanced_meta)

            # Create vector store
            success = self.vector_store_impl.create_vector_store(text_chunks, enhanced_metadata)

            end_time = time.time()

            # Record performance metrics
            self.performance_metrics = {
                "creation_time": end_time - start_time,
                "total_documents": len(text_chunks),
                "avg_doc_length": sum(len(text) for text in text_chunks) / len(text_chunks),
                "vector_db_type": self.vector_db_type,
                "indexing_speed": len(text_chunks) / (end_time - start_time),
                "timestamp": time.time()
            }

            if success:
                logger.info(f"Vector store created successfully in {end_time - start_time:.2f}s")
                return self.vector_store_impl.get_vector_store()
            else:
                logger.error("Failed to create vector store")
                return None

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None

    def add_documents(self, text_chunks: List[str], chunk_metadata: List[Dict[str, Any]] = None) -> bool:
        """Add documents to existing vector store."""
        try:
            if not self.vector_store_impl:
                logger.error("Vector store not initialized")
                return False

            # Prepare metadata if not provided
            if chunk_metadata is None:
                chunk_metadata = [{"chunk_id": f"new_{i}"} for i in range(len(text_chunks))]

            # Enhance metadata
            enhanced_metadata = []
            for text, metadata in zip(text_chunks, chunk_metadata):
                enhanced_meta = {
                    **metadata,
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "added_at": time.time(),
                    "vector_db_type": self.vector_db_type
                }
                enhanced_metadata.append(enhanced_meta)

            success = self.vector_store_impl.add_texts(text_chunks, enhanced_metadata)

            if success:
                logger.info(f"Added {len(text_chunks)} documents to vector store")

            return success

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False

    def search_similar_documents(self, query: str, k: int = 5, filter_dict: Dict[str, Any] = None,
                                 return_scores: bool = False) -> Union[List[Any], List[tuple]]:
        """Enhanced similarity search with filtering and optional scores."""
        try:
            start_time = time.time()

            if not self.vector_store_impl:
                logger.error("Vector store not initialized")
                return []

            # Perform search
            if return_scores and hasattr(self.vector_store_impl, 'similarity_search_with_score'):
                results = self.vector_store_impl.similarity_search_with_score(query, k, filter_dict)
            else:
                results = self.vector_store_impl.similarity_search(query, k, filter_dict)

            search_time = time.time() - start_time

            # Update performance metrics
            self.performance_metrics.update({
                "last_search_time": search_time,
                "last_search_results": len(results),
                "last_search_query_length": len(query)
            })

            logger.info(f"Found {len(results)} similar documents in {search_time:.3f}s")
            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def load_vector_store(self) -> Optional[Any]:
        """Load existing vector store."""
        try:
            self._initialize_vector_store_impl()

            if hasattr(self.vector_store_impl, 'load_vector_store'):
                if self.vector_store_impl.load_vector_store():
                    return self.vector_store_impl.get_vector_store()

            return None

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None

    def delete_vector_store(self) -> bool:
        """Delete vector store and clear memory."""
        try:
            success = True

            if self.vector_store_impl:
                success = self.vector_store_impl.delete_vector_store()
                self.vector_store_impl = None

            # Clear performance metrics
            self.performance_metrics = {}

            if success:
                logger.info("Vector store deleted successfully")

            return success

        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")
            return False

    def get_vector_store(self) -> Optional[Any]:
        """Get current vector store instance."""
        if self.vector_store_impl:
            return self.vector_store_impl.get_vector_store()
        return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the vector store."""
        return self.performance_metrics.copy()

    def get_available_vector_dbs(self) -> List[str]:
        """Get list of available vector database types."""
        available = ["faiss"]

        if CHROMA_AVAILABLE:
            available.append("chroma")
        if QDRANT_AVAILABLE:
            available.append("qdrant")
        if PINECONE_AVAILABLE:
            available.append("pinecone")

        return available

    def switch_vector_db(self, new_db_type: str) -> bool:
        """Switch to a different vector database type."""
        available_types = self.get_available_vector_dbs()

        if new_db_type.lower() not in available_types:
            logger.error(f"Vector DB type '{new_db_type}' not available")
            return False

        # Delete current vector store
        if self.vector_store_impl:
            self.delete_vector_store()

        # Switch to new type
        self.vector_db_type = new_db_type.lower()
        self.vector_store_impl = None

        logger.info(f"Switched to {self.vector_db_type} vector database")
        return True

    def benchmark_performance(self, test_queries: List[str]) -> Dict[str, float]:
        """Benchmark search performance with test queries."""
        if not test_queries or not self.vector_store_impl:
            return {}

        search_times = []
        result_counts = []

        for query in test_queries:
            start_time = time.time()
            results = self.vector_store_impl.similarity_search(query, k=5)
            search_time = time.time() - start_time

            search_times.append(search_time)
            result_counts.append(len(results))

        return {
            "avg_search_time": sum(search_times) / len(search_times),
            "min_search_time": min(search_times),
            "max_search_time": max(search_times),
            "avg_results_returned": sum(result_counts) / len(result_counts),
            "total_queries_tested": len(test_queries)
        }


# Global instance
vector_manager = VectorStoreManager()