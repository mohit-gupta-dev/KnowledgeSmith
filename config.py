import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Application constants
APP_TITLE = "PDF RAG"

# Chunking configuration - OPTIMIZED for better retrieval
CHUNK_SIZE_TOKENS = 512  # Reduced from 1024 for more focused chunks
CHUNK_OVERLAP_TOKENS = 128  # Reduced from 256 to avoid too much redundancy
CHUNK_SIZE = 2048  # Approximate character equivalent (4 chars per token)
CHUNK_OVERLAP = 512  # Approximate character equivalent

# Token estimation constants
CHARS_PER_TOKEN = 4  # Average characters per token for English text
TOKENS_PER_WORD = 1.3  # Average tokens per word

# LLM Configuration - OPTIMIZED for Mistral
TEMPERATURE = 0.1  # Keep low for factual accuracy
MAX_CONTEXT_LENGTH = 4096  # Maximum context length for LLM
CONTEXT_BUFFER = 512  # Buffer to ensure we don't exceed limits

# Vector Database Configuration
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "nomic-embed-text"
EXCLUDED_MODELS = []

# Supported vector databases
VECTOR_DB_TYPES = ["faiss", "chroma", "qdrant", "pinecone"]
DEFAULT_VECTOR_DB = "faiss"

# Batch processing configuration
BATCH_SIZE = 100  # Number of documents to process in batch
MAX_BATCH_SIZE = 1000

# Evaluation configuration
EVALUATION_METRICS = ["precision", "recall", "f1", "mrr", "ndcg", "hit_rate"]
K_VALUES = [1, 3, 5, 10]

# Response quality thresholds - ADJUSTED for better quality control
MIN_RESPONSE_LENGTH = 50
MAX_RESPONSE_LENGTH = 2000
HALLUCINATION_KEYWORDS = [
    "I don't have information", "not mentioned in", "cannot find",
    "not specified", "unclear from the document", "not provided in the documents",
    "this information is not", "the documents do not", "unable to find"
]

# UI constants
DEFAULT_ZOOM = 700
MIN_ZOOM = 100
MAX_ZOOM = 1000
ZOOM_STEP = 50
PDF_VIEWER_HEIGHT = 500
CHAT_HEIGHT = 500

# File processing limits
MAX_FILE_SIZE_MB = 100
SUPPORTED_FILE_TYPES = [".pdf"]

# Re-ranking configuration
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Default cross-encoder model
RERANK_TOP_K = 10  # Number of documents to keep after re-ranking
INITIAL_RETRIEVAL_K = 20  # Number of documents to retrieve before re-ranking

# Quality control thresholds
MIN_CHUNK_LENGTH = 50  # Minimum chunk length in characters
MAX_CHUNK_LENGTH = 8000  # Maximum chunk length in characters
MIN_CHUNK_WORDS = 10  # Minimum words per chunk
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score for retrieval

def get_token_count_estimate(text: str) -> int:
    """Estimate token count from text."""
    return len(text) // CHARS_PER_TOKEN

def get_char_count_from_tokens(tokens: int) -> int:
    """Convert token count to approximate character count."""
    return tokens * CHARS_PER_TOKEN