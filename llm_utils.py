import json
import os
import urllib
import re
import time
from urllib.error import URLError, HTTPError
from typing import List, Dict, Any, Tuple

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

from config import (
    logger, TEMPERATURE, EXCLUDED_MODELS, MAX_CONTEXT_LENGTH,
    CONTEXT_BUFFER, MIN_RESPONSE_LENGTH, MAX_RESPONSE_LENGTH,
    HALLUCINATION_KEYWORDS, get_token_count_estimate
)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    estimated_chars = max_tokens * 4  # Rough estimate
    if len(text) <= estimated_chars:
        return text

    # Truncate at sentence boundary if possible
    truncated = text[:estimated_chars]
    last_sentence = truncated.rfind('.')

    if last_sentence > estimated_chars * 0.7:  # If we can keep most of the text
        return truncated[:last_sentence + 1] + "..."
    else:
        return truncated + "..."


def _extract_source_info(metadata: Dict[str, Any]) -> str:
    """Extract source information from metadata."""
    source_parts = []

    if "source_file" in metadata:
        filename = os.path.basename(metadata["source_file"])
        source_parts.append(f"File: {filename}")

    if "chunk_id" in metadata:
        source_parts.append(f"Chunk: {metadata['chunk_id']}")

    if "section_title" in metadata and metadata["section_title"]:
        source_parts.append(f"Section: {metadata['section_title']}")

    if "page_number" in metadata:
        source_parts.append(f"Page: {metadata['page_number']}")

    return " | ".join(source_parts) if source_parts else "Unknown source"


class ContextManager:
    """Manages LLM context length and content preparation."""

    def __init__(self, max_context_length: int = MAX_CONTEXT_LENGTH):
        self.max_context_length = max_context_length
        self.context_buffer = CONTEXT_BUFFER
        self.available_context = max_context_length - self.context_buffer

    def prepare_context(self, query: str, retrieved_docs: List[Any],
                        system_prompt: str = "", prioritize_by_relevance: bool = True) -> Tuple[
        str, List[Dict[str, Any]]]:
        """Prepare context while respecting token limits and prioritizing relevance."""
        # Estimate token usage
        query_tokens = get_token_count_estimate(query)
        system_tokens = get_token_count_estimate(system_prompt)

        # Reserve tokens for response
        response_buffer = 500  # Reserve tokens for response
        available_for_docs = self.available_context - query_tokens - system_tokens - response_buffer

        if available_for_docs <= 0:
            logger.warning("Query and system prompt too long, minimal context available")
            return "", []

        # Sort documents by relevance if they have scores
        if prioritize_by_relevance:
            # Try to extract scores from documents if available
            docs_with_scores = []
            for i, doc in enumerate(retrieved_docs):
                # Priority order ensures re-ranked docs stay at top
                score = getattr(doc, 'score', 1.0 - (i * 0.01))  # Descending score by position
                docs_with_scores.append((score, doc))

            # Sort by score descending
            docs_with_scores.sort(key=lambda x: x[0], reverse=True)
            retrieved_docs = [doc for _, doc in docs_with_scores]

        # Select and truncate documents to fit context
        selected_docs = []
        used_tokens = 0

        for doc in retrieved_docs:
            doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            doc_tokens = get_token_count_estimate(doc_content)

            if used_tokens + doc_tokens <= available_for_docs:
                selected_docs.append({
                    "content": doc_content,
                    "metadata": getattr(doc, 'metadata', {}),
                    "tokens": doc_tokens
                })
                used_tokens += doc_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = available_for_docs - used_tokens
                if remaining_tokens > 100:  # Minimum meaningful size
                    truncated_content = _truncate_to_tokens(doc_content, remaining_tokens)
                    selected_docs.append({
                        "content": truncated_content,
                        "metadata": getattr(doc, 'metadata', {}),
                        "tokens": remaining_tokens,
                        "truncated": True
                    })
                    used_tokens = available_for_docs
                break

        # Combine documents into context
        context_parts = []
        for i, doc in enumerate(selected_docs):
            source_info = _extract_source_info(doc.get("metadata", {}))
            context_parts.append(f"[Source {i + 1}] {source_info}\n{doc['content']}")

        context = "\n\n".join(context_parts)

        logger.info(f"Prepared context with {len(selected_docs)} documents, {used_tokens} tokens")
        return context, selected_docs


def _check_response_length(response: str) -> Dict[str, Any]:
    """Check if response length is appropriate."""
    length = len(response)
    word_count = len(response.split())

    return {
        "character_count": length,
        "word_count": word_count,
        "is_too_short": length < MIN_RESPONSE_LENGTH,
        "is_too_long": length > MAX_RESPONSE_LENGTH,
        "is_appropriate": MIN_RESPONSE_LENGTH <= length <= MAX_RESPONSE_LENGTH
    }


def _check_context_relevance(response: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check how well response uses provided context."""
    if not context_docs:
        return {"relevance_score": 0.0, "context_usage": 0}

    response_words = set(response.lower().split())
    context_overlap_scores = []

    for doc in context_docs:
        doc_words = set(doc["content"].lower().split())
        if doc_words:
            overlap = len(response_words.intersection(doc_words))
            overlap_ratio = overlap / len(doc_words.union(response_words))
            context_overlap_scores.append(overlap_ratio)

    avg_relevance = sum(context_overlap_scores) / len(context_overlap_scores) if context_overlap_scores else 0

    return {
        "relevance_score": avg_relevance,
        "context_usage": len([score for score in context_overlap_scores if score > 0.1]),
        "docs_utilized": len([score for score in context_overlap_scores if score > 0.05])
    }


def _check_query_relevance(response: str, query: str) -> Dict[str, Any]:
    """Check how well response addresses the query."""
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())

    if not query_words:
        return {"relevance_score": 0.0}

    overlap = len(query_words.intersection(response_words))
    relevance_score = overlap / len(query_words)

    # Check if response directly addresses query intent
    question_indicators = ["what", "how", "why", "when", "where", "who", "which"]
    query_has_question = any(indicator in query.lower() for indicator in question_indicators)

    return {
        "relevance_score": relevance_score,
        "addresses_question": query_has_question,
        "keyword_coverage": overlap / len(query_words) if query_words else 0
    }


def _check_source_attribution(response: str) -> Dict[str, Any]:
    """Check for proper source attribution."""
    # Look for source references
    source_patterns = [
        r'\[Source \d+\]', r'according to', r'based on', r'as stated in',
        r'the document', r'the text', r'page \d+', r'section \d+'
    ]

    attribution_count = sum(1 for pattern in source_patterns
                            if re.search(pattern, response, re.IGNORECASE))

    return {
        "attribution_count": attribution_count,
        "has_proper_attribution": attribution_count > 0,
        "attribution_density": attribution_count / len(response.split()) if response else 0
    }


def _check_coherence(response: str) -> Dict[str, Any]:
    """Check response coherence and structure."""
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        return {"coherence_score": 1.0, "sentence_count": len(sentences)}

    # Simple coherence check based on word overlap between consecutive sentences
    coherence_scores = []
    for i in range(len(sentences) - 1):
        words1 = set(sentences[i].lower().split())
        words2 = set(sentences[i + 1].lower().split())

        if words1 and words2:
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            coherence_scores.append(overlap / union if union > 0 else 0)

    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

    return {
        "coherence_score": avg_coherence,
        "sentence_count": len(sentences),
        "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences)
    }


def _extract_confidence_indicators(response: str) -> Dict[str, Any]:
    """Extract confidence indicators from response."""
    high_confidence = ["definitely", "certainly", "clearly", "obviously", "undoubtedly"]
    medium_confidence = ["likely", "probably", "appears", "seems", "suggests"]
    low_confidence = ["might", "could", "possibly", "perhaps", "may be"]

    response_lower = response.lower()

    return {
        "high_confidence_count": sum(1 for word in high_confidence if word in response_lower),
        "medium_confidence_count": sum(1 for word in medium_confidence if word in response_lower),
        "low_confidence_count": sum(1 for word in low_confidence if word in response_lower),
        "overall_confidence": "high" if any(word in response_lower for word in high_confidence)
        else "medium" if any(word in response_lower for word in medium_confidence)
        else "low" if any(word in response_lower for word in low_confidence)
        else "neutral"
    }


def _calculate_overall_score(analysis: Dict[str, Any]) -> float:
    """Calculate overall quality score."""
    score = 0.0

    # Length appropriateness (0.1 weight)
    if analysis["length_check"]["is_appropriate"]:
        score += 0.1

    # Hallucination risk (0.3 weight)
    hallucination = analysis["hallucination_check"]
    if hallucination["risk_level"] == "low":
        score += 0.3
    elif hallucination["risk_level"] == "medium":
        score += 0.15

    # Context relevance (0.25 weight)
    score += analysis["context_relevance"]["relevance_score"] * 0.25

    # Query relevance (0.25 weight)
    score += analysis["query_relevance"]["relevance_score"] * 0.25

    # Source attribution (0.1 weight)
    if analysis["source_attribution"]["has_proper_attribution"]:
        score += 0.1

    return min(1.0, score)


def validate_response_accuracy(response: str, context: str, question: str) -> dict:
    """Validate that response claims exist in context."""

    # Extract claims from response
    response_sentences = response.split('.')

    validation_results = {
        "claims_verified": 0,
        "claims_unverified": 0,
        "verification_details": []
    }

    for sentence in response_sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:  # Skip very short fragments
            continue

        # Check if key terms from sentence exist in context
        key_terms = [word for word in sentence.split() if len(word) > 4]

        # Count how many key terms are found in context
        terms_found = 0
        for term in key_terms:
            if term.lower() in context.lower():
                terms_found += 1

        # Consider claim verified if >50% of key terms are in context
        if key_terms and (terms_found / len(key_terms)) > 0.5:
            validation_results["claims_verified"] += 1
        else:
            validation_results["claims_unverified"] += 1
            validation_results["verification_details"].append(
                f"Unverified: {sentence[:100]}..."
            )

    return validation_results


class ResponseQualityAnalyzer:
    """Analyzes response quality and detects potential issues."""

    def __init__(self):
        self.hallucination_indicators = HALLUCINATION_KEYWORDS
        self.quality_metrics = {}

    def analyze_response(self, response: str, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive response quality analysis."""
        analysis = {
            "length_check": _check_response_length(response),
            "hallucination_check": self._check_hallucination_indicators(response),
            "context_relevance": _check_context_relevance(response, context_docs),
            "query_relevance": _check_query_relevance(response, query),
            "source_attribution": _check_source_attribution(response),
            "coherence": _check_coherence(response),
            "confidence_indicators": _extract_confidence_indicators(response),
            "overall_score": 0.0
        }

        # Calculate overall quality score
        analysis["overall_score"] = _calculate_overall_score(analysis)

        return analysis

    def _check_hallucination_indicators(self, response: str) -> Dict[str, Any]:
        """Check for hallucination indicators."""
        response_lower = response.lower()

        # Check for uncertainty phrases (good indicators)
        uncertainty_phrases = [
            "according to the document", "based on the provided text",
            "the document states", "as mentioned in", "the text indicates"
        ]

        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response_lower)

        # Check for definitive statements without attribution (potential hallucination)
        definitive_patterns = [
            r'\bis known that\b', r'\bit is certain\b', r'\bobviously\b',
            r'\bclearly\b', r'\bundoubtedly\b', r'\bwithout question\b'
        ]

        definitive_count = sum(1 for pattern in definitive_patterns
                               if re.search(pattern, response_lower))

        # Check for hallucination keywords
        hallucination_count = sum(1 for keyword in self.hallucination_indicators
                                  if keyword.lower() in response_lower)

        return {
            "uncertainty_phrases": uncertainty_count,
            "definitive_statements": definitive_count,
            "hallucination_keywords": hallucination_count,
            "has_attribution": uncertainty_count > 0,
            "risk_level": "low" if uncertainty_count > definitive_count else "medium" if definitive_count <= 2 else "high"
        }


def extract_model_names() -> Tuple[str, ...]:
    """Return chat-capable Ollama models. Robust to client/schema changes."""
    host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    url = f"{host.rstrip('/')}/api/tags"

    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.load(r)
    except (URLError, HTTPError) as e:
        logger.error("Failed to reach Ollama at %s: %s", url, e)
        return tuple()

    names = []
    for m in data.get("models", []):
        # tolerate both 'name' and 'model'
        name = m.get("name") or m.get("model")
        if not name:
            continue
        base = name.split(":", 1)[0].lower()

        # exclude known embedding families by name
        if any(x in base for x in ["embed", "e5", "bge", "gte", "nomic-embed", "text-embedding", "minilm"]):
            continue

        # respect local excludes (match on base)
        if base in {x.lower() for x in EXCLUDED_MODELS}:
            continue

        names.append(name)

    if not names:
        logger.warning("No chat models after filtering. Raw: %s",
                       [m.get("name") or m.get("model") for m in data.get("models", [])])
    return tuple(sorted(set(names)))


def _enhance_citations(response: str, context_docs: List[Dict[str, Any]]) -> str:
    """Enhance citations in the response."""
    # Already handled in prompt template
    return response


def _has_source_summary(response: str) -> bool:
    """Check if response already has source information."""
    source_indicators = ["source", "according to", "based on", "from the document"]
    response_lower = response.lower()
    return any(indicator in response_lower for indicator in source_indicators)


def _add_source_summary(response: str, context_docs: List[Dict[str, Any]]) -> str:
    """Add source summary to response."""
    if not context_docs or _has_source_summary(response):
        return response

    source_lines = []
    for i, doc in enumerate(context_docs[:3]):  # Limit to top 3 sources
        metadata = doc.get("metadata", {})
        source_info = _extract_source_info(metadata)
        source_lines.append(f"{i + 1}. {source_info}")

    if source_lines:
        sources_text = "\n\n**Sources:**\n" + "\n".join(source_lines)
        return response + sources_text

    return response


def _create_response_dict(response: str, question: str, context_docs: List[Dict[str, Any]],
                          model: str, processing_time: float, error: bool = False,
                          reranking_applied: bool = False) -> Dict[str, Any]:
    """Create standardized response dictionary."""
    return {
        "response": response,
        "question": question,
        "model": model,
        "processing_time": processing_time,
        "timestamp": time.time(),
        "context_docs_count": len(context_docs),
        "context_docs": context_docs,
        "error": error,
        "response_length": len(response),
        "response_word_count": len(response.split()),
        "reranking_applied": reranking_applied
    }


class LLMManager:
    """Enhanced LLM operations manager with context handling and quality analysis."""

    def __init__(self):
        self.llm_cache = {}
        self.context_manager = ContextManager()
        self.quality_analyzer = ResponseQualityAnalyzer()
        self.response_history = []

    def get_llm(self, selected_model: str) -> ChatOllama:
        """Get or create ChatOllama instance for selected model."""
        if selected_model not in self.llm_cache:
            logger.info(f"Creating LLM instance for model: {selected_model}")
            self.llm_cache[selected_model] = ChatOllama(
                model=selected_model,
                temperature=TEMPERATURE
            )
        return self.llm_cache[selected_model]

    def process_question(self, question: str, vector_db: FAISS, selected_model: str,
                         enable_quality_analysis: bool = True,
                         enable_reranking: bool = False,
                         reranking_manager=None) -> Dict[str, Any]:
        """Enhanced question processing with proper re-ranking integration."""
        start_time = time.time()

        logger.info(f"Processing question: {question} using model: {selected_model}")
        logger.info(
            f"Re-ranking enabled: {enable_reranking}, Re-ranking manager available: {reranking_manager is not None}")

        try:
            llm = self.get_llm(selected_model)

            # Define the query prompt template
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="Original question: {question}",
            )

            # Retrieve MORE documents initially for re-ranking
            initial_k = 20 if enable_reranking else 10

            # Create retriever with LLM for multiple query retrieval
            retriever = MultiQueryRetriever.from_llm(
                vector_db.as_retriever(search_kwargs={"k": initial_k}),
                llm, prompt=QUERY_PROMPT
            )

            # Retrieve documents
            retrieved_docs = retriever.get_relevant_documents(question)
            logger.info(f"Retrieved {len(retrieved_docs)} initial documents")

            reranking_applied = False

            # APPLY RE-RANKING BEFORE CONTEXT PREPARATION
            if enable_reranking and reranking_manager:
                try:
                    if hasattr(reranking_manager, 'cross_encoder') and reranking_manager.cross_encoder:
                        logger.info("Applying cross-encoder re-ranking...")

                        # Convert documents for re-ranking
                        documents = []
                        for doc in retrieved_docs:
                            documents.append({
                                "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                                "metadata": getattr(doc, 'metadata', {})
                            })

                        # Re-rank and get top documents
                        reranked_docs, rerank_metrics = reranking_manager.cross_encoder_rerank(
                            question, documents, top_k=10
                        )

                        # Convert back to original format
                        class SimpleDoc:
                            def __init__(self, content, metadata, score=1.0):
                                self.page_content = content
                                self.metadata = metadata
                                self.score = score

                        retrieved_docs = [
                            SimpleDoc(
                                d["content"],
                                d["metadata"],
                                rerank_metrics.get("score_range", {}).get("max", 1.0) - i * 0.1
                            )
                            for i, d in enumerate(reranked_docs)
                        ]

                        reranking_applied = True
                        logger.info(f"Re-ranked to {len(retrieved_docs)} documents")
                        logger.info(f"Re-ranking scores: {rerank_metrics.get('score_range', {})}")
                    else:
                        logger.warning("Cross-encoder not initialized in re-ranking manager")
                except Exception as e:
                    logger.error(f"Re-ranking failed: {e}")
                    # Continue with original documents if re-ranking fails

            # Prepare context with re-ranked or original documents
            context, selected_docs = self.context_manager.prepare_context(
                question, retrieved_docs, prioritize_by_relevance=True
            )

            if not context:
                return _create_response_dict(
                    "I don't have sufficient information to answer your question based on the provided documents.",
                    question, [], selected_model, time.time() - start_time, reranking_applied=reranking_applied
                )

            # Enhanced prompt template for better accuracy (especially for Mistral)
            template = """You are a precise AI assistant analyzing documents. You must follow these rules EXACTLY:

CRITICAL RULES:
1. ONLY use information EXPLICITLY stated in the context below
2. Do NOT add, infer, or assume ANY information not directly written
3. If the specific answer is not in the context, say: "This information is not provided in the documents"
4. Use exact quotes from the context when possible with quotation marks
5. Cite sources using [Source X] for every claim
6. Do NOT create or invent terms, categories, or classifications not in the documents
7. Be extremely literal - only state what is written word-for-word

Context from documents:
{context}

Question: {question}

Instructions: Answer using ONLY the exact information in the context above. If you cannot find the specific answer, clearly state that it's not in the documents. Use direct quotes and source citations.

Answer:"""

            prompt = ChatPromptTemplate.from_template(template)

            # Set up the chain
            chain = (
                    {"context": lambda _: context, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )

            # Get the response from the chain
            response = chain.invoke(question)

            # Validate response accuracy
            validation = validate_response_accuracy(response, context, question)

            if validation["claims_unverified"] > validation["claims_verified"]:
                logger.warning(f"High number of unverified claims: {validation['claims_unverified']}")
                # Prepend warning to response
                response = "⚠️ Note: Some claims in this response may not be fully supported by the documents.\n\n" + response

            # Post-process response
            processed_response = self._post_process_response(response, selected_docs)

            # Create response dictionary
            response_dict = _create_response_dict(
                processed_response, question, selected_docs, selected_model,
                time.time() - start_time, reranking_applied=reranking_applied
            )

            # Add validation results
            response_dict["validation"] = validation

            # Perform quality analysis if enabled
            if enable_quality_analysis:
                quality_analysis = self.quality_analyzer.analyze_response(
                    processed_response, question, selected_docs
                )
                response_dict["quality_analysis"] = quality_analysis

            # Store response in history
            self.response_history.append(response_dict)

            logger.info(f"Question processed. Re-ranking applied: {reranking_applied}")
            return response_dict

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return _create_response_dict(
                f"Error processing your question: {str(e)}",
                question, [], selected_model, time.time() - start_time,
                error=True, reranking_applied=False
            )

    def _post_process_response(self, response: str, context_docs: List[Dict[str, Any]]) -> str:
        """Post-process response to enhance citations and formatting."""
        if not response or not response.strip():
            return "I don't have sufficient information to answer your question based on the provided documents."

        # Enhance source citations
        enhanced_response = _enhance_citations(response, context_docs)

        # Add source summary if not present
        enhanced_response = _add_source_summary(enhanced_response, context_docs)

        return enhanced_response

    def get_response_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent response history."""
        return self.response_history[-limit:] if self.response_history else []

    def clear_response_history(self):
        """Clear response history."""
        self.response_history = []

    def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics across models."""
        if not self.response_history:
            return {}

        stats = {}
        for response in self.response_history:
            model = response["model"]
            if model not in stats:
                stats[model] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                    "avg_response_length": 0,
                    "error_count": 0,
                    "reranking_used_count": 0
                }

            stats[model]["count"] += 1
            stats[model]["total_time"] += response["processing_time"]
            stats[model]["avg_response_length"] += response["response_length"]

            if response.get("error", False):
                stats[model]["error_count"] += 1

            if response.get("reranking_applied", False):
                stats[model]["reranking_used_count"] += 1

        # Calculate averages
        for model_stats in stats.values():
            if model_stats["count"] > 0:
                model_stats["avg_time"] = model_stats["total_time"] / model_stats["count"]
                model_stats["avg_response_length"] = model_stats["avg_response_length"] / model_stats["count"]
                model_stats["error_rate"] = model_stats["error_count"] / model_stats["count"]
                model_stats["reranking_rate"] = model_stats["reranking_used_count"] / model_stats["count"]

        return stats

    def analyze_hallucination_patterns(self) -> Dict[str, Any]:
        """Analyze hallucination patterns across responses."""
        if not self.response_history:
            return {}

        hallucination_data = []

        for response in self.response_history:
            if "quality_analysis" in response:
                hallucination_check = response["quality_analysis"]["hallucination_check"]
                hallucination_data.append({
                    "model": response["model"],
                    "risk_level": hallucination_check["risk_level"],
                    "has_attribution": hallucination_check["has_attribution"],
                    "uncertainty_phrases": hallucination_check["uncertainty_phrases"],
                    "definitive_statements": hallucination_check["definitive_statements"],
                    "reranking_applied": response.get("reranking_applied", False)
                })

        if not hallucination_data:
            return {}

        # Aggregate statistics
        total_responses = len(hallucination_data)
        high_risk_count = sum(1 for item in hallucination_data if item["risk_level"] == "high")
        medium_risk_count = sum(1 for item in hallucination_data if item["risk_level"] == "medium")
        low_risk_count = sum(1 for item in hallucination_data if item["risk_level"] == "low")

        with_attribution = sum(1 for item in hallucination_data if item["has_attribution"])
        with_reranking = sum(1 for item in hallucination_data if item["reranking_applied"])

        return {
            "total_responses_analyzed": total_responses,
            "high_risk_percentage": (high_risk_count / total_responses) * 100,
            "medium_risk_percentage": (medium_risk_count / total_responses) * 100,
            "low_risk_percentage": (low_risk_count / total_responses) * 100,
            "attribution_percentage": (with_attribution / total_responses) * 100,
            "reranking_percentage": (with_reranking / total_responses) * 100,
            "avg_uncertainty_phrases": sum(
                item["uncertainty_phrases"] for item in hallucination_data) / total_responses,
            "avg_definitive_statements": sum(
                item["definitive_statements"] for item in hallucination_data) / total_responses
        }


# Global instance
llm_manager = LLMManager()