import time
import numpy as np
from typing import List, Dict, Any, Tuple
from config import logger

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None


class ReRankingManager:
    """Advanced re-ranking strategies for RAG pipeline."""

    def __init__(self):
        self.cross_encoder = None
        self.reranking_models = {}
        self.performance_metrics = {}

    def initialize_cross_encoder(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder model for re-ranking."""
        if not CROSS_ENCODER_AVAILABLE:
            logger.error("sentence-transformers not available. Cross-encoder re-ranking will not work.")
            return False

        try:
            self.cross_encoder = CrossEncoder(model_name)
            logger.info(f"Initialized cross-encoder: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            return False

    def cross_encoder_rerank(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            top_k: int = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Re-rank documents using cross-encoder model."""
        if not self.cross_encoder:
            logger.warning("Cross-encoder not initialized. Returning original order.")
            return documents, {"error": "Cross-encoder not initialized"}

        start_time = time.time()

        try:
            # Prepare query-document pairs
            query_doc_pairs = []
            for doc in documents:
                content = doc.get("content", "")
                query_doc_pairs.append([query, content])

            # Get scores from cross-encoder
            scores = self.cross_encoder.predict(query_doc_pairs)

            # Sort documents by scores
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Extract reranked documents
            reranked_docs = [doc for doc, score in scored_docs]

            # Apply top_k filtering if specified
            if top_k:
                reranked_docs = reranked_docs[:top_k]

            end_time = time.time()

            metrics = {
                "reranking_time": end_time - start_time,
                "documents_processed": len(documents),
                "avg_time_per_doc": (end_time - start_time) / len(documents),
                "score_range": {
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores))
                }
            }

            logger.info(f"Cross-encoder reranked {len(documents)} documents in {metrics['reranking_time']:.3f}s")

            return reranked_docs, metrics

        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            return documents, {"error": str(e)}

    def relevance_score_rerank(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            weights: Dict[str, float] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Re-rank based on multiple relevance factors."""
        if weights is None:
            weights = {
                "content_similarity": 0.5,
                "metadata_relevance": 0.2,
                "document_quality": 0.2,
                "recency": 0.1
            }

        start_time = time.time()

        try:
            scored_docs = []

            for doc in documents:
                score = self._calculate_relevance_score(query, doc, weights)
                scored_docs.append((doc, score))

            # Sort by relevance score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, score in scored_docs]

            end_time = time.time()

            metrics = {
                "reranking_time": end_time - start_time,
                "documents_processed": len(documents),
                "weights_used": weights,
                "score_distribution": {
                    "min": min(score for _, score in scored_docs),
                    "max": max(score for _, score in scored_docs),
                    "mean": sum(score for _, score in scored_docs) / len(scored_docs)
                }
            }

            logger.info(f"Relevance-based reranked {len(documents)} documents")

            return reranked_docs, metrics

        except Exception as e:
            logger.error(f"Error in relevance score reranking: {e}")
            return documents, {"error": str(e)}

    def diversity_rerank(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            diversity_factor: float = 0.3,
            top_k: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Re-rank to promote diversity while maintaining relevance."""
        start_time = time.time()

        try:
            if len(documents) <= top_k:
                return documents, {"message": "No reranking needed, fewer documents than top_k"}

            # First, get relevance scores
            relevance_docs, _ = self.relevance_score_rerank(query, documents)

            # Then apply diversity selection
            selected_docs = []
            remaining_docs = relevance_docs.copy()

            # Select first document (highest relevance)
            if remaining_docs:
                selected_docs.append(remaining_docs.pop(0))

            # Select remaining documents balancing relevance and diversity
            while len(selected_docs) < top_k and remaining_docs:
                best_doc = None
                best_score = -1
                best_idx = -1

                for i, candidate in enumerate(remaining_docs):
                    # Calculate diversity score
                    diversity_score = self._calculate_diversity_score(candidate, selected_docs)

                    # Calculate relevance score (position-based for already ranked docs)
                    relevance_score = 1.0 - (i / len(remaining_docs))

                    # Combine scores
                    combined_score = (1 - diversity_factor) * relevance_score + diversity_factor * diversity_score

                    if combined_score > best_score:
                        best_score = combined_score
                        best_doc = candidate
                        best_idx = i

                if best_doc:
                    selected_docs.append(best_doc)
                    remaining_docs.pop(best_idx)

            end_time = time.time()

            metrics = {
                "reranking_time": end_time - start_time,
                "original_count": len(documents),
                "selected_count": len(selected_docs),
                "diversity_factor": diversity_factor,
                "diversity_score": self._calculate_overall_diversity(selected_docs)
            }

            logger.info(f"Diversity reranking selected {len(selected_docs)} documents")

            return selected_docs, metrics

        except Exception as e:
            logger.error(f"Error in diversity reranking: {e}")
            return documents[:top_k], {"error": str(e)}

    def _calculate_relevance_score(
            self,
            query: str,
            document: Dict[str, Any],
            weights: Dict[str, float]
    ) -> float:
        """Calculate multi-factor relevance score."""
        score = 0.0

        content = document.get("content", "")
        metadata = document.get("metadata", {})

        # Content similarity (simple keyword overlap)
        if weights.get("content_similarity", 0) > 0:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            if len(query_words) > 0:
                overlap = len(query_words.intersection(content_words)) / len(query_words)
                score += weights["content_similarity"] * overlap

        # Metadata relevance
        if weights.get("metadata_relevance", 0) > 0:
            metadata_score = 0.0

            # Chunk type preference
            chunk_type = metadata.get("chunk_type", "")
            if chunk_type in ["semantic", "hierarchical"]:
                metadata_score += 0.3
            elif chunk_type == "custom":
                metadata_score += 0.5

            # Document structure
            if metadata.get("section_title"):
                metadata_score += 0.2

            score += weights["metadata_relevance"] * metadata_score

        # Document quality
        if weights.get("document_quality", 0) > 0:
            quality_score = 0.0

            # Chunk size quality (prefer medium-sized chunks)
            chunk_size = metadata.get("chunk_size", 0)
            if 200 <= chunk_size <= 1000:
                quality_score += 0.5
            elif 100 <= chunk_size <= 2000:
                quality_score += 0.3

            # Special elements boost
            special_elements = metadata.get("special_elements", [])
            if special_elements:
                quality_score += 0.2

            score += weights["document_quality"] * quality_score

        # Recency (based on file index - later files are more recent)
        if weights.get("recency", 0) > 0:
            file_index = metadata.get("file_index", 0)
            # Normalize file index (assuming max 10 files)
            recency_score = min(file_index / 10.0, 1.0)
            score += weights["recency"] * recency_score

        return score

    def _calculate_diversity_score(
            self,
            candidate: Dict[str, Any],
            selected_docs: List[Dict[str, Any]]
    ) -> float:
        """Calculate how diverse a candidate document is from already selected ones."""
        if not selected_docs:
            return 1.0

        candidate_content = set(candidate.get("content", "").lower().split())

        similarities = []
        for selected in selected_docs:
            selected_content = set(selected.get("content", "").lower().split())

            if len(candidate_content) == 0 or len(selected_content) == 0:
                similarity = 0.0
            else:
                # Jaccard similarity
                intersection = len(candidate_content.intersection(selected_content))
                union = len(candidate_content.union(selected_content))
                similarity = intersection / union if union > 0 else 0.0

            similarities.append(similarity)

        # Return inverse of maximum similarity (higher diversity = lower max similarity)
        max_similarity = max(similarities)
        return 1.0 - max_similarity

    def _calculate_overall_diversity(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate overall diversity score for a document set."""
        if len(documents) <= 1:
            return 1.0

        total_similarity = 0.0
        pairs = 0

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                content_i = set(documents[i].get("content", "").lower().split())
                content_j = set(documents[j].get("content", "").lower().split())

                if len(content_i) > 0 and len(content_j) > 0:
                    intersection = len(content_i.intersection(content_j))
                    union = len(content_i.union(content_j))
                    similarity = intersection / union if union > 0 else 0.0
                    total_similarity += similarity
                    pairs += 1

        if pairs == 0:
            return 1.0

        avg_similarity = total_similarity / pairs
        return 1.0 - avg_similarity  # Higher diversity = lower average similarity

    def compare_reranking_methods(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            ground_truth_relevant: List[str] = None
    ) -> Dict[str, Any]:
        """Compare different reranking methods."""
        comparison_results = {}

        # Original order (baseline)
        comparison_results["original"] = {
            "documents": documents,
            "method": "original_order"
        }

        # Cross-encoder reranking
        if self.cross_encoder:
            ce_docs, ce_metrics = self.cross_encoder_rerank(query, documents)
            comparison_results["cross_encoder"] = {
                "documents": ce_docs,
                "metrics": ce_metrics,
                "method": "cross_encoder"
            }

        # Relevance score reranking
        rel_docs, rel_metrics = self.relevance_score_rerank(query, documents)
        comparison_results["relevance_score"] = {
            "documents": rel_docs,
            "metrics": rel_metrics,
            "method": "relevance_score"
        }

        # Diversity reranking
        div_docs, div_metrics = self.diversity_rerank(query, documents)
        comparison_results["diversity"] = {
            "documents": div_docs,
            "metrics": div_metrics,
            "method": "diversity"
        }

        # Evaluate against ground truth if provided
        if ground_truth_relevant:
            for method, result in comparison_results.items():
                evaluation = self._evaluate_reranking_quality(
                    result["documents"], ground_truth_relevant
                )
                result["evaluation"] = evaluation

        logger.info(f"Compared {len(comparison_results)} reranking methods")
        return comparison_results

    def _evaluate_reranking_quality(
            self,
            reranked_docs: List[Dict[str, Any]],
            ground_truth_relevant: List[str]
    ) -> Dict[str, float]:
        """Evaluate quality of reranking against ground truth."""
        relevant_set = set(ground_truth_relevant)

        # Extract document IDs from reranked docs
        doc_ids = []
        for doc in reranked_docs:
            doc_id = doc.get("metadata", {}).get("chunk_id", "")
            if doc_id:
                doc_ids.append(str(doc_id))

        if not doc_ids:
            return {"precision_at_5": 0.0, "recall_at_5": 0.0, "mrr": 0.0}

        # Calculate metrics
        top_5 = doc_ids[:5]
        relevant_in_top_5 = sum(1 for doc_id in top_5 if doc_id in relevant_set)

        precision_at_5 = relevant_in_top_5 / min(5, len(top_5))
        recall_at_5 = relevant_in_top_5 / len(relevant_set) if relevant_set else 0.0

        # Calculate MRR
        mrr = 0.0
        for i, doc_id in enumerate(doc_ids):
            if doc_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break

        return {
            "precision_at_5": precision_at_5,
            "recall_at_5": recall_at_5,
            "mrr": mrr
        }

    def generate_reranking_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate comprehensive reranking comparison report."""
        if not comparison_results:
            return "No comparison results available."

        report = []
        report.append("=== RE-RANKING COMPARISON REPORT ===\n")

        for method, results in comparison_results.items():
            report.append(f"**{method.upper()}**:")

            # Performance metrics
            if "metrics" in results:
                metrics = results["metrics"]
                if "reranking_time" in metrics:
                    report.append(f"  Performance:")
                    report.append(f"    - Reranking time: {metrics['reranking_time']:.3f}s")
                    report.append(f"    - Docs processed: {metrics.get('documents_processed', 0)}")
                    if "avg_time_per_doc" in metrics:
                        report.append(f"    - Avg time per doc: {metrics['avg_time_per_doc']:.4f}s")

            # Quality evaluation
            if "evaluation" in results:
                eval_metrics = results["evaluation"]
                report.append(f"  Quality:")
                report.append(f"    - Precision@5: {eval_metrics.get('precision_at_5', 0):.3f}")
                report.append(f"    - Recall@5: {eval_metrics.get('recall_at_5', 0):.3f}")
                report.append(f"    - MRR: {eval_metrics.get('mrr', 0):.3f}")

            # Method-specific metrics
            if method == "diversity" and "metrics" in results:
                div_score = results["metrics"].get("diversity_score", 0)
                report.append(f"    - Diversity score: {div_score:.3f}")

            report.append("")

        # Recommendations
        report.append("=== RECOMMENDATIONS ===")
        recommendations = self._generate_reranking_recommendations(comparison_results)
        for rec in recommendations:
            report.append(f"- {rec}")

        return "\n".join(report)

    def _generate_reranking_recommendations(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on reranking comparison."""
        recommendations = []

        # Find best performing method
        best_mrr = 0.0
        best_method = "original"

        for method, results in comparison_results.items():
            if "evaluation" in results:
                mrr = results["evaluation"].get("mrr", 0)
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_method = method

        if best_method != "original":
            recommendations.append(f"Use {best_method} reranking for best relevance (MRR: {best_mrr:.3f})")

        # Performance recommendations
        fastest_method = None
        fastest_time = float('inf')

        for method, results in comparison_results.items():
            if "metrics" in results and "reranking_time" in results["metrics"]:
                time_taken = results["metrics"]["reranking_time"]
                if time_taken < fastest_time:
                    fastest_time = time_taken
                    fastest_method = method

        if fastest_method:
            recommendations.append(f"For fastest performance, use {fastest_method} ({fastest_time:.3f}s)")

        # Specific method recommendations
        if "cross_encoder" in comparison_results:
            ce_mrr = comparison_results["cross_encoder"].get("evaluation", {}).get("mrr", 0)
            if ce_mrr > 0.7:
                recommendations.append("Cross-encoder shows excellent performance - recommended for production")
            elif ce_mrr < 0.3:
                recommendations.append("Cross-encoder may need fine-tuning for your domain")

        if "diversity" in comparison_results:
            div_score = comparison_results["diversity"].get("metrics", {}).get("diversity_score", 0)
            if div_score > 0.7:
                recommendations.append("High diversity achieved - good for exploratory queries")

        if not recommendations:
            recommendations.append("Consider fine-tuning reranking parameters based on your specific use case")

        return recommendations