import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from config import logger


class RetrievalEvaluator:
    """Comprehensive retrieval evaluation for RAG pipeline."""

    def __init__(self):
        self.ground_truth = {}
        self.test_queries = []
        self.evaluation_results = {}

    def load_ground_truth(self, ground_truth_data: Dict[str, List[str]]):
        """Load ground truth query-document pairs."""
        self.ground_truth = ground_truth_data
        self.test_queries = list(ground_truth_data.keys())
        logger.info(f"Loaded ground truth for {len(self.test_queries)} queries")

    def create_test_dataset(
            self,
            documents: List[Dict[str, Any]],
            num_easy: int = 10,
            num_medium: int = 10,
            num_hard: int = 10
    ) -> Dict[str, List[str]]:
        """Create test dataset with easy, medium, and hard queries."""
        test_dataset = {}

        # Easy queries: Direct content match
        easy_queries = self._generate_easy_queries(documents, num_easy)

        # Medium queries: Conceptual/semantic match
        medium_queries = self._generate_medium_queries(documents, num_medium)

        # Hard queries: Multi-hop reasoning, edge cases
        hard_queries = self._generate_hard_queries(documents, num_hard)

        test_dataset.update(easy_queries)
        test_dataset.update(medium_queries)
        test_dataset.update(hard_queries)

        logger.info(f"Created test dataset with {len(test_dataset)} queries")
        return test_dataset

    def _generate_easy_queries(self, documents: List[Dict[str, Any]], num_queries: int) -> Dict[str, List[str]]:
        """Generate easy queries that directly match document content."""
        queries = {}

        for i, doc in enumerate(documents[:num_queries]):
            content = doc.get("content", "")
            if len(content) > 100:
                # Extract a sentence or phrase as query
                sentences = content.split('.')
                if len(sentences) > 1:
                    query = sentences[0].strip() + "?"
                    queries[f"easy_query_{i}"] = [doc.get("metadata", {}).get("chunk_id", str(i))]

        return queries

    def _generate_medium_queries(self, documents: List[Dict[str, Any]], num_queries: int) -> Dict[str, List[str]]:
        """Generate medium difficulty queries requiring semantic understanding."""
        queries = {}

        # Example medium queries that would require semantic matching
        medium_templates = [
            "What is the main topic discussed in",
            "How does this relate to",
            "What are the key points about",
            "Explain the concept of",
            "What factors influence"
        ]

        for i in range(min(num_queries, len(documents))):
            doc = documents[i]
            content = doc.get("content", "")
            if len(content) > 50:
                template = medium_templates[i % len(medium_templates)]
                # Extract key terms from content
                words = content.split()[:10]
                key_term = max(words, key=len) if words else "topic"
                query = f"{template} {key_term}?"
                queries[f"medium_query_{i}"] = [doc.get("metadata", {}).get("chunk_id", str(i))]

        return queries

    def _generate_hard_queries(self, documents: List[Dict[str, Any]], num_queries: int) -> Dict[str, List[str]]:
        """Generate hard queries with edge cases and multi-hop reasoning."""
        queries = {}

        hard_templates = [
            "Compare and contrast",
            "What are the implications of",
            "How might this apply to",
            "What are the limitations of",
            "What evidence supports"
        ]

        for i in range(min(num_queries, len(documents) // 2)):
            # Multi-document queries
            doc1 = documents[i * 2]
            doc2 = documents[i * 2 + 1] if i * 2 + 1 < len(documents) else documents[i]

            template = hard_templates[i % len(hard_templates)]
            query = f"{template} the concepts discussed in the documents?"

            relevant_docs = [
                doc1.get("metadata", {}).get("chunk_id", str(i * 2)),
                doc2.get("metadata", {}).get("chunk_id", str(i * 2 + 1))
            ]
            queries[f"hard_query_{i}"] = relevant_docs

        return queries

    def evaluate_retrieval(
            self,
            retrieval_results: Dict[str, List[str]],
            k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate retrieval performance using multiple metrics."""
        if not self.ground_truth:
            logger.error("No ground truth data loaded")
            return {}

        metrics = {}

        for k in k_values:
            metrics[f"k_{k}"] = {
                "precision": self._calculate_precision_at_k(retrieval_results, k),
                "recall": self._calculate_recall_at_k(retrieval_results, k),
                "f1": 0.0,  # Will be calculated after precision and recall
                "hit_rate": self._calculate_hit_rate_at_k(retrieval_results, k),
                "ndcg": self._calculate_ndcg_at_k(retrieval_results, k)
            }

            # Calculate F1 score
            precision = metrics[f"k_{k}"]["precision"]
            recall = metrics[f"k_{k}"]["recall"]
            if precision + recall > 0:
                metrics[f"k_{k}"]["f1"] = 2 * (precision * recall) / (precision + recall)

        # Calculate MRR
        metrics["mrr"] = self._calculate_mrr(retrieval_results)

        # Overall metrics
        metrics["overall"] = self._calculate_overall_metrics(retrieval_results)

        self.evaluation_results = metrics
        logger.info("Retrieval evaluation completed")
        return metrics

    def _calculate_precision_at_k(self, retrieval_results: Dict[str, List[str]], k: int) -> float:
        """Calculate Precision@K."""
        precisions = []

        for query, retrieved_docs in retrieval_results.items():
            if query not in self.ground_truth:
                continue

            relevant_docs = set(self.ground_truth[query])
            retrieved_k = retrieved_docs[:k]

            if len(retrieved_k) == 0:
                precisions.append(0.0)
                continue

            relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_docs)
            precision = relevant_retrieved / len(retrieved_k)
            precisions.append(precision)

        return sum(precisions) / len(precisions) if precisions else 0.0

    def _calculate_recall_at_k(self, retrieval_results: Dict[str, List[str]], k: int) -> float:
        """Calculate Recall@K."""
        recalls = []

        for query, retrieved_docs in retrieval_results.items():
            if query not in self.ground_truth:
                continue

            relevant_docs = set(self.ground_truth[query])
            retrieved_k = set(retrieved_docs[:k])

            if len(relevant_docs) == 0:
                recalls.append(1.0 if len(retrieved_k) == 0 else 0.0)
                continue

            relevant_retrieved = len(relevant_docs.intersection(retrieved_k))
            recall = relevant_retrieved / len(relevant_docs)
            recalls.append(recall)

        return sum(recalls) / len(recalls) if recalls else 0.0

    def _calculate_hit_rate_at_k(self, retrieval_results: Dict[str, List[str]], k: int) -> float:
        """Calculate Hit Rate@K."""
        hits = 0
        total_queries = 0

        for query, retrieved_docs in retrieval_results.items():
            if query not in self.ground_truth:
                continue

            relevant_docs = set(self.ground_truth[query])
            retrieved_k = set(retrieved_docs[:k])

            if len(relevant_docs.intersection(retrieved_k)) > 0:
                hits += 1
            total_queries += 1

        return hits / total_queries if total_queries > 0 else 0.0

    def _calculate_ndcg_at_k(self, retrieval_results: Dict[str, List[str]], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        ndcg_scores = []

        for query, retrieved_docs in retrieval_results.items():
            if query not in self.ground_truth:
                continue

            relevant_docs = set(self.ground_truth[query])
            retrieved_k = retrieved_docs[:k]

            # Calculate DCG
            dcg = 0.0
            for i, doc in enumerate(retrieved_k):
                relevance = 1.0 if doc in relevant_docs else 0.0
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0

            # Calculate IDCG (Ideal DCG)
            ideal_relevances = [1.0] * min(len(relevant_docs), k)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    def _calculate_mrr(self, retrieval_results: Dict[str, List[str]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []

        for query, retrieved_docs in retrieval_results.items():
            if query not in self.ground_truth:
                continue

            relevant_docs = set(self.ground_truth[query])

            # Find rank of first relevant document
            for i, doc in enumerate(retrieved_docs):
                if doc in relevant_docs:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)  # No relevant document found

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    def _calculate_overall_metrics(self, retrieval_results: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        total_queries = len([q for q in retrieval_results.keys() if q in self.ground_truth])

        if total_queries == 0:
            return {}

        # Coverage metrics
        queries_with_results = sum(1 for q, results in retrieval_results.items()
                                   if q in self.ground_truth and len(results) > 0)

        # Average number of results per query
        avg_results_per_query = sum(len(results) for q, results in retrieval_results.items()
                                    if q in self.ground_truth) / total_queries

        return {
            "total_queries": total_queries,
            "coverage": queries_with_results / total_queries,
            "avg_results_per_query": avg_results_per_query
        }

    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report."""
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_retrieval first."

        report = []
        report.append("=== RETRIEVAL EVALUATION REPORT ===\n")

        # Overall metrics
        if "overall" in self.evaluation_results:
            overall = self.evaluation_results["overall"]
            report.append(f"Overall Performance:")
            report.append(f"  - Total queries evaluated: {overall.get('total_queries', 0)}")
            report.append(f"  - Coverage: {overall.get('coverage', 0):.3f}")
            report.append(f"  - Avg results per query: {overall.get('avg_results_per_query', 0):.1f}")
            report.append("")

        # MRR
        if "mrr" in self.evaluation_results:
            report.append(f"Mean Reciprocal Rank: {self.evaluation_results['mrr']:.3f}\n")

        # Metrics by K
        report.append("Performance by K:")
        for k_key in sorted([k for k in self.evaluation_results.keys() if k.startswith("k_")]):
            k_value = k_key.split("_")[1]
            metrics = self.evaluation_results[k_key]

            report.append(f"  K={k_value}:")
            report.append(f"    - Precision@{k_value}: {metrics.get('precision', 0):.3f}")
            report.append(f"    - Recall@{k_value}: {metrics.get('recall', 0):.3f}")
            report.append(f"    - F1@{k_value}: {metrics.get('f1', 0):.3f}")
            report.append(f"    - Hit Rate@{k_value}: {metrics.get('hit_rate', 0):.3f}")
            report.append(f"    - NDCG@{k_value}: {metrics.get('ndcg', 0):.3f}")
            report.append("")

        # Recommendations
        report.append("=== RECOMMENDATIONS ===")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"- {rec}")

        return "\n".join(report)

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        if not self.evaluation_results:
            return ["Run evaluation first to get recommendations."]

        # Check precision and recall balance
        k5_metrics = self.evaluation_results.get("k_5", {})
        precision_5 = k5_metrics.get("precision", 0)
        recall_5 = k5_metrics.get("recall", 0)

        if precision_5 < 0.3:
            recommendations.append(
                "Low precision suggests many irrelevant documents are being retrieved. Consider improving embedding quality or adding re-ranking.")

        if recall_5 < 0.5:
            recommendations.append(
                "Low recall suggests relevant documents are being missed. Consider increasing chunk overlap or using different chunking strategies.")

        # Check MRR
        mrr = self.evaluation_results.get("mrr", 0)
        if mrr < 0.5:
            recommendations.append(
                "Low MRR indicates relevant documents are not appearing at the top. Implement re-ranking to improve result ordering.")

        # Check hit rate
        hit_rate_5 = k5_metrics.get("hit_rate", 0)
        if hit_rate_5 < 0.7:
            recommendations.append(
                "Low hit rate suggests the retrieval system is missing relevant documents entirely. Review chunking strategy and embedding model.")

        # Check NDCG
        ndcg_5 = k5_metrics.get("ndcg", 0)
        if ndcg_5 < 0.6:
            recommendations.append(
                "Low NDCG indicates poor ranking quality. Consider implementing cross-encoder re-ranking.")

        if not recommendations:
            recommendations.append("Retrieval performance looks good across all metrics!")

        return recommendations

    def analyze_failure_cases(self, retrieval_results: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze queries where retrieval failed."""
        failure_analysis = {
            "zero_recall_queries": [],
            "low_precision_queries": [],
            "low_mrr_queries": [],
            "common_failure_patterns": []
        }

        for query, retrieved_docs in retrieval_results.items():
            if query not in self.ground_truth:
                continue

            relevant_docs = set(self.ground_truth[query])
            retrieved_set = set(retrieved_docs[:5])  # Check top 5

            # Zero recall cases
            if len(relevant_docs.intersection(retrieved_set)) == 0:
                failure_analysis["zero_recall_queries"].append({
                    "query": query,
                    "expected_docs": list(relevant_docs),
                    "retrieved_docs": retrieved_docs[:5]
                })

            # Low precision cases (< 20% relevant in top 5)
            relevant_count = len(relevant_docs.intersection(retrieved_set))
            if len(retrieved_set) > 0 and relevant_count / len(retrieved_set) < 0.2:
                failure_analysis["low_precision_queries"].append({
                    "query": query,
                    "precision": relevant_count / len(retrieved_set),
                    "retrieved_docs": retrieved_docs[:5]
                })

        logger.info(f"Analyzed {len(failure_analysis['zero_recall_queries'])} zero-recall queries")
        return failure_analysis