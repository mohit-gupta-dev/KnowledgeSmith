import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from langchain_ollama import OllamaEmbeddings
from config import logger

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    import cohere

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    cohere = None


def _calculate_effective_dimension(embeddings_array: np.ndarray) -> float:
    """Calculate effective dimension using PCA."""
    try:
        from sklearn.decomposition import PCA

        # Perform PCA
        pca = PCA()
        pca.fit(embeddings_array)

        # Calculate effective dimension (number of components needed for 95% variance)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        effective_dim = np.argmax(cumsum >= 0.95) + 1

        return float(effective_dim)
    except ImportError:
        logger.warning("sklearn not available for effective dimension calculation")
        return float(embeddings_array.shape[1])


def _calculate_isotropy(embeddings_array: np.ndarray) -> float:
    """Calculate isotropy score (uniformity of embedding distribution)."""
    try:
        # Center the embeddings
        centered = embeddings_array - np.mean(embeddings_array, axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)

        # Calculate eigenvalues
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Filter out near-zero eigenvalues

        if len(eigenvals) <= 1:
            return 1.0

        # Isotropy is related to the ratio of min to max eigenvalue
        isotropy = np.min(eigenvals) / np.max(eigenvals)
        return float(isotropy)

    except Exception as e:
        logger.warning(f"Error calculating isotropy: {e}")
        return 0.0


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity based on word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0.0


def _evaluate_clustering_quality(embeddings: List[List[float]], texts: List[str]) -> dict[str, str] | dict[
    Any, Any]:
    """Evaluate clustering quality of embeddings."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        embeddings_array = np.array(embeddings)

        if len(embeddings_array) < 10:
            return {"error": "Not enough samples for clustering evaluation"}

        # Try different numbers of clusters
        best_silhouette = -1
        best_k = 2

        metrics = {}

        for k in range(2, min(10, len(embeddings_array) // 2)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings_array)

                # Calculate silhouette score
                silhouette = silhouette_score(embeddings_array, cluster_labels)

                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k

                    # Calculate additional metrics for best clustering
                    ch_score = calinski_harabasz_score(embeddings_array, cluster_labels)

                    metrics.update({
                        "best_silhouette_score": float(silhouette),
                        "best_k_clusters": k,
                        "calinski_harabasz_score": float(ch_score),
                        "inertia": float(kmeans.inertia_)
                    })

            except Exception as e:
                logger.warning(f"Clustering failed for k={k}: {e}")
                continue

        return metrics

    except ImportError:
        logger.warning("sklearn not available for clustering evaluation")
        return {"error": "sklearn not available"}
    except Exception as e:
        logger.warning(f"Error in clustering evaluation: {e}")
        return {"error": str(e)}


def _calculate_overall_score(quality_metrics: Dict, similarity_metrics: Dict,
                             clustering_metrics: Dict, retrieval_metrics: Dict) -> float:
    """Calculate overall embedding quality score."""
    score = 0.0
    weight_sum = 0.0

    # Quality metrics (40% weight)
    if "avg_cosine_similarity" in quality_metrics:
        score += quality_metrics["avg_cosine_similarity"] * 0.1
        weight_sum += 0.1

    if "isotropy_score" in quality_metrics:
        score += quality_metrics["isotropy_score"] * 0.1
        weight_sum += 0.1

    if "alignment_score" in quality_metrics:
        score += quality_metrics["alignment_score"] * 0.1
        weight_sum += 0.1

    if "uniformity_score" in quality_metrics:
        score += quality_metrics["uniformity_score"] * 0.1
        weight_sum += 0.1

    # Similarity metrics (30% weight)
    if "avg_similarity" in similarity_metrics:
        score += similarity_metrics["avg_similarity"] * 0.3
        weight_sum += 0.3

    # Clustering metrics (20% weight)
    if "best_silhouette_score" in clustering_metrics:
        # Normalize silhouette score from [-1, 1] to [0, 1]
        normalized_silhouette = (clustering_metrics["best_silhouette_score"] + 1) / 2
        score += normalized_silhouette * 0.2
        weight_sum += 0.2

    # Retrieval metrics (10% weight)
    if "avg_precision_at_5" in retrieval_metrics:
        score += retrieval_metrics["avg_precision_at_5"] * 0.1
        weight_sum += 0.1

    return score / weight_sum if weight_sum > 0 else 0.0


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def _calculate_alignment_uniformity(embeddings_array: np.ndarray, texts: List[str]) -> Tuple[float, float]:
    """Calculate alignment and uniformity metrics."""
    try:
        # Alignment: how well similar texts have similar embeddings
        alignment_scores = []

        # Sample pairs for efficiency
        n_samples = min(100, len(texts))
        indices = np.random.choice(len(texts), n_samples, replace=False)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]

                # Text similarity (simple word overlap)
                text_sim = _text_similarity(texts[idx1], texts[idx2])

                # Embedding similarity
                emb_sim = np.dot(embeddings_array[idx1], embeddings_array[idx2]) / (
                        np.linalg.norm(embeddings_array[idx1]) * np.linalg.norm(embeddings_array[idx2])
                )

                alignment_scores.append(abs(text_sim - emb_sim))

        alignment = 1.0 - np.mean(alignment_scores) if alignment_scores else 0.0

        # Uniformity: how uniformly embeddings are distributed
        norms = np.linalg.norm(embeddings_array, axis=1)
        normalized = embeddings_array / norms[:, np.newaxis]

        # Calculate pairwise distances
        distances = []
        for i in range(min(100, len(normalized))):
            for j in range(i + 1, min(100, len(normalized))):
                dist = np.linalg.norm(normalized[i] - normalized[j])
                distances.append(dist)

        # Uniformity is inversely related to variance in distances
        uniformity = 1.0 / (1.0 + np.var(distances)) if distances else 0.0

        return float(alignment), float(uniformity)

    except Exception as e:
        logger.warning(f"Error calculating alignment/uniformity: {e}")
        return 0.0, 0.0


class EmbeddingComparator:
    """Enhanced embedding model comparison for RAG pipeline with multiple providers."""

    def __init__(self):
        self.embedding_models = {}
        self.comparison_results = {}
        self.performance_benchmarks = {}

    def initialize_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize multiple embedding models for comprehensive comparison."""
        results = {}

        for config in model_configs:
            model_name = config["name"]
            model_type = config["type"]

            try:
                if model_type == "ollama":
                    model = OllamaEmbeddings(model=config["model_path"])
                elif model_type == "sentence_transformers":
                    if not SENTENCE_TRANSFORMERS_AVAILABLE:
                        logger.error(f"sentence-transformers not available for model {model_name}")
                        results[model_name] = "failed: sentence-transformers not installed"
                        continue
                    model = SentenceTransformer(config["model_path"])
                elif model_type == "openai":
                    if not OPENAI_AVAILABLE:
                        logger.error(f"OpenAI not available for model {model_name}")
                        results[model_name] = "failed: openai not installed"
                        continue
                    import openai
                    openai.api_key = config.get("api_key")
                    model = OpenAIEmbeddingWrapper(config["model_path"])
                elif model_type == "cohere":
                    if not COHERE_AVAILABLE:
                        logger.error(f"Cohere not available for model {model_name}")
                        results[model_name] = "failed: cohere not installed"
                        continue
                    model = CohereEmbeddingWrapper(config.get("api_key"), config["model_path"])
                else:
                    logger.error(f"Unsupported model type: {model_type}")
                    continue

                self.embedding_models[model_name] = {
                    "model": model,
                    "type": model_type,
                    "config": config
                }

                results[model_name] = "initialized"
                logger.info(f"Initialized {model_name} ({model_type})")

            except Exception as e:
                logger.error(f"Failed to initialize {model_name}: {e}")
                results[model_name] = f"failed: {str(e)}"

        return results

    def embed_texts(self, texts: List[str], model_name: str) -> Tuple[List[List[float]], Dict[str, Any]]:
        """Embed texts using specified model with comprehensive metrics."""
        if model_name not in self.embedding_models:
            raise ValueError(f"Model {model_name} not initialized")

        model_info = self.embedding_models[model_name]
        model = model_info["model"]
        model_type = model_info["type"]

        start_time = time.time()
        memory_before = _get_memory_usage()

        try:
            if model_type == "ollama":
                embeddings = model.embed_documents(texts)
            elif model_type == "sentence_transformers":
                embeddings = model.encode(texts, show_progress_bar=True).tolist()
            elif model_type == "openai":
                embeddings = model.embed_documents(texts)
            elif model_type == "cohere":
                embeddings = model.embed_documents(texts)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            end_time = time.time()
            memory_after = _get_memory_usage()

            # Calculate comprehensive metrics
            metrics = {
                "embedding_time": end_time - start_time,
                "texts_processed": len(texts),
                "avg_time_per_text": (end_time - start_time) / len(texts),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "total_tokens": sum(len(text.split()) for text in texts),
                "avg_tokens_per_text": sum(len(text.split()) for text in texts) / len(texts),
                "memory_usage_mb": memory_after - memory_before,
                "throughput_texts_per_second": len(texts) / (end_time - start_time),
                "throughput_tokens_per_second": sum(len(text.split()) for text in texts) / (end_time - start_time),
                "model_type": model_type,
                "timestamp": time.time()
            }

            logger.info(f"Embedded {len(texts)} texts with {model_name} in {metrics['embedding_time']:.2f}s")

            return embeddings, metrics

        except Exception as e:
            logger.error(f"Error embedding texts with {model_name}: {e}")
            raise

    def compare_embedding_quality(
            self,
            texts: List[str],
            query_text_pairs: List[Tuple[str, str]] = None,
            benchmark_tasks: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Comprehensive embedding quality comparison across different models."""
        comparison_results = {}

        for model_name in self.embedding_models.keys():
            try:
                # Basic embedding and performance metrics
                embeddings, performance_metrics = self.embed_texts(texts, model_name)

                # Calculate intrinsic quality metrics
                quality_metrics = self._calculate_enhanced_embedding_quality(embeddings, texts)

                # Semantic similarity evaluation if query-text pairs provided
                similarity_metrics = {}
                if query_text_pairs:
                    similarity_metrics = self._evaluate_semantic_similarity(
                        query_text_pairs, model_name
                    )

                # Benchmark evaluation on standard tasks
                benchmark_metrics = {}
                if benchmark_tasks:
                    benchmark_metrics = self._run_benchmark_tasks(
                        texts, model_name, benchmark_tasks
                    )

                # Clustering evaluation
                clustering_metrics = _evaluate_clustering_quality(embeddings, texts)

                # Retrieval evaluation
                retrieval_metrics = self._evaluate_retrieval_quality(embeddings, texts, model_name)

                comparison_results[model_name] = {
                    "performance_metrics": performance_metrics,
                    "quality_metrics": quality_metrics,
                    "similarity_metrics": similarity_metrics,
                    "benchmark_metrics": benchmark_metrics,
                    "clustering_metrics": clustering_metrics,
                    "retrieval_metrics": retrieval_metrics,
                    "model_config": self.embedding_models[model_name]["config"],
                    "overall_score": _calculate_overall_score(
                        quality_metrics, similarity_metrics, clustering_metrics, retrieval_metrics
                    )
                }

            except Exception as e:
                logger.error(f"Error in quality comparison for {model_name}: {e}")
                comparison_results[model_name] = {"error": str(e)}

        self.comparison_results = comparison_results
        return comparison_results

    def _calculate_enhanced_embedding_quality(self, embeddings: List[List[float]], texts: List[str]) -> Dict[
        str, float]:
        """Calculate comprehensive intrinsic quality metrics for embeddings."""
        if not embeddings:
            return {}

        embeddings_array = np.array(embeddings)
        metrics = {}

        # Basic dimensionality metrics
        metrics["dimension"] = embeddings_array.shape[1]
        metrics["num_embeddings"] = embeddings_array.shape[0]

        # Vector norm statistics
        norms = np.linalg.norm(embeddings_array, axis=1)
        metrics["avg_norm"] = float(np.mean(norms))
        metrics["norm_std"] = float(np.std(norms))
        metrics["norm_variance"] = float(np.var(norms))
        metrics["min_norm"] = float(np.min(norms))
        metrics["max_norm"] = float(np.max(norms))

        # Cosine similarity statistics
        similarities = self._compute_pairwise_similarities(embeddings_array)
        if similarities:
            metrics["avg_cosine_similarity"] = float(np.mean(similarities))
            metrics["similarity_std"] = float(np.std(similarities))
            metrics["min_similarity"] = float(np.min(similarities))
            metrics["max_similarity"] = float(np.max(similarities))
            metrics["similarity_median"] = float(np.median(similarities))

        # Embedding space distribution analysis
        metrics["embedding_variance"] = float(np.mean(np.var(embeddings_array, axis=0)))
        metrics["embedding_std"] = float(np.mean(np.std(embeddings_array, axis=0)))

        # Density and sparsity analysis
        metrics["sparsity_ratio"] = float(np.mean(embeddings_array == 0))
        metrics["effective_dimension"] = _calculate_effective_dimension(embeddings_array)

        # Isotropy (uniformity of distribution)
        metrics["isotropy_score"] = _calculate_isotropy(embeddings_array)

        # Alignment and uniformity metrics
        alignment, uniformity = _calculate_alignment_uniformity(embeddings_array, texts)
        metrics["alignment_score"] = alignment
        metrics["uniformity_score"] = uniformity

        return metrics

    def _compute_pairwise_similarities(self, embeddings_array: np.ndarray, sample_size: int = 1000) -> List[float]:
        """Compute pairwise cosine similarities with sampling for efficiency."""
        n = len(embeddings_array)
        if n <= 1:
            return []

        # Sample pairs for large datasets
        if n > sample_size:
            indices = np.random.choice(n, min(sample_size, n), replace=False)
            sampled_embeddings = embeddings_array[indices]
        else:
            sampled_embeddings = embeddings_array

        similarities = []
        norms = np.linalg.norm(sampled_embeddings, axis=1)

        for i in range(len(sampled_embeddings)):
            for j in range(i + 1, len(sampled_embeddings)):
                if norms[i] > 0 and norms[j] > 0:
                    sim = np.dot(sampled_embeddings[i], sampled_embeddings[j]) / (norms[i] * norms[j])
                    similarities.append(sim)

        return similarities

    def _evaluate_retrieval_quality(self, embeddings: List[List[float]], texts: List[str], model_name: str) -> dict[
                                                                                                                   str, str] | \
                                                                                                               dict[
                                                                                                                   str, float | int]:
        """Evaluate retrieval quality using synthetic queries."""
        try:
            embeddings_array = np.array(embeddings)

            if len(embeddings_array) < 5:
                return {"error": "Not enough samples for retrieval evaluation"}

            # Generate synthetic queries from texts
            queries = []
            relevant_docs = []

            for i, text in enumerate(texts[:min(20, len(texts))]):  # Limit for efficiency
                # Create query by taking first sentence or first few words
                words = text.split()[:10]
                if len(words) >= 3:
                    query = " ".join(words[:5])  # Use first 5 words as query
                    queries.append(query)
                    relevant_docs.append([i])  # Document i is relevant to query i

            if not queries:
                return {"error": "No valid queries generated"}

            # Evaluate retrieval performance
            precision_scores = []
            recall_scores = []

            for query, relevant in zip(queries, relevant_docs):
                # Get query embedding
                query_emb, _ = self.embed_texts([query], model_name)
                query_vector = np.array(query_emb[0])

                # Calculate similarities to all documents
                similarities = []
                for doc_emb in embeddings_array:
                    sim = np.dot(query_vector, doc_emb) / (
                            np.linalg.norm(query_vector) * np.linalg.norm(doc_emb)
                    )
                    similarities.append(sim)

                # Get top-k retrieved documents
                k = 5
                top_indices = np.argsort(similarities)[-k:][::-1]

                # Calculate precision and recall
                retrieved_relevant = len(set(top_indices) & set(relevant))
                precision = retrieved_relevant / k
                recall = retrieved_relevant / len(relevant)

                precision_scores.append(precision)
                recall_scores.append(recall)

            return {
                "avg_precision_at_5": float(np.mean(precision_scores)),
                "avg_recall_at_5": float(np.mean(recall_scores)),
                "queries_evaluated": len(queries)
            }

        except Exception as e:
            logger.warning(f"Error in retrieval evaluation: {e}")
            return {"error": str(e)}

    def _run_benchmark_tasks(self, texts: List[str], model_name: str, benchmark_tasks: List[str]) -> Dict[str, Any]:
        """Run standard benchmark tasks for embedding evaluation."""
        benchmark_results = {}

        for task in benchmark_tasks:
            try:
                if task == "text_classification":
                    result = self._benchmark_text_classification(texts, model_name)
                elif task == "semantic_similarity":
                    result = self._benchmark_semantic_similarity(texts, model_name)
                elif task == "clustering":
                    result = self._benchmark_clustering(texts, model_name)
                else:
                    result = {"error": f"Unknown benchmark task: {task}"}

                benchmark_results[task] = result

            except Exception as e:
                benchmark_results[task] = {"error": str(e)}

        return benchmark_results

    def _benchmark_text_classification(self, texts: List[str], model_name: str) -> dict[str, str] | dict[str, float]:
        """Benchmark text classification performance."""
        # This is a simplified benchmark - in practice you'd use labeled data
        try:
            embeddings, _ = self.embed_texts(texts, model_name)

            # Simple heuristic: classify based on text length
            labels = ["short" if len(text.split()) < 50 else "long" for text in texts]

            # Use a simple classifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            if len(set(labels)) < 2:
                return {"error": "Not enough label diversity"}

            classifier = LogisticRegression(random_state=42)
            scores = cross_val_score(classifier, embeddings, labels, cv=min(5, len(texts)))

            return {
                "accuracy": float(np.mean(scores)),
                "std": float(np.std(scores))
            }

        except Exception as e:
            return {"error": str(e)}

    def _benchmark_semantic_similarity(self, texts: List[str], model_name: str) -> dict[str, str] | dict[
        str, float | int]:
        """Benchmark semantic similarity performance."""
        try:
            # Create pairs with known similarity relationships
            pairs = []
            similarities = []

            for i in range(min(10, len(texts))):
                for j in range(i + 1, min(10, len(texts))):
                    pairs.append((texts[i], texts[j]))
                    # Simple similarity metric based on word overlap
                    sim = _text_similarity(texts[i], texts[j])
                    similarities.append(sim)

            if not pairs:
                return {"error": "No pairs generated"}

            # Get embeddings for all texts in pairs
            all_texts = list(set([text for pair in pairs for text in pair]))
            embeddings, _ = self.embed_texts(all_texts, model_name)
            text_to_emb = dict(zip(all_texts, embeddings))

            # Calculate embedding similarities
            emb_similarities = []
            for text1, text2 in pairs:
                emb1 = np.array(text_to_emb[text1])
                emb2 = np.array(text_to_emb[text2])

                cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                emb_similarities.append(cos_sim)

            # Calculate correlation
            correlation = np.corrcoef(similarities, emb_similarities)[0, 1]

            return {
                "similarity_correlation": float(correlation),
                "pairs_evaluated": len(pairs)
            }

        except Exception as e:
            return {"error": str(e)}

    def _benchmark_clustering(self, texts: List[str], model_name: str) -> Dict[str, float]:
        """Benchmark clustering performance."""
        embeddings, _ = self.embed_texts(texts, model_name)
        return _evaluate_clustering_quality(embeddings, texts)

    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report with recommendations."""
        if not self.comparison_results:
            return "No comparison results available. Run compare_embedding_quality first."

        report = ["=== COMPREHENSIVE EMBEDDING MODEL COMPARISON REPORT ===\n"]

        # Summary statistics
        valid_results = {k: v for k, v in self.comparison_results.items() if "error" not in v}

        if not valid_results:
            report.append("No valid results to compare.")
            return "\n".join(report)

        # Performance summary
        report.append("## PERFORMANCE SUMMARY")
        for model_name, results in valid_results.items():
            perf = results.get("performance_metrics", {})
            report.append(f"\n**{model_name}**:")
            report.append(f"  - Embedding time: {perf.get('embedding_time', 0):.3f}s")
            report.append(f"  - Throughput: {perf.get('throughput_texts_per_second', 0):.1f} texts/sec")
            report.append(f"  - Memory usage: {perf.get('memory_usage_mb', 0):.1f} MB")
            report.append(f"  - Dimension: {perf.get('embedding_dimension', 0)}")

        # Quality comparison
        report.append("\n## QUALITY METRICS COMPARISON")
        quality_metrics = ["avg_cosine_similarity", "isotropy_score", "alignment_score", "uniformity_score"]

        for metric in quality_metrics:
            report.append(f"\n### {metric.replace('_', ' ').title()}:")
            metric_values = []
            for model_name, results in valid_results.items():
                value = results.get("quality_metrics", {}).get(metric, 0)
                metric_values.append((model_name, value))
                report.append(f"  - {model_name}: {value:.3f}")

            # Highlight best performer
            if metric_values:
                best_model, best_value = max(metric_values, key=lambda x: x[1])
                report.append(f"  🏆 **Best**: {best_model} ({best_value:.3f})")

        # Overall rankings
        report.append("\n## OVERALL RANKINGS")
        rankings = []
        for model_name, results in valid_results.items():
            overall_score = results.get("overall_score", 0)
            rankings.append((model_name, overall_score))

        rankings.sort(key=lambda x: x[1], reverse=True)

        for i, (model_name, score) in enumerate(rankings, 1):
            report.append(f"{i}. **{model_name}**: {score:.3f}")

        # Recommendations
        report.append("\n## RECOMMENDATIONS")

        if rankings:
            best_model = rankings[0][0]
            report.append(f"🏆 **Best Overall Model**: {best_model}")

            best_result = valid_results[best_model]
            reasons = []

            # Add specific reasons
            if best_result.get("quality_metrics", {}).get("isotropy_score", 0) > 0.7:
                reasons.append("excellent isotropy (uniform distribution)")

            if best_result.get("performance_metrics", {}).get("throughput_texts_per_second", 0) > 10:
                reasons.append("high throughput")

            if best_result.get("clustering_metrics", {}).get("best_silhouette_score", -1) > 0.5:
                reasons.append("good clustering performance")

            if reasons:
                report.append(f"   Reasons: {', '.join(reasons)}")

        # Use case specific recommendations
        report.append("\n### Use Case Specific Recommendations:")

        # Best for speed
        fastest_model = min(valid_results.items(),
                            key=lambda x: x[1].get("performance_metrics", {}).get("embedding_time", float('inf')))
        report.append(f"- **For Speed**: {fastest_model[0]} "
                      f"({fastest_model[1].get('performance_metrics', {}).get('embedding_time', 0):.3f}s)")

        # Best for quality
        highest_quality = max(valid_results.items(),
                              key=lambda x: x[1].get("overall_score", 0))
        report.append(f"- **For Quality**: {highest_quality[0]} "
                      f"(score: {highest_quality[1].get('overall_score', 0):.3f})")

        # Best for memory efficiency
        most_efficient = min(valid_results.items(),
                             key=lambda x: x[1].get("performance_metrics", {}).get("memory_usage_mb", float('inf')))
        report.append(f"- **For Memory Efficiency**: {most_efficient[0]} "
                      f"({most_efficient[1].get('performance_metrics', {}).get('memory_usage_mb', 0):.1f} MB)")

        return "\n".join(report)

    def get_recommended_model(self) -> Tuple[str, str]:
        """Get recommended model based on comprehensive evaluation."""
        if not self.comparison_results:
            return "", "No comparison results available"

        valid_results = {k: v for k, v in self.comparison_results.items() if "error" not in v}

        if not valid_results:
            return "", "No valid models found"

        # Find best scoring model
        best_model = max(valid_results.items(), key=lambda x: x[1].get("overall_score", 0))

        model_name = best_model[0]
        model_results = best_model[1]

        recommendation_parts = []

        # Performance characteristics
        perf = model_results.get("performance_metrics", {})
        if perf.get("throughput_texts_per_second", 0) > 10:
            recommendation_parts.append("high throughput")

        # Quality characteristics
        quality = model_results.get("quality_metrics", {})
        if quality.get("isotropy_score", 0) > 0.7:
            recommendation_parts.append("excellent isotropy")
        if quality.get("alignment_score", 0) > 0.7:
            recommendation_parts.append("good semantic alignment")

        # Overall score
        overall_score = model_results.get("overall_score", 0)
        recommendation_parts.append(f"overall score: {overall_score:.3f}")

        recommendation = f"Recommended based on: {', '.join(recommendation_parts)}"

        return model_name, recommendation

    def _evaluate_semantic_similarity(
            self,
            query_text_pairs: List[Tuple[str, str]],
            model_name: str
    ) -> Dict[str, float]:
        """Evaluate semantic similarity performance using query-text pairs."""
        if not query_text_pairs:
            return {}

        model_info = self.embedding_models[model_name]
        model = model_info["model"]
        model_type = model_info["type"]

        similarities = []

        for query, text in query_text_pairs:
            try:
                # Get embeddings for query and text
                if model_type == "ollama":
                    query_emb = model.embed_query(query)
                    text_emb = model.embed_documents([text])[0]
                elif model_type == "sentence_transformers":
                    query_emb = model.encode([query])[0]
                    text_emb = model.encode([text])[0]
                elif model_type == "openai":
                    query_emb = model.embed_documents([query])[0]
                    text_emb = model.embed_documents([text])[0]
                elif model_type == "cohere":
                    query_emb = model.embed_documents([query])[0]
                    text_emb = model.embed_documents([text])[0]
                else:
                    continue

                # Calculate cosine similarity
                query_emb = np.array(query_emb)
                text_emb = np.array(text_emb)

                similarity = np.dot(query_emb, text_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(text_emb)
                )
                similarities.append(similarity)

            except Exception as e:
                logger.error(f"Error calculating similarity for pair: {e}")
                continue

        if similarities:
            return {
                "avg_similarity": float(np.mean(similarities)),
                "similarity_std": float(np.std(similarities)),
                "min_similarity": float(np.min(similarities)),
                "max_similarity": float(np.max(similarities)),
                "pairs_evaluated": len(similarities)
            }

        return {}

# Wrapper classes for different embedding providers

class OpenAIEmbeddingWrapper:
    """Wrapper for OpenAI embeddings to match interface."""

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using OpenAI API."""
        import openai

        response = openai.Embedding.create(
            model=self.model_name,
            input=texts
        )

        return [item['embedding'] for item in response['data']]


class CohereEmbeddingWrapper:
    """Wrapper for Cohere embeddings to match interface."""

    def __init__(self, api_key: str, model_name: str = "embed-english-v2.0"):
        import cohere
        self.client = cohere.Client(api_key)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Cohere API."""
        response = self.client.embed(
            texts=texts,
            model=self.model_name
        )

        return response.embeddings


# Enhanced default embedding model configurations
DEFAULT_EMBEDDING_CONFIGS = [
    {
        "name": "nomic_embed",
        "type": "ollama",
        "model_path": "nomic-embed-text",
        "description": "Ollama Nomic embedding model - local, fast"
    }
]

# Add sentence-transformers models if available
if SENTENCE_TRANSFORMERS_AVAILABLE:
    DEFAULT_EMBEDDING_CONFIGS.extend([
        {
            "name": "sentence_bert_mini",
            "type": "sentence_transformers",
            "model_path": "all-MiniLM-L6-v2",
            "description": "Lightweight sentence transformer - 384 dimensions"
        },
        {
            "name": "sentence_bert_base",
            "type": "sentence_transformers",
            "model_path": "all-mpnet-base-v2",
            "description": "High-quality sentence transformer - 768 dimensions"
        },
        {
            "name": "sentence_bert_large",
            "type": "sentence_transformers",
            "model_path": "all-roberta-large-v1",
            "description": "Large sentence transformer - 1024 dimensions"
        }
    ])

# Add OpenAI models if available (requires API key)
if OPENAI_AVAILABLE:
    DEFAULT_EMBEDDING_CONFIGS.extend([
        {
            "name": "openai_ada",
            "type": "openai",
            "model_path": "text-embedding-ada-002",
            "description": "OpenAI Ada embedding model - 1536 dimensions",
            "api_key": None  # Should be set by user
        }
    ])

# Add Cohere models if available (requires API key)
if COHERE_AVAILABLE:
    DEFAULT_EMBEDDING_CONFIGS.extend([
        {
            "name": "cohere_english",
            "type": "cohere",
            "model_path": "embed-english-v2.0",
            "description": "Cohere English embedding model - 4096 dimensions",
            "api_key": None  # Should be set by user
        }
    ])