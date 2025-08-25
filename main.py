import gradio as gr
import time
from typing import List, Tuple, Optional, Any, Dict
from config import (
    APP_TITLE, DEFAULT_ZOOM, MIN_ZOOM, MAX_ZOOM, ZOOM_STEP,
    PDF_VIEWER_HEIGHT, CHAT_HEIGHT, logger, CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS, MAX_CONTEXT_LENGTH
)
from pdf_utils import process_uploaded_pdfs
from vector_utils import vector_manager
from llm_utils import llm_manager, extract_model_names
from chunking_strategies import ChunkingManager
from embedding_comparison import EmbeddingComparator, DEFAULT_EMBEDDING_CONFIGS
from retrieval_evaluation import RetrievalEvaluator
from reranking import ReRankingManager


class PDFRAGApp:
    """PDF RAG Gradio application with comprehensive evaluation and production features."""

    def __init__(self):
        self.chat_history = []
        self.pdf_images = []
        self.current_zoom = DEFAULT_ZOOM
        self.chunking_manager = ChunkingManager()
        self.embedding_comparator = EmbeddingComparator()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.reranking_manager = ReRankingManager()
        self.current_chunks = []
        self.evaluation_results = {}
        self.processing_stats = {}
        self.vector_db_type = "faiss"  # Default vector database
        self.reranking_initialized = False  # Track re-ranking initialization

    def upload_and_process_pdfs(
            self,
            files: List[Any],
            chunking_strategy: str,
            chunk_size: int,
            chunk_overlap: int,
            enable_reranking: bool,
            vector_db_selection: str,
            progress=gr.Progress()
    ) -> Tuple[str, gr.Gallery, gr.Dropdown, str, str]:
        """Enhanced PDF processing with configurable parameters and comprehensive analysis."""
        if not files:
            return "Please upload PDF files first.", gr.Gallery(visible=False), gr.Dropdown(), "", ""

        try:
            start_time = time.time()
            progress(0.1, desc="Initializing processing...")

            # Switch vector database if needed
            if vector_db_selection != self.vector_db_type:
                vector_manager.switch_vector_db(vector_db_selection)
                self.vector_db_type = vector_db_selection

            progress(0.2, desc="Extracting text from PDFs...")

            # Process PDFs with enhanced chunking
            chunks, self.pdf_images = process_uploaded_pdfs(
                [f.name for f in files],
                chunking_strategy=chunking_strategy
            )

            if not chunks:
                return "No text found in uploaded PDFs.", gr.Gallery(visible=False), gr.Dropdown(), "", ""

            self.current_chunks = chunks

            progress(0.4, desc="Preparing chunk metadata...")

            # Prepare enhanced metadata for vector store
            chunk_texts = []
            chunk_metadata = []

            for chunk in chunks:
                chunk_texts.append(chunk["content"])

                # Enhanced metadata with processing info
                metadata = {
                    **chunk["metadata"],
                    "processing_timestamp": time.time(),
                    "chunking_strategy": chunking_strategy,
                    "chunk_size_config": chunk_size,
                    "chunk_overlap_config": chunk_overlap,
                    "vector_db_type": vector_db_selection
                }
                chunk_metadata.append(metadata)

            progress(0.6, desc=f"Creating {vector_db_selection} vector store...")

            # Create vector store with enhanced metadata
            vector_store = vector_manager.create_vector_store(chunk_texts, chunk_metadata)

            if vector_store is None:
                return "Failed to create vector store.", gr.Gallery(visible=False), gr.Dropdown(), "", ""

            progress(0.8, desc="Initializing re-ranking..." if enable_reranking else "Finalizing...")

            # Initialize re-ranking if enabled
            self.reranking_initialized = False
            if enable_reranking:
                rerank_success = self.reranking_manager.initialize_cross_encoder()
                if rerank_success:
                    self.reranking_initialized = True
                    logger.info("Re-ranking successfully initialized")
                else:
                    logger.warning("Re-ranking initialization failed, continuing without re-ranking")

            processing_time = time.time() - start_time

            # Store processing statistics
            self.processing_stats = {
                "total_files": len(files),
                "total_chunks": len(chunks),
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "vector_db_type": vector_db_selection,
                "processing_time": processing_time,
                "reranking_enabled": enable_reranking,
                "reranking_initialized": self.reranking_initialized,
                "timestamp": time.time()
            }

            progress(1.0, desc="Processing complete!")

            # Update PDF gallery
            gallery_update = gr.Gallery(
                value=self.pdf_images,
                visible=True,
                height=PDF_VIEWER_HEIGHT,
                object_fit="contain"
            )

            # Generate comprehensive analysis
            chunking_analysis = self._generate_comprehensive_analysis(chunks, chunking_strategy)

            # Generate processing summary
            processing_summary = self._generate_processing_summary()

            reranking_status = "with re-ranking" if self.reranking_initialized else "without re-ranking"
            success_message = (
                f"✅ Successfully processed {len(files)} PDF(s) with {len(chunks)} chunks "
                f"using {chunking_strategy} strategy and {vector_db_selection} vector database "
                f"({reranking_status}). "
                f"Processing time: {processing_time:.2f}s"
            )

            return (
                success_message,
                gallery_update,
                gr.Dropdown(),  # Clear file upload
                chunking_analysis,
                processing_summary
            )

        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            error_message = f"⚠️ Error processing PDFs: {str(e)}"
            return error_message, gr.Gallery(visible=False), gr.Dropdown(), "", ""

    def run_embedding_comparison(self, comparison_depth: str) -> str:
        """Run comprehensive embedding model comparison with configurable depth."""
        try:
            if not self.current_chunks:
                return "Please upload and process PDFs first."

            # Configure comparison based on depth
            if comparison_depth == "basic":
                configs = DEFAULT_EMBEDDING_CONFIGS[:2]  # Limit to 2 models
                text_samples = [chunk["content"] for chunk in self.current_chunks[:10]]
            elif comparison_depth == "standard":
                configs = DEFAULT_EMBEDDING_CONFIGS[:3]  # Limit to 3 models
                text_samples = [chunk["content"] for chunk in self.current_chunks[:20]]
            else:  # comprehensive
                configs = DEFAULT_EMBEDDING_CONFIGS
                text_samples = [chunk["content"] for chunk in self.current_chunks[:50]]

            # Initialize embedding models
            init_results = self.embedding_comparator.initialize_models(configs)

            logger.info(f"Running {comparison_depth} embedding comparison with {len(text_samples)} text samples")

            # Create query-text pairs for semantic evaluation
            query_text_pairs = []
            for i, chunk in enumerate(self.current_chunks[:min(5, len(self.current_chunks))]):
                # Create synthetic query from chunk content
                words = chunk["content"].split()[:10]
                if len(words) >= 5:
                    query = " ".join(words[:5]) + "?"
                    query_text_pairs.append((query, chunk["content"]))

            # Run comprehensive comparison
            comparison_results = self.embedding_comparator.compare_embedding_quality(
                text_samples,
                query_text_pairs=query_text_pairs,
                benchmark_tasks=["semantic_similarity", "clustering"] if comparison_depth != "basic" else []
            )

            # Generate enhanced report
            report = self.embedding_comparator.generate_comparison_report()

            # Add recommendation
            recommended_model, recommendation_reason = self.embedding_comparator.get_recommended_model()
            if recommended_model:
                report += f"\n\n## FINAL RECOMMENDATION\n"
                report += f"**Recommended Model**: {recommended_model}\n"
                report += f"**Reason**: {recommendation_reason}\n"

            return report

        except Exception as e:
            logger.error(f"Error in embedding comparison: {e}")
            return f"Error in embedding comparison: {str(e)}"

    def run_retrieval_evaluation(self, evaluation_depth: str) -> str:
        """Run comprehensive retrieval evaluation with configurable depth."""
        try:
            if not self.current_chunks:
                return "Please upload and process PDFs first."

            vector_store = vector_manager.get_vector_store()
            if not vector_store:
                return "No vector store available. Please process PDFs first."

            # Configure evaluation based on depth
            if evaluation_depth == "basic":
                num_easy, num_medium, num_hard = 5, 3, 2
                k_values = [1, 3, 5]
            elif evaluation_depth == "standard":
                num_easy, num_medium, num_hard = 10, 5, 5
                k_values = [1, 3, 5, 10]
            else:  # comprehensive
                num_easy, num_medium, num_hard = 15, 10, 5
                k_values = [1, 3, 5, 10, 20]

            logger.info(f"Running {evaluation_depth} retrieval evaluation")

            # Create enhanced test dataset
            test_dataset = self.retrieval_evaluator.create_test_dataset(
                self.current_chunks,
                num_easy=num_easy,
                num_medium=num_medium,
                num_hard=num_hard
            )
            self.retrieval_evaluator.load_ground_truth(test_dataset)

            # Simulate retrieval results using actual vector store
            retrieval_results = {}
            for query in test_dataset.keys():
                try:
                    # Use the actual vector store for retrieval
                    similar_docs = vector_manager.search_similar_documents(query, k=max(k_values))

                    # Extract chunk IDs from results
                    doc_ids = []
                    for doc in similar_docs:
                        if hasattr(doc, 'metadata') and 'chunk_id' in doc.metadata:
                            doc_ids.append(str(doc.metadata['chunk_id']))

                    retrieval_results[query] = doc_ids

                except Exception as e:
                    logger.warning(f"Error retrieving for query '{query}': {e}")
                    retrieval_results[query] = []

            # Run comprehensive evaluation
            eval_results = self.retrieval_evaluator.evaluate_retrieval(retrieval_results, k_values)

            # Analyze failure cases
            failure_analysis = self.retrieval_evaluator.analyze_failure_cases(retrieval_results)

            # Generate comprehensive report
            report = self.retrieval_evaluator.generate_evaluation_report()

            # Add failure case analysis
            if failure_analysis["zero_recall_queries"]:
                report += f"\n\n## FAILURE CASE ANALYSIS\n"
                report += f"- Queries with zero recall: {len(failure_analysis['zero_recall_queries'])}\n"
                report += f"- Low precision queries: {len(failure_analysis['low_precision_queries'])}\n"

                # Show example failure cases
                if failure_analysis["zero_recall_queries"]:
                    report += f"\n**Example Zero Recall Query**:\n"
                    example = failure_analysis["zero_recall_queries"][0]
                    report += f"Query: {example['query']}\n"
                    report += f"Expected docs: {example['expected_docs']}\n"

            return report

        except Exception as e:
            logger.error(f"Error in retrieval evaluation: {e}")
            return f"Error in retrieval evaluation: {str(e)}"

    def run_reranking_comparison(self) -> str:
        """Run comprehensive re-ranking method comparison."""
        try:
            if not self.current_chunks:
                return "Please upload and process PDFs first."

            # Create a test query
            if len(self.current_chunks) < 5:
                return "Not enough chunks for re-ranking comparison."

            # Use the first chunk to create a test query
            test_chunk = self.current_chunks[0]
            words = test_chunk["content"].split()[:8]
            test_query = " ".join(words[:4]) + "?"

            # Get documents from vector store
            vector_store = vector_manager.get_vector_store()
            if not vector_store:
                return "No vector store available for re-ranking comparison."

            # Retrieve documents
            retrieved_docs = vector_manager.search_similar_documents(test_query, k=20)

            if len(retrieved_docs) < 5:
                return "Not enough retrieved documents for meaningful re-ranking comparison."

            # Convert to format expected by re-ranking manager
            documents = []
            for doc in retrieved_docs:
                doc_dict = {
                    "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    "metadata": getattr(doc, 'metadata', {})
                }
                documents.append(doc_dict)

            # Compare different re-ranking methods
            comparison_results = self.reranking_manager.compare_reranking_methods(
                test_query, documents
            )

            # Generate comprehensive report
            report = self.reranking_manager.generate_reranking_report(comparison_results)

            return report

        except Exception as e:
            logger.error(f"Error in re-ranking comparison: {e}")
            return f"Error in re-ranking comparison: {str(e)}"

    def chat_with_pdf(
            self,
            message: str,
            history: List[List[str]],
            selected_model: str,
            use_reranking: bool,
            enable_quality_analysis: bool
    ) -> Tuple[List[List[str]], str]:
        """Enhanced chat with comprehensive response analysis and proper re-ranking."""
        if not message.strip():
            return history, ""

        if not selected_model:
            history.append([message, "Please select a model first."])
            return history, ""

        vector_store = vector_manager.get_vector_store()
        if vector_store is None:
            history.append([message, "Please upload and process a PDF file first."])
            return history, ""

        try:
            start_time = time.time()

            # Determine if re-ranking should be used
            should_use_reranking = use_reranking and self.reranking_initialized

            if use_reranking and not self.reranking_initialized:
                logger.warning("Re-ranking requested but not initialized")

            # Pass re-ranking manager if it should be used
            reranking_manager = self.reranking_manager if should_use_reranking else None

            logger.info(f"Chat processing - Re-ranking requested: {use_reranking}, "
                        f"Re-ranking initialized: {self.reranking_initialized}, "
                        f"Using re-ranking: {should_use_reranking}")

            # Get enhanced response with proper re-ranking
            response_data = llm_manager.process_question(
                message,
                vector_store,
                selected_model,
                enable_quality_analysis,
                enable_reranking=should_use_reranking,
                reranking_manager=reranking_manager
            )

            # Extract response and metadata
            response = response_data["response"]
            processing_time = response_data["processing_time"]
            context_docs_count = response_data["context_docs_count"]
            reranking_applied = response_data.get("reranking_applied", False)

            # Add response metadata for transparency
            metadata_info = []
            metadata_info.append(f"Processing time: {processing_time:.2f}s")
            metadata_info.append(f"Documents used: {context_docs_count}")
            metadata_info.append(f"Model: {selected_model}")
            metadata_info.append(f"Re-ranking: {'Applied' if reranking_applied else 'Not applied'}")

            if enable_quality_analysis and "quality_analysis" in response_data:
                qa = response_data["quality_analysis"]
                metadata_info.append(f"Quality score: {qa.get('overall_score', 0):.2f}")

                # Add quality warnings if needed
                hallucination_risk = qa.get("hallucination_check", {}).get("risk_level", "unknown")
                if hallucination_risk == "high":
                    response += "\n\n⚠️ **Quality Warning**: This response may contain inaccuracies. Please verify with source documents."

            # Add validation info if available
            if "validation" in response_data:
                validation = response_data["validation"]
                verified = validation.get("claims_verified", 0)
                unverified = validation.get("claims_unverified", 0)
                if unverified > 0:
                    metadata_info.append(f"Claims: {verified} verified, {unverified} unverified")

            # Add metadata as footnote
            response += f"\n\n---\n*{' | '.join(metadata_info)}*"

            history.append([message, response])
            self.chat_history = history
            return history, ""

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_response = f"Error processing your message: {str(e)}"
            history.append([message, error_response])
            return history, ""

    def get_system_diagnostics(self) -> str:
        """Generate comprehensive system diagnostics report."""
        try:
            diagnostics = []
            diagnostics.append("=== SYSTEM DIAGNOSTICS ===\n")

            # Processing statistics
            if self.processing_stats:
                diagnostics.append("## Processing Statistics")
                stats = self.processing_stats
                diagnostics.append(f"- Files processed: {stats.get('total_files', 0)}")
                diagnostics.append(f"- Chunks created: {stats.get('total_chunks', 0)}")
                diagnostics.append(f"- Chunking strategy: {stats.get('chunking_strategy', 'unknown')}")
                diagnostics.append(f"- Vector database: {stats.get('vector_db_type', 'unknown')}")
                diagnostics.append(f"- Processing time: {stats.get('processing_time', 0):.2f}s")
                diagnostics.append(f"- Re-ranking enabled: {stats.get('reranking_enabled', False)}")
                diagnostics.append(f"- Re-ranking initialized: {stats.get('reranking_initialized', False)}")
                diagnostics.append("")

            # Vector store performance
            vector_metrics = vector_manager.get_performance_metrics()
            if vector_metrics:
                diagnostics.append("## Vector Store Performance")
                diagnostics.append(f"- Creation time: {vector_metrics.get('creation_time', 0):.2f}s")
                diagnostics.append(f"- Indexing speed: {vector_metrics.get('indexing_speed', 0):.1f} docs/sec")
                diagnostics.append(f"- Last search time: {vector_metrics.get('last_search_time', 0):.3f}s")
                diagnostics.append("")

            # LLM performance
            llm_stats = llm_manager.get_model_performance_stats()
            if llm_stats:
                diagnostics.append("## LLM Performance")
                for model, stats in llm_stats.items():
                    diagnostics.append(f"**{model}**:")
                    diagnostics.append(f"  - Requests: {stats.get('count', 0)}")
                    diagnostics.append(f"  - Avg response time: {stats.get('avg_time', 0):.2f}s")
                    diagnostics.append(f"  - Error rate: {stats.get('error_rate', 0):.1%}")
                    diagnostics.append(f"  - Re-ranking usage: {stats.get('reranking_rate', 0):.1%}")
                diagnostics.append("")

            # Hallucination analysis
            hallucination_stats = llm_manager.analyze_hallucination_patterns()
            if hallucination_stats:
                diagnostics.append("## Response Quality Analysis")
                diagnostics.append(f"- Responses analyzed: {hallucination_stats.get('total_responses_analyzed', 0)}")
                diagnostics.append(f"- High risk responses: {hallucination_stats.get('high_risk_percentage', 0):.1f}%")
                diagnostics.append(
                    f"- Responses with attribution: {hallucination_stats.get('attribution_percentage', 0):.1f}%")
                diagnostics.append(
                    f"- Responses with re-ranking: {hallucination_stats.get('reranking_percentage', 0):.1f}%")
                diagnostics.append("")

            # Available capabilities
            diagnostics.append("## Available Capabilities")
            available_dbs = vector_manager.get_available_vector_dbs()
            diagnostics.append(f"- Vector databases: {', '.join(available_dbs)}")

            models = extract_model_names()
            diagnostics.append(f"- LLM models: {len(models)} available")

            diagnostics.append(
                f"- Re-ranking: {'✅ Initialized' if self.reranking_initialized else '❌ Not initialized'}")
            diagnostics.append(f"- Cross-encoder: {'✅' if self.reranking_manager.cross_encoder else '❌'}")
            diagnostics.append("")

            # Memory and resource usage
            diagnostics.append("## Resource Usage")
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            diagnostics.append(f"- Memory usage: {memory_mb:.1f} MB")
            diagnostics.append(f"- CPU usage: {cpu_percent:.1f}%")

            return "\n".join(diagnostics)

        except Exception as e:
            return f"Error generating diagnostics: {str(e)}"

    def delete_collection(self) -> Tuple[str, gr.Gallery, gr.Chatbot, str, str]:
        """Enhanced collection deletion with cleanup."""
        try:
            success = vector_manager.delete_vector_store()

            # Clear all application state
            self.chat_history = []
            self.pdf_images = []
            self.current_chunks = []
            self.evaluation_results = {}
            self.processing_stats = {}
            self.reranking_initialized = False

            # Clear LLM response history
            llm_manager.clear_response_history()

            if success:
                message = "✅ Collection and all temporary files deleted successfully. Application state reset."
            else:
                message = "⚠️ Error deleting collection, but application state has been reset."

            return (
                message,
                gr.Gallery(visible=False),
                gr.Chatbot(value=[]),
                "",  # Clear analysis
                ""  # Clear summary
            )

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return (
                f"⚠️ Error deleting collection: {str(e)}",
                gr.Gallery(visible=len(self.pdf_images) > 0, value=self.pdf_images),
                gr.Chatbot(value=self.chat_history),
                "",
                ""
            )

    def _generate_comprehensive_analysis(self, chunks: List[Dict[str, Any]], strategy: str) -> str:
        """Generate comprehensive chunking analysis with enhanced metrics."""
        quality_metrics = self.chunking_manager.get_chunking_quality_metrics(chunks)

        if not quality_metrics:
            return "No chunking metrics available."

        analysis = []
        analysis.append(f"=== COMPREHENSIVE CHUNKING ANALYSIS ({strategy.upper()}) ===\n")

        # Basic statistics
        analysis.append("## Basic Statistics")
        analysis.append(f"- Total chunks: {quality_metrics.get('total_chunks', 0)}")
        analysis.append(f"- Average chunk size: {quality_metrics.get('avg_chunk_size_chars', 0):.0f} characters")
        analysis.append(f"- Average tokens per chunk: {quality_metrics.get('avg_chunk_size_tokens', 0):.0f}")
        analysis.append(
            f"- Size range: {quality_metrics.get('min_chunk_size_chars', 0):.0f} - {quality_metrics.get('max_chunk_size_chars', 0):.0f} chars")
        analysis.append(
            f"- Token range: {quality_metrics.get('min_chunk_size_tokens', 0):.0f} - {quality_metrics.get('max_chunk_size_tokens', 0):.0f} tokens")
        analysis.append("")

        # Quality analysis
        analysis.append("## Quality Metrics")
        analysis.append(f"- Size variance: {quality_metrics.get('size_variance_chars', 0):.0f}")
        analysis.append(f"- Metadata coverage: {quality_metrics.get('metadata_coverage', 0):.1%}")
        analysis.append(
            f"- Special elements preserved: {quality_metrics.get('chunks_with_special_elements', 0)} chunks")
        analysis.append(f"- Average complexity: {quality_metrics.get('avg_complexity', 0):.3f}")
        analysis.append(f"- Average coherence: {quality_metrics.get('avg_coherence', 0):.3f}")
        analysis.append(f"- Structure preservation: {quality_metrics.get('structure_preservation_rate', 0):.1%}")
        analysis.append("")

        # Strategy-specific recommendations
        analysis.append("## Strategy Assessment & Recommendations")

        if strategy == "recursive":
            avg_tokens = quality_metrics.get('avg_chunk_size_tokens', 0)
            if avg_tokens > CHUNK_SIZE_TOKENS * 1.2:
                analysis.append("⚠️ Chunks larger than target size - consider reducing chunk_size parameter")
            elif avg_tokens < CHUNK_SIZE_TOKENS * 0.8:
                analysis.append("💡 Chunks smaller than target size - could increase chunk_size for more context")
            else:
                analysis.append("✅ Chunk sizes within target range")

            if quality_metrics.get('size_variance_chars', 0) > 100000:
                analysis.append("💡 High size variance - consider semantic chunking for more uniform chunks")

        elif strategy == "semantic":
            coherence = quality_metrics.get('avg_coherence', 0)
            if coherence > 0.3:
                analysis.append("✅ Good semantic coherence detected")
            else:
                analysis.append("⚠️ Low semantic coherence - may need better boundary detection")

        elif strategy == "hierarchical":
            structure_rate = quality_metrics.get('structure_preservation_rate', 0)
            if structure_rate > 0.8:
                analysis.append("✅ Excellent hierarchical structure preservation")
            else:
                analysis.append("💡 Limited hierarchical structure detected - document may lack clear sections")

        elif strategy == "custom":
            special_elements = quality_metrics.get('chunks_with_special_elements', 0)
            if special_elements > 0:
                analysis.append(f"✅ Successfully preserved special elements in {special_elements} chunks")
            else:
                analysis.append("💡 No special elements detected or preserved")

        # Performance recommendations
        analysis.append("")
        analysis.append("## Performance Recommendations")
        total_chunks = quality_metrics.get('total_chunks', 0)

        if total_chunks > 1000:
            analysis.append("⚠️ Large number of chunks may impact search performance - consider increasing chunk size")
        elif total_chunks < 10:
            analysis.append("⚠️ Very few chunks may limit retrieval precision - consider decreasing chunk size")

        avg_tokens = quality_metrics.get('avg_chunk_size_tokens', 0)
        if avg_tokens > MAX_CONTEXT_LENGTH // 4:
            analysis.append("⚠️ Large chunks may cause context length issues with some LLMs")

        return "\n".join(analysis)

    def _generate_processing_summary(self) -> str:
        """Generate processing summary with performance metrics."""
        if not self.processing_stats:
            return "No processing statistics available."

        summary = ["=== PROCESSING SUMMARY ===\n"]

        stats = self.processing_stats

        summary.append("## Files & Content")
        summary.append(f"- Files processed: {stats.get('total_files', 0)}")
        summary.append(f"- Total chunks created: {stats.get('total_chunks', 0)}")
        summary.append(
            f"- Average chunks per file: {stats.get('total_chunks', 0) / max(1, stats.get('total_files', 1)):.1f}")
        summary.append("")

        summary.append("## Configuration")
        summary.append(f"- Chunking strategy: {stats.get('chunking_strategy', 'unknown')}")
        summary.append(f"- Chunk size: {stats.get('chunk_size', 0)} tokens")
        summary.append(f"- Chunk overlap: {stats.get('chunk_overlap', 0)} tokens")
        summary.append(f"- Vector database: {stats.get('vector_db_type', 'unknown')}")
        summary.append(f"- Re-ranking enabled: {'Yes' if stats.get('reranking_enabled', False) else 'No'}")
        summary.append(f"- Re-ranking initialized: {'Yes' if stats.get('reranking_initialized', False) else 'No'}")
        summary.append("")

        summary.append("## Performance")
        summary.append(f"- Total processing time: {stats.get('processing_time', 0):.2f} seconds")

        files_per_sec = stats.get('total_files', 0) / max(0.1, stats.get('processing_time', 0.1))
        chunks_per_sec = stats.get('total_chunks', 0) / max(0.1, stats.get('processing_time', 0.1))

        summary.append(f"- Processing speed: {files_per_sec:.1f} files/sec, {chunks_per_sec:.1f} chunks/sec")

        # Add vector store metrics if available
        vector_metrics = vector_manager.get_performance_metrics()
        if vector_metrics:
            summary.append(f"- Vector indexing speed: {vector_metrics.get('indexing_speed', 0):.1f} docs/sec")

        summary.append("")
        summary.append(f"## Status")
        summary.append(f"- ✅ Ready for queries and evaluation")
        summary.append(f"- Vector store contains {stats.get('total_chunks', 0)} embedded chunks")

        if stats.get('reranking_initialized'):
            summary.append(f"- ✅ Re-ranking is active and will be applied to queries")
        elif stats.get('reranking_enabled'):
            summary.append(f"- ⚠️ Re-ranking was requested but failed to initialize")

        return "\n".join(summary)

    def update_zoom(self, zoom_level: int) -> gr.Gallery:
        """Update PDF gallery zoom level."""
        self.current_zoom = zoom_level
        if self.pdf_images:
            return gr.Gallery(
                value=self.pdf_images,
                visible=True,
                height=PDF_VIEWER_HEIGHT,
                object_fit="contain"
            )
        return gr.Gallery(visible=False)

    def create_interface(self) -> gr.Blocks:
        """Create comprehensive Gradio interface with all RAG pipeline features."""

        # Get available models and vector databases
        available_models = extract_model_names()
        available_vector_dbs = vector_manager.get_available_vector_dbs()

        with gr.Blocks(
                title=APP_TITLE,
                theme=gr.themes.Default(),  # neutral, no "soft" styling
                css="""
            .pdf-viewer { max-height: 500px; overflow-y: auto; }
            .chat-container { height: 500px; }
            .evaluation-panel { border: 1px solid #e5e7eb; padding: 12px; margin: 8px 0; border-radius: 6px; background:#fff; }
            .metrics-box { border: 1px solid #e5e7eb; padding: 10px; border-radius: 6px; background:#fff; }
            """
        ) as demo:
            gr.Markdown(f"# {APP_TITLE}")
            gr.Markdown("RAG pipeline with evaluation, multiple chunking strategies, and analytics.")

            with gr.Tabs():
                # Main RAG Pipeline Tab
                with gr.TabItem("RAG Pipeline"):
                    with gr.Row():
                        # Left column - Configuration and upload
                        with gr.Column(scale=1):
                            gr.Markdown("### PDF Upload & Configuration")

                            # File upload
                            pdf_upload = gr.File(
                                file_count="multiple",
                                file_types=[".pdf"],
                                label="Upload PDF files",
                                # height=100
                            )

                            # Advanced configuration
                            with gr.Accordion("Advanced Configuration", open=False):
                                # Chunking strategy selection
                                chunking_strategy = gr.Dropdown(
                                    choices=["recursive", "semantic", "hierarchical", "custom"],
                                    value="recursive",
                                    label="Chunking strategy",
                                    info="How to split documents into chunks."
                                )

                                # Chunk size configuration
                                chunk_size = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    value=CHUNK_SIZE_TOKENS,
                                    step=64,
                                    label="Chunk size (tokens)",
                                    info="Target size for each chunk."
                                )

                                chunk_overlap = gr.Slider(
                                    minimum=0,
                                    maximum=512,
                                    value=CHUNK_OVERLAP_TOKENS,
                                    step=32,
                                    label="Chunk overlap (tokens)",
                                    info="Overlap between consecutive chunks."
                                )

                                # Vector database selection
                                vector_db_selection = gr.Dropdown(
                                    choices=available_vector_dbs,
                                    value=available_vector_dbs[0] if available_vector_dbs else "faiss",
                                    label="Vector database",
                                    info="Choose vector database backend"
                                )

                                # Re-ranking option
                                enable_reranking = gr.Checkbox(
                                    label="Enable Cross-Encoder Re-ranking",
                                    value=False,
                                    info="Improve result quality with re-ranking (slower)"
                                )

                            # Control buttons
                            with gr.Row():
                                process_btn = gr.Button(
                                    "Process Documents",
                                    variant="primary",
                                    scale=2
                                )
                                delete_btn = gr.Button(
                                    "Clear All",
                                    variant="secondary",
                                    scale=1
                                )

                            # Status message
                            status_msg = gr.Textbox(
                                label="Status",
                                interactive=False,
                                lines=3,
                                elem_classes=["status-success"]
                            )

                            # Processing summary
                            with gr.Accordion("Processing Summary", open=False):
                                processing_summary = gr.Textbox(
                                    label="Processing Metrics",
                                    interactive=True,
                                    elem_classes=["metrics-box"]
                                )

                            # Chunking analysis
                            with gr.Accordion("Chunking Analysis", open=False):
                                chunking_analysis = gr.Textbox(
                                    label="Detailed Analysis",
                                    interactive=False,
                                    elem_classes=["evaluation-panel"]
                                )

                            # Zoom control for PDF viewer
                            zoom_slider = gr.Slider(
                                minimum=MIN_ZOOM,
                                maximum=MAX_ZOOM,
                                value=DEFAULT_ZOOM,
                                step=ZOOM_STEP,
                                label="PDF Zoom Level",
                                interactive=True
                            )

                            # PDF viewer
                            pdf_gallery = gr.Gallery(
                                label="PDF Preview",
                                visible=False,
                                height=PDF_VIEWER_HEIGHT,
                                object_fit="contain",
                                elem_classes=["pdf-viewer"]
                            )

                        # Right column - Chat interface
                        with gr.Column(scale=2):
                            gr.Markdown("### Chat with Documents")

                            # Model and chat settings
                            with gr.Row():
                                model_dropdown = gr.Dropdown(
                                    choices=list(available_models),
                                    label="Select LLM Model",
                                    value=(available_models[0] if available_models else None),
                                    info=(
                                        None if available_models else "No models found. Install Ollama and pull a model."),
                                    interactive=True,
                                    scale=2
                                )

                                with gr.Column(scale=1):
                                    use_reranking_chat = gr.Checkbox(
                                        label="Use Re-ranking",
                                        value=False,
                                        info="Apply re-ranking to search results"
                                    )

                                    enable_quality_analysis = gr.Checkbox(
                                        label="Quality Analysis",
                                        value=True,
                                        info="Analyze response quality"
                                    )

                            # Chat interface
                            chatbot = gr.Chatbot(
                                height=CHAT_HEIGHT,
                                elem_classes=["chat-container"],
                                show_copy_button=True,
                                bubble_full_width=False
                            )

                            # Chat input
                            with gr.Row():
                                chat_input = gr.Textbox(
                                    placeholder="Ask a question about your documents...",
                                    label="Your Message",
                                    lines=2,
                                    scale=4
                                )
                                chat_btn = gr.Button("📤 Send", variant="primary", scale=1)

                # Comprehensive Evaluation Tab
                with gr.TabItem("Evaluation & Analytics"):
                    gr.Markdown("### Comprehensive RAG Pipeline Evaluation")
                    gr.Markdown("Analyze and optimize your RAG pipeline with detailed metrics and comparisons.")

                    with gr.Row():
                        # Embedding comparison
                        with gr.Column():
                            gr.Markdown("#### Embedding Model Comparison")

                            embedding_depth = gr.Dropdown(
                                choices=["basic", "standard", "comprehensive"],
                                value="standard",
                                label="Comparison Depth",
                                info="Choose evaluation thoroughness"
                            )

                            embedding_btn = gr.Button("Compare Embeddings", variant="primary")
                            embedding_results = gr.Textbox(
                                label="Embedding Analysis Results",
                                lines=20,
                                interactive=False,
                                elem_classes=["evaluation-panel"]
                            )

                        # Retrieval evaluation
                        with gr.Column():
                            gr.Markdown("#### Retrieval Performance")

                            retrieval_depth = gr.Dropdown(
                                choices=["basic", "standard", "comprehensive"],
                                value="standard",
                                label="Evaluation Depth",
                                info="Choose evaluation comprehensiveness"
                            )

                            retrieval_btn = gr.Button("Evaluate Retrieval", variant="primary")
                            retrieval_results = gr.Textbox(
                                label="Retrieval Evaluation Results",
                                lines=20,
                                interactive=False,
                                elem_classes=["evaluation-panel"]
                            )

                    with gr.Row():
                        # Re-ranking comparison
                        with gr.Column():
                            gr.Markdown("#### Re-ranking Analysis")
                            reranking_btn = gr.Button("Analyze Re-ranking", variant="primary")
                            reranking_results = gr.Textbox(
                                label="Re-ranking Comparison Results",
                                lines=15,
                                interactive=False,
                                elem_classes=["evaluation-panel"]
                            )

                        # System diagnostics
                        with gr.Column():
                            gr.Markdown("#### System Diagnostics")
                            diagnostics_btn = gr.Button("Run Diagnostics", variant="primary")
                            diagnostics_results = gr.Textbox(
                                label="System Diagnostics Report",
                                lines=15,
                                interactive=False,
                                elem_classes=["evaluation-panel"]
                            )

            # Event handlers
            process_btn.click(
                fn=self.upload_and_process_pdfs,
                inputs=[pdf_upload, chunking_strategy, chunk_size, chunk_overlap,
                        enable_reranking, vector_db_selection],
                outputs=[status_msg, pdf_gallery, pdf_upload, chunking_analysis, processing_summary],
                show_progress="full"
            )

            delete_btn.click(
                fn=self.delete_collection,
                outputs=[status_msg, pdf_gallery, chatbot, chunking_analysis, processing_summary]
            )

            chat_btn.click(
                fn=self.chat_with_pdf,
                inputs=[chat_input, chatbot, model_dropdown, use_reranking_chat, enable_quality_analysis],
                outputs=[chatbot, chat_input]
            )

            chat_input.submit(
                fn=self.chat_with_pdf,
                inputs=[chat_input, chatbot, model_dropdown, use_reranking_chat, enable_quality_analysis],
                outputs=[chatbot, chat_input]
            )

            zoom_slider.change(
                fn=self.update_zoom,
                inputs=[zoom_slider],
                outputs=[pdf_gallery]
            )

            # Evaluation handlers
            embedding_btn.click(
                fn=self.run_embedding_comparison,
                inputs=[embedding_depth],
                outputs=[embedding_results]
            )

            retrieval_btn.click(
                fn=self.run_retrieval_evaluation,
                inputs=[retrieval_depth],
                outputs=[retrieval_results]
            )

            reranking_btn.click(
                fn=self.run_reranking_comparison,
                outputs=[reranking_results]
            )

            diagnostics_btn.click(
                fn=self.get_system_diagnostics,
                outputs=[diagnostics_results]
            )

        return demo


def main():
    """Main function to run the enhanced RAG application."""
    logger.info("Starting Enhanced PDF RAG Application...")

    try:
        app = PDFRAGApp()
        demo = app.create_interface()

        logger.info("Launching Gradio interface...")

        # Launch the interface with production settings
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            max_threads=10
        )

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise


if __name__ == "__main__":
    main()