import re
import tiktoken
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import logger, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, get_token_count_estimate

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None


class TokenizedTextSplitter:
    """Token-based text splitter for more accurate chunking."""

    def __init__(self, model_name="gpt-3.5-turbo"):
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            return get_token_count_estimate(text)

    def split_text_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text by token count."""
        if not self.encoding:
            # Fallback to character-based splitting
            char_chunk_size = chunk_size * 4  # Approximate
            char_overlap = overlap * 4
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=char_chunk_size,
                chunk_overlap=char_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            return splitter.split_text(text)

        # Token-based splitting
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start position with overlap
            start += chunk_size - overlap

        return chunks


class ChunkingManager:
    """Manages multiple chunking strategies for RAG pipeline with enhanced capabilities."""

    def __init__(self):
        self.nlp = None
        self.token_splitter = TokenizedTextSplitter()
        self._load_spacy_model()

    def _load_spacy_model(self):
        """Load spaCy model for semantic chunking."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not installed. Semantic chunking will use regex fallback.")
            self.nlp = None
            return

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        except Exception as e:
            logger.warning(f"Error loading spaCy model: {e}. Using regex fallback.")
            self.nlp = None

    def recursive_chunking(
            self,
            text: str,
            chunk_size: int = CHUNK_SIZE_TOKENS,
            chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
            use_tokens: bool = True,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """Token-aware recursive chunking with overlap."""
        try:
            if use_tokens:
                chunks = self.token_splitter.split_text_by_tokens(text, chunk_size, chunk_overlap)
            else:
                # Character-based fallback
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size * 4,  # Convert tokens to approximate chars
                    chunk_overlap=chunk_overlap * 4,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks = text_splitter.split_text(text)

            # Create chunks with enhanced metadata
            chunk_objects = []
            total_tokens = 0

            for i, chunk in enumerate(chunks):
                token_count = self.token_splitter.count_tokens(chunk)
                total_tokens += token_count

                chunk_obj = {
                    "content": chunk,
                    "metadata": {
                        "chunk_id": i,
                        "chunk_type": "recursive",
                        "chunk_size_chars": len(chunk),
                        "chunk_size_tokens": token_count,
                        "overlap_tokens": chunk_overlap if i > 0 else 0,
                        "source_file": kwargs.get("source_file", "unknown"),
                        "file_index": kwargs.get("file_index", 0),
                        "start_char": self._estimate_char_position(i, chunk_size, chunk_overlap),
                        "end_char": self._estimate_char_position(i, chunk_size, chunk_overlap) + len(chunk),
                        "language": self._detect_language(chunk),
                        "complexity_score": self._calculate_complexity(chunk)
                    }
                }
                chunk_objects.append(chunk_obj)

            logger.info(f"Created {len(chunk_objects)} recursive chunks, total tokens: {total_tokens}")
            return chunk_objects

        except Exception as e:
            logger.error(f"Error in recursive chunking: {e}")
            return []

    def semantic_chunking(
            self,
            text: str,
            min_chunk_tokens: int = 100,
            max_chunk_tokens: int = CHUNK_SIZE_TOKENS,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """Enhanced semantic chunking with better boundary detection."""
        try:
            chunks = []

            if self.nlp:
                # Use spaCy for sophisticated sentence segmentation
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                # Enhanced regex-based sentence splitting
                sentences = self._advanced_sentence_split(text)

            current_chunk = ""
            current_start = 0
            sentence_boundaries = []

            for i, sentence in enumerate(sentences):
                current_tokens = self.token_splitter.count_tokens(current_chunk)
                sentence_tokens = self.token_splitter.count_tokens(sentence)

                # Check if adding this sentence would exceed max chunk size
                if current_tokens + sentence_tokens > max_chunk_tokens and current_tokens > min_chunk_tokens:
                    # Save current chunk if it meets minimum size
                    if current_chunk.strip():
                        chunk_obj = self._create_semantic_chunk(
                            current_chunk.strip(), len(chunks), current_start, i,
                            sentence_boundaries, **kwargs
                        )
                        chunks.append(chunk_obj)

                    # Start new chunk
                    current_chunk = sentence
                    current_start = i
                    sentence_boundaries = [i]
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    sentence_boundaries.append(i)

            # Add final chunk
            if current_chunk.strip():
                chunk_obj = self._create_semantic_chunk(
                    current_chunk.strip(), len(chunks), current_start, len(sentences),
                    sentence_boundaries, **kwargs
                )
                chunks.append(chunk_obj)

            logger.info(f"Created {len(chunks)} semantic chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return self.recursive_chunking(text, **kwargs)

    def hierarchical_chunking(
            self,
            text: str,
            section_markers: Optional[List[str]] = None,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """Enhanced hierarchical chunking with better structure detection."""
        try:
            if section_markers is None:
                section_markers = [
                    r'^#{1,6}\s+(.+)',  # Markdown headers with capture group
                    r'^(\d+\.(?:\d+\.)*)\s+(.+)',  # Numbered sections
                    r'^([A-Z][A-Z\s]+):(.+)',  # ALL CAPS headers
                    r'^(Chapter\s+\d+):?\s*(.+)?',  # Chapter markers
                    r'^(Section\s+\d+(?:\.\d+)*):?\s*(.+)?',  # Section markers
                    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):$',  # Title case headers
                ]

            chunks = []
            sections = self._enhanced_section_split(text, section_markers)

            for section_idx, section in enumerate(sections):
                section_tokens = self.token_splitter.count_tokens(section["content"])

                if section_tokens > CHUNK_SIZE_TOKENS:
                    # Further split large sections while preserving hierarchy
                    subsections = self._split_large_section_tokens(section["content"])
                    for subsection_idx, subsection in enumerate(subsections):
                        chunk_obj = self._create_hierarchical_chunk(
                            subsection, len(chunks), section_idx, subsection_idx,
                            section, subsections=True, **kwargs
                        )
                        chunks.append(chunk_obj)
                else:
                    chunk_obj = self._create_hierarchical_chunk(
                        section["content"], len(chunks), section_idx, 0,
                        section, subsections=False, **kwargs
                    )
                    chunks.append(chunk_obj)

            logger.info(f"Created {len(chunks)} hierarchical chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error in hierarchical chunking: {e}")
            return self.recursive_chunking(text, **kwargs)

    def custom_chunking(
            self,
            text: str,
            preserve_tables: bool = True,
            preserve_code_blocks: bool = True,
            preserve_lists: bool = True,
            preserve_figures: bool = True,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """Enhanced custom chunking with better special element handling."""
        try:
            chunks = []

            # Extract and preserve special elements with enhanced detection
            special_elements = self._extract_enhanced_special_elements(
                text, preserve_tables, preserve_code_blocks, preserve_lists, preserve_figures
            )

            # Process remaining text
            processed_text = text
            element_map = {}

            for element in special_elements:
                placeholder = f"[SPECIAL_ELEMENT_{element['id']}]"
                processed_text = processed_text.replace(element["content"], placeholder)
                element_map[placeholder] = element

            # Apply semantic chunking to processed text
            base_chunks = self.semantic_chunking(processed_text, **kwargs)

            # Reintegrate special elements with enhanced metadata
            for chunk in base_chunks:
                content = chunk["content"]
                chunk_elements = []

                # Replace placeholders with actual special elements
                for placeholder, element in element_map.items():
                    if placeholder in content:
                        content = content.replace(placeholder, element["content"])
                        chunk_elements.append(element)

                chunk_obj = {
                    "content": content,
                    "metadata": {
                        **chunk["metadata"],
                        "chunk_type": "custom",
                        "special_elements": chunk_elements,
                        "has_tables": any(elem["type"] == "table" for elem in chunk_elements),
                        "has_code": any(elem["type"] == "code" for elem in chunk_elements),
                        "has_lists": any(elem["type"] == "list" for elem in chunk_elements),
                        "has_figures": any(elem["type"] == "figure" for elem in chunk_elements),
                        "element_count": len(chunk_elements),
                        "structure_preserved": True
                    }
                }
                chunks.append(chunk_obj)

            logger.info(f"Created {len(chunks)} custom chunks with {len(special_elements)} special elements")
            return chunks

        except Exception as e:
            logger.error(f"Error in custom chunking: {e}")
            return self.recursive_chunking(text, **kwargs)

    # Helper methods

    def _advanced_sentence_split(self, text: str) -> List[str]:
        """Enhanced sentence splitting with better abbreviation handling."""
        # Common abbreviations that shouldn't trigger sentence breaks
        abbreviations = r'\b(?:Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Co|etc|vs|i\.e|e\.g|cf|al|St|Ave|Blvd)\.'

        # Replace abbreviations temporarily
        protected_text = re.sub(abbreviations, lambda m: m.group().replace('.', '<DOT>'), text, flags=re.IGNORECASE)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected_text)

        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences if s.strip()]

        return sentences

    def _create_semantic_chunk(self, content: str, chunk_id: int, start_sentence: int,
                               end_sentence: int, sentence_boundaries: List[int], **kwargs) -> Dict[str, Any]:
        """Create semantic chunk with enhanced metadata."""
        return {
            "content": content,
            "metadata": {
                "chunk_id": chunk_id,
                "chunk_type": "semantic",
                "chunk_size_chars": len(content),
                "chunk_size_tokens": self.token_splitter.count_tokens(content),
                "sentence_count": len(sentence_boundaries),
                "sentence_boundaries": sentence_boundaries,
                "source_file": kwargs.get("source_file", "unknown"),
                "file_index": kwargs.get("file_index", 0),
                "start_sentence": start_sentence,
                "end_sentence": end_sentence,
                "semantic_coherence": self._calculate_coherence(content),
                "language": self._detect_language(content),
                "complexity_score": self._calculate_complexity(content)
            }
        }

    def _enhanced_section_split(self, text: str, section_markers: List[str]) -> List[Dict[str, Any]]:
        """Enhanced section splitting with better header detection."""
        sections = []
        lines = text.split('\n')
        current_section = {"content": "", "title": "", "level": 1, "type": "content"}

        for line in lines:
            is_section_header = False
            header_info = None

            for marker in section_markers:
                match = re.match(marker, line.strip())
                if match:
                    # Save previous section
                    if current_section["content"].strip():
                        sections.append(current_section)

                    # Extract header information
                    groups = match.groups()
                    if len(groups) >= 2:
                        header_number, header_title = groups[0], groups[1] if groups[1] else ""
                    else:
                        header_number, header_title = "", groups[0] if groups else line.strip()

                    # Start new section
                    current_section = {
                        "content": "",
                        "title": header_title.strip(),
                        "level": self._determine_section_level(line, marker),
                        "type": "section",
                        "header_number": header_number,
                        "full_header": line.strip()
                    }
                    is_section_header = True
                    break

            if not is_section_header:
                current_section["content"] += line + "\n"

        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _split_large_section_tokens(self, content: str) -> List[str]:
        """Split large sections by token count."""
        return self.token_splitter.split_text_by_tokens(content, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)

    def _create_hierarchical_chunk(self, content: str, chunk_id: int, section_idx: int,
                                   subsection_idx: int, section: Dict[str, Any],
                                   subsections: bool, **kwargs) -> Dict[str, Any]:
        """Create hierarchical chunk with enhanced metadata."""
        return {
            "content": content,
            "metadata": {
                "chunk_id": chunk_id,
                "chunk_type": "hierarchical",
                "chunk_size_chars": len(content),
                "chunk_size_tokens": self.token_splitter.count_tokens(content),
                "section_id": section_idx,
                "subsection_id": subsection_idx,
                "section_title": section.get("title", f"Section {section_idx}"),
                "section_number": section.get("header_number", ""),
                "hierarchy_level": section.get("level", 1) + (1 if subsections else 0),
                "has_subsections": subsections,
                "section_type": section.get("type", "content"),
                "source_file": kwargs.get("source_file", "unknown"),
                "file_index": kwargs.get("file_index", 0),
                "structure_preserved": True
            }
        }

    def _extract_enhanced_special_elements(self, text: str, preserve_tables: bool,
                                           preserve_code_blocks: bool, preserve_lists: bool,
                                           preserve_figures: bool) -> List[Dict[str, Any]]:
        """Enhanced special element extraction."""
        special_elements = []
        element_id = 0

        if preserve_tables:
            # Enhanced table detection
            table_patterns = [
                r'(?:^.*\|.*\|.*$\n?)+',  # Pipe-separated tables
                r'(?:^.*\t.*\t.*$\n?)+',  # Tab-separated tables
                r'(?:^.+\s{3,}.+\s{3,}.+$\n?)+',  # Space-separated tables
            ]

            for pattern in table_patterns:
                tables = re.finditer(pattern, text, re.MULTILINE)
                for match in tables:
                    special_elements.append({
                        "id": element_id,
                        "type": "table",
                        "content": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "rows": len(match.group().strip().split('\n')),
                        "columns": match.group().count('|') // len(
                            match.group().strip().split('\n')) if '|' in match.group() else 0
                    })
                    element_id += 1

        if preserve_code_blocks:
            # Enhanced code block detection
            code_patterns = [
                r'```[\s\S]*?```',  # Fenced code blocks
                r'(?:^[ \t]{4,}.*$\n?)+',  # Indented code blocks
                r'`[^`\n]+`',  # Inline code
                r'<code>[\s\S]*?</code>',  # HTML code tags
            ]

            for pattern in code_patterns:
                code_blocks = re.finditer(pattern, text, re.MULTILINE)
                for match in code_blocks:
                    special_elements.append({
                        "id": element_id,
                        "type": "code",
                        "content": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "language": self._extract_code_language(match.group()),
                        "lines": len(match.group().strip().split('\n'))
                    })
                    element_id += 1

        if preserve_lists:
            # Enhanced list detection
            list_patterns = [
                r'(?:^[ \t]*[-*+]\s+.+$\n?)+',  # Bullet lists
                r'(?:^[ \t]*\d+\.\s+.+$\n?)+',  # Numbered lists
                r'(?:^[ \t]*[a-zA-Z]\.\s+.+$\n?)+',  # Lettered lists
            ]

            for pattern in list_patterns:
                lists = re.finditer(pattern, text, re.MULTILINE)
                for match in lists:
                    special_elements.append({
                        "id": element_id,
                        "type": "list",
                        "content": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "items": len(re.findall(r'^[ \t]*[-*+\d+a-zA-Z]\.\s+', match.group(), re.MULTILINE)),
                        "list_type": "ordered" if re.search(r'\d+\.', match.group()) else "unordered"
                    })
                    element_id += 1

        if preserve_figures:
            # Enhanced figure/image detection
            figure_patterns = [
                r'!\[.*?\]\(.*?\)',  # Markdown images
                r'<img[^>]*>',  # HTML images
                r'Figure\s+\d+:.*?(?=\n\n|\n[A-Z]|\Z)',  # Figure captions
                r'Chart\s+\d+:.*?(?=\n\n|\n[A-Z]|\Z)',  # Chart captions
            ]

            for pattern in figure_patterns:
                figures = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
                for match in figures:
                    special_elements.append({
                        "id": element_id,
                        "type": "figure",
                        "content": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "has_caption": ":" in match.group(),
                        "figure_type": self._extract_figure_type(match.group())
                    })
                    element_id += 1

        return special_elements

    def _extract_code_language(self, code_block: str) -> str:
        """Extract programming language from code block."""
        if code_block.startswith('```'):
            first_line = code_block.split('\n')[0]
            lang_match = re.search(r'```(\w+)', first_line)
            return lang_match.group(1) if lang_match else "unknown"
        return "unknown"

    def _extract_figure_type(self, figure_content: str) -> str:
        """Extract figure type from content."""
        if 'chart' in figure_content.lower():
            return "chart"
        elif 'graph' in figure_content.lower():
            return "graph"
        elif 'diagram' in figure_content.lower():
            return "diagram"
        elif '.png' in figure_content or '.jpg' in figure_content or '.jpeg' in figure_content:
            return "image"
        return "unknown"

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        if not words:
            return 0.0

        # Basic complexity metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_sentence_length = len(words) / sentence_count if sentence_count > 0 else 0

        # Normalize to 0-1 scale
        complexity = min(1.0, (avg_word_length * 0.1 + avg_sentence_length * 0.05) / 3)
        return complexity

    def _calculate_coherence(self, text: str) -> float:
        """Calculate semantic coherence score."""
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return 1.0

        # Simple coherence metric based on word overlap between consecutive sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            if words1 and words2:
                overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                coherence_scores.append(overlap)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Basic heuristic - could be enhanced with proper language detection library
        english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = text.lower().split()
        english_score = sum(1 for word in words if word in english_indicators) / len(words) if words else 0
        return "en" if english_score > 0.1 else "unknown"

    def _estimate_char_position(self, chunk_index: int, chunk_size: int, overlap: int) -> int:
        """Estimate character position for chunk."""
        return chunk_index * (chunk_size * 4 - overlap * 4)  # Convert tokens to chars

    def _determine_section_level(self, line: str, marker: str) -> int:
        """Determine the hierarchical level of a section."""
        if marker.startswith('^#{'):
            return len(re.match(r'^#+', line.strip()).group())
        elif marker.startswith('^(\\d+'):
            dots = line.count('.')
            return dots + 1
        elif 'Chapter' in marker:
            return 1
        elif 'Section' in marker:
            return 2
        else:
            return 3

    def get_chunking_quality_metrics(self, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate enhanced quality metrics for chunking strategy."""
        if not chunks:
            return {}

        chunk_sizes_chars = [chunk["metadata"]["chunk_size_chars"] for chunk in chunks]
        chunk_sizes_tokens = [chunk["metadata"].get("chunk_size_tokens", 0) for chunk in chunks]

        metrics = {
            "total_chunks": len(chunks),
            "avg_chunk_size_chars": sum(chunk_sizes_chars) / len(chunk_sizes_chars),
            "avg_chunk_size_tokens": sum(chunk_sizes_tokens) / len(chunk_sizes_tokens) if any(
                chunk_sizes_tokens) else 0,
            "min_chunk_size_chars": min(chunk_sizes_chars),
            "max_chunk_size_chars": max(chunk_sizes_chars),
            "min_chunk_size_tokens": min(chunk_sizes_tokens) if any(chunk_sizes_tokens) else 0,
            "max_chunk_size_tokens": max(chunk_sizes_tokens) if any(chunk_sizes_tokens) else 0,
            "size_variance_chars": sum(
                (size - sum(chunk_sizes_chars) / len(chunk_sizes_chars)) ** 2 for size in chunk_sizes_chars) / len(
                chunk_sizes_chars),
            "size_variance_tokens": sum(
                (size - sum(chunk_sizes_tokens) / len(chunk_sizes_tokens)) ** 2 for size in chunk_sizes_tokens) / len(
                chunk_sizes_tokens) if any(chunk_sizes_tokens) else 0,
            "chunks_with_metadata": sum(1 for chunk in chunks if chunk.get("metadata")),
            "metadata_coverage": sum(1 for chunk in chunks if chunk.get("metadata")) / len(chunks),
            "chunks_with_special_elements": sum(
                1 for chunk in chunks if chunk.get("metadata", {}).get("special_elements")),
            "avg_complexity": sum(chunk.get("metadata", {}).get("complexity_score", 0) for chunk in chunks) / len(
                chunks),
            "avg_coherence": sum(chunk.get("metadata", {}).get("semantic_coherence", 0) for chunk in chunks) / len(
                chunks),
            "structure_preservation_rate": sum(
                1 for chunk in chunks if chunk.get("metadata", {}).get("structure_preserved")) / len(chunks)
        }

        return metrics