import pdfplumber
from PyPDF2 import PdfReader
import pandas as pd
import re
from typing import List, Any, Dict, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chunking_strategies import ChunkingManager
from config import logger, CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_SIZE_MB

# Optional imports for enhanced extraction
try:
    import tabula

    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    logger.info("tabula-py not available. Advanced table extraction disabled.")

try:
    from PIL import Image
    import pytesseract

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.info("OCR libraries not available. Text extraction from images disabled.")


class EnhancedPDFExtractor:
    """Enhanced PDF extraction with improved table, image, and structure handling."""

    def __init__(self):
        self.extraction_stats = {}

    def extract_text_with_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text while preserving document structure."""
        try:
            extracted_data = {
                "raw_text": "",
                "tables": [],
                "images": [],
                "metadata": {},
                "structure": {
                    "pages": [],
                    "sections": [],
                    "headings": []
                },
                "extraction_stats": {}
            }

            # Check file size
            file_size_mb = self._get_file_size_mb(pdf_path)
            if file_size_mb > MAX_FILE_SIZE_MB:
                logger.warning(f"File size {file_size_mb:.1f}MB exceeds limit of {MAX_FILE_SIZE_MB}MB")
                return extracted_data

            # Extract using pdfplumber for better structure preservation
            with pdfplumber.open(pdf_path) as pdf:
                extracted_data["metadata"] = self._extract_pdf_metadata(pdf)

                for page_num, page in enumerate(pdf.pages):
                    page_data = self._extract_page_content(page, page_num)
                    extracted_data["structure"]["pages"].append(page_data)
                    extracted_data["raw_text"] += page_data["text"] + "\n\n"

                    # Extract tables from page
                    if page_data["tables"]:
                        extracted_data["tables"].extend(page_data["tables"])

                    # Extract images from page
                    if page_data["images"]:
                        extracted_data["images"].extend(page_data["images"])

            # Fallback extraction with PyPDF2 if pdfplumber fails
            if not extracted_data["raw_text"].strip():
                logger.info("Falling back to PyPDF2 extraction")
                extracted_data["raw_text"] = self._extract_with_pypdf2(pdf_path)

            # Extract document structure
            extracted_data["structure"]["headings"] = self._extract_headings(extracted_data["raw_text"])
            extracted_data["structure"]["sections"] = self._extract_sections(extracted_data["raw_text"])

            # Advanced table extraction with tabula if available
            if TABULA_AVAILABLE:
                tabula_tables = self._extract_tables_with_tabula(pdf_path)
                extracted_data["tables"].extend(tabula_tables)

            # Generate extraction statistics
            extracted_data["extraction_stats"] = self._generate_extraction_stats(extracted_data)

            logger.info(f"Extracted {len(extracted_data['raw_text'])} characters, "
                        f"{len(extracted_data['tables'])} tables, "
                        f"{len(extracted_data['images'])} images from {pdf_path}")

            return extracted_data

        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {e}")
            return {"raw_text": "", "tables": [], "images": [], "metadata": {},
                    "structure": {"pages": [], "sections": [], "headings": []},
                    "extraction_stats": {}}

    def _extract_page_content(self, page, page_num: int) -> Dict[str, Any]:
        """Extract content from a single page."""
        page_data = {
            "page_number": page_num + 1,
            "text": "",
            "tables": [],
            "images": [],
            "bbox": page.bbox,
            "rotation": getattr(page, 'rotation', 0)
        }

        try:
            # Extract text
            page_data["text"] = page.extract_text() or ""

            # Extract tables
            tables = page.extract_tables()
            if tables:
                for table_idx, table in enumerate(tables):
                    table_data = self._process_table(table, page_num + 1, table_idx)
                    if table_data:
                        page_data["tables"].append(table_data)

            # Extract images (if available)
            if hasattr(page, 'images') and page.images:
                for img_idx, img in enumerate(page.images):
                    img_data = self._process_image(img, page_num + 1, img_idx)
                    if img_data:
                        page_data["images"].append(img_data)

            # Detect special elements
            page_data.update(self._detect_special_elements(page_data["text"]))

        except Exception as e:
            logger.warning(f"Error extracting from page {page_num + 1}: {e}")
            page_data["text"] = f"[Error extracting page {page_num + 1}]"

        return page_data

    def _process_table(self, table: List[List], page_num: int, table_idx: int) -> Optional[Dict[str, Any]]:
        """Process and clean extracted table."""
        if not table or not any(any(cell for cell in row if cell) for row in table):
            return None

        try:
            # Clean table data
            cleaned_table = []
            for row in table:
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                if any(cleaned_row):  # Skip empty rows
                    cleaned_table.append(cleaned_row)

            if not cleaned_table:
                return None

            # Convert to DataFrame for easier processing
            df = pd.DataFrame(cleaned_table)

            # Try to identify headers
            potential_header = cleaned_table[0] if cleaned_table else []
            has_header = self._likely_header_row(potential_header)

            table_data = {
                "table_id": f"page_{page_num}_table_{table_idx}",
                "page_number": page_num,
                "rows": len(cleaned_table),
                "columns": len(cleaned_table[0]) if cleaned_table else 0,
                "data": cleaned_table,
                "has_header": has_header,
                "header": potential_header if has_header else None,
                "csv_representation": df.to_csv(index=False, header=False),
                "text_representation": self._table_to_text(cleaned_table, has_header)
            }

            return table_data

        except Exception as e:
            logger.warning(f"Error processing table on page {page_num}: {e}")
            return None

    def _process_image(self, img_info: Dict, page_num: int, img_idx: int) -> Optional[Dict[str, Any]]:
        """Process extracted image information."""
        try:
            image_data = {
                "image_id": f"page_{page_num}_img_{img_idx}",
                "page_number": page_num,
                "bbox": img_info.get("bbox", []),
                "width": img_info.get("width", 0),
                "height": img_info.get("height", 0),
                "name": img_info.get("name", f"image_{img_idx}"),
                "extracted_text": ""
            }

            # If OCR is available, try to extract text from image
            if OCR_AVAILABLE and "image" in img_info:
                try:
                    # This would require more complex implementation to extract actual image data
                    # For now, we'll just note that OCR capability exists
                    image_data["ocr_available"] = True
                except Exception as e:
                    logger.warning(f"OCR failed for image on page {page_num}: {e}")

            return image_data

        except Exception as e:
            logger.warning(f"Error processing image on page {page_num}: {e}")
            return None

    def _extract_tables_with_tabula(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using tabula-py for better accuracy."""
        tables = []

        try:
            # Extract tables from all pages
            tabula_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

            for idx, df in enumerate(tabula_tables):
                if not df.empty:
                    table_data = {
                        "table_id": f"tabula_table_{idx}",
                        "page_number": -1,  # tabula doesn't provide page info easily
                        "rows": len(df),
                        "columns": len(df.columns),
                        "data": df.values.tolist(),
                        "header": df.columns.tolist(),
                        "has_header": True,
                        "csv_representation": df.to_csv(index=False),
                        "text_representation": self._dataframe_to_text(df),
                        "extraction_method": "tabula"
                    }
                    tables.append(table_data)

            logger.info(f"Extracted {len(tables)} tables using tabula")

        except Exception as e:
            logger.warning(f"Tabula table extraction failed: {e}")

        return tables

    def _extract_headings(self, text: str) -> List[Dict[str, Any]]:
        """Extract document headings and their hierarchy."""
        headings = []

        # Patterns for different heading styles
        heading_patterns = [
            (r'^#{1,6}\s+(.+)$', 'markdown'),  # Markdown headings
            (r'^(\d+\.(?:\d+\.)*)\s+(.+)$', 'numbered'),  # Numbered headings
            (r'^([A-Z][A-Z\s]+)$', 'caps'),  # ALL CAPS headings
            (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$', 'title_case'),  # Title case
        ]

        lines = text.split('\n')

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            for pattern, style in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    if style == 'markdown':
                        level = len(line.split()[0])  # Count # symbols
                        title = match.group(1)
                    elif style == 'numbered':
                        level = line.count('.') + 1
                        title = match.group(2)
                    else:
                        level = 1  # Default level for other styles
                        title = match.group(1) if match.groups() else line

                    headings.append({
                        "title": title.strip(),
                        "level": level,
                        "style": style,
                        "line_number": line_num,
                        "full_text": line
                    })
                    break

        return headings

    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract document sections based on headings."""
        headings = self._extract_headings(text)
        sections = []

        lines = text.split('\n')

        for i, heading in enumerate(headings):
            start_line = heading["line_number"]

            # Find end line (next heading of same or higher level)
            end_line = len(lines)
            for j in range(i + 1, len(headings)):
                if headings[j]["level"] <= heading["level"]:
                    end_line = headings[j]["line_number"]
                    break

            # Extract section content
            section_lines = lines[start_line:end_line]
            section_content = '\n'.join(section_lines).strip()

            sections.append({
                "title": heading["title"],
                "level": heading["level"],
                "start_line": start_line,
                "end_line": end_line,
                "content": section_content,
                "word_count": len(section_content.split()),
                "char_count": len(section_content)
            })

        return sections

    def _detect_special_elements(self, text: str) -> Dict[str, Any]:
        """Detect special elements in text."""
        elements = {
            "has_equations": bool(re.search(r'\$.*?\$|\\[a-zA-Z]+|∑|∫|∂', text)),
            "has_code": bool(re.search(r'```|def |class |import |function\(', text)),
            "has_urls": bool(re.search(r'https?://\S+', text)),
            "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            "has_phone_numbers": bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)),
            "has_dates": bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)),
            "language": self._detect_language(text)
        }

        return elements

    def _likely_header_row(self, row: List[str]) -> bool:
        """Determine if a table row is likely a header."""
        if not row:
            return False

        # Check for header indicators
        header_indicators = 0

        for cell in row:
            if not cell:
                continue

            cell_lower = cell.lower()

            # Check for typical header characteristics
            if any(word in cell_lower for word in ['name', 'id', 'date', 'value', 'type', 'description']):
                header_indicators += 1

            # Headers often don't contain numbers
            if not re.search(r'\d', cell):
                header_indicators += 0.5

            # Headers are often shorter
            if len(cell.split()) <= 3:
                header_indicators += 0.3

        # If more than half the cells look like headers
        return header_indicators >= len([cell for cell in row if cell]) / 2

    def _table_to_text(self, table_data: List[List[str]], has_header: bool) -> str:
        """Convert table to readable text representation."""
        if not table_data:
            return ""

        text_parts = []

        if has_header and table_data:
            headers = table_data[0]
            data_rows = table_data[1:]

            # Create column-based representation
            text_parts.append("Table with columns: " + ", ".join(headers))

            for row in data_rows[:5]:  # Limit to first 5 rows for readability
                row_text = []
                for i, cell in enumerate(row):
                    if i < len(headers) and cell:
                        row_text.append(f"{headers[i]}: {cell}")

                if row_text:
                    text_parts.append(" | ".join(row_text))
        else:
            # Simple row-based representation
            for row in table_data[:5]:
                row_text = " | ".join(str(cell) for cell in row if cell)
                if row_text:
                    text_parts.append(row_text)

        return "\n".join(text_parts)

    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to readable text."""
        text_parts = []

        # Add column information
        text_parts.append(f"Table with columns: {', '.join(df.columns)}")

        # Add first few rows
        for idx, row in df.head().iterrows():
            row_text = []
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    row_text.append(f"{col}: {value}")

            if row_text:
                text_parts.append(" | ".join(row_text))

        return "\n".join(text_parts)

    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Basic heuristic for English detection
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = text.lower().split()[:100]  # Check first 100 words

        if not words:
            return "unknown"

        english_score = sum(1 for word in words if word in english_words) / len(words)
        return "en" if english_score > 0.1 else "unknown"

    def _extract_pdf_metadata(self, pdf) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {}

        try:
            if hasattr(pdf, 'metadata') and pdf.metadata:
                for key, value in pdf.metadata.items():
                    if value:
                        metadata[key.lower()] = str(value)

            # Add page count
            metadata['page_count'] = len(pdf.pages)

            # Add page sizes
            if pdf.pages:
                first_page = pdf.pages[0]
                metadata['page_width'] = first_page.width
                metadata['page_height'] = first_page.height

        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")

        return metadata

    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Fallback extraction using PyPDF2."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")

        return text

    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0

    def _generate_extraction_stats(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate extraction statistics."""
        stats = {
            "total_text_length": len(extracted_data["raw_text"]),
            "total_pages": len(extracted_data["structure"]["pages"]),
            "total_tables": len(extracted_data["tables"]),
            "total_images": len(extracted_data["images"]),
            "total_headings": len(extracted_data["structure"]["headings"]),
            "total_sections": len(extracted_data["structure"]["sections"]),
            "avg_page_length": 0,
            "table_extraction_rate": 0,
            "structure_detected": False
        }

        if stats["total_pages"] > 0:
            page_lengths = [len(page.get("text", "")) for page in extracted_data["structure"]["pages"]]
            stats["avg_page_length"] = sum(page_lengths) / len(page_lengths)

        if stats["total_pages"] > 0:
            stats["table_extraction_rate"] = stats["total_tables"] / stats["total_pages"]

        stats["structure_detected"] = stats["total_headings"] > 0 or stats["total_sections"] > 0

        return stats


def get_pdf_text(pdf_docs: List[Any]) -> str:
    """Extract text from uploaded PDF files using enhanced extraction."""
    extractor = EnhancedPDFExtractor()
    combined_text = ""

    for pdf in pdf_docs:
        try:
            extracted_data = extractor.extract_text_with_structure(pdf)

            # Combine text with table representations
            text_parts = [extracted_data["raw_text"]]

            # Add table text representations
            for table in extracted_data["tables"]:
                if table.get("text_representation"):
                    text_parts.append(f"\n[TABLE]\n{table['text_representation']}\n[/TABLE]\n")

            combined_text += "\n".join(text_parts) + "\n\n"

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")

    return combined_text


def get_text_chunks(text: str) -> List[str]:
    """Split text into chunks for processing."""
    chunks = []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Error at get_text_chunks function: {e}")
    return chunks


def extract_all_pages_as_images(file_path: str) -> List[Any]:
    """Extract all pages from PDF as images for viewer."""
    logger.info(f"Extracting all pages as images from file: {file_path}")
    pdf_pages = []
    try:
        with pdfplumber.open(file_path) as pdf:
            pdf_pages = [page.to_image().original for page in pdf.pages]
        logger.info(f"Extracted {len(pdf_pages)} pages as images")
    except Exception as e:
        logger.error(f"Error extracting PDF pages as images: {e}")
    return pdf_pages


def process_uploaded_pdfs(files, chunking_strategy="recursive"):
    """
    Enhanced PDF processing with improved structure preservation and metadata.
    """
    if not files:
        return [], []

    all_chunks = []
    pdf_images = []
    extractor = EnhancedPDFExtractor()

    # Initialize chunking manager
    chunking_manager = ChunkingManager()

    # Process each file
    for i, file_path in enumerate(files):
        try:
            # Enhanced extraction with structure preservation
            extracted_data = extractor.extract_text_with_structure(file_path)
            raw_text = extracted_data["raw_text"]

            if not raw_text.strip():
                logger.warning(f"No text extracted from {file_path}")
                continue

            # Combine text with structured elements
            enhanced_text = raw_text

            # Add table representations to text
            for table in extracted_data["tables"]:
                if table.get("text_representation"):
                    table_marker = f"\n[TABLE_{table['table_id']}]\n{table['text_representation']}\n[/TABLE]\n"
                    enhanced_text += table_marker

            # Apply chunking strategy with enhanced metadata
            base_kwargs = {
                "source_file": file_path,
                "file_index": i,
                "pdf_metadata": extracted_data["metadata"],
                "extraction_stats": extracted_data["extraction_stats"],
                "has_tables": len(extracted_data["tables"]) > 0,
                "has_images": len(extracted_data["images"]) > 0,
                "structure_info": extracted_data["structure"]
            }

            if chunking_strategy == "recursive":
                chunks = chunking_manager.recursive_chunking(
                    enhanced_text, **base_kwargs
                )
            elif chunking_strategy == "semantic":
                chunks = chunking_manager.semantic_chunking(
                    enhanced_text, **base_kwargs
                )
            elif chunking_strategy == "hierarchical":
                # Use extracted headings for better hierarchical chunking
                section_markers = None
                if extracted_data["structure"]["headings"]:
                    # Create patterns from detected headings
                    section_markers = [rf'^{re.escape(h["title"])}\s*$' for h in
                                       extracted_data["structure"]["headings"][:10]]

                chunks = chunking_manager.hierarchical_chunking(
                    enhanced_text, section_markers=section_markers, **base_kwargs
                )
            elif chunking_strategy == "custom":
                chunks = chunking_manager.custom_chunking(
                    enhanced_text,
                    preserve_tables=True,
                    preserve_code_blocks=True,
                    preserve_lists=True,
                    preserve_figures=True,
                    **base_kwargs
                )
            else:
                # Default to recursive
                chunks = chunking_manager.recursive_chunking(
                    enhanced_text, **base_kwargs
                )

            # Enhance chunk metadata with extraction info
            for chunk in chunks:
                chunk["metadata"].update({
                    "tables_in_file": len(extracted_data["tables"]),
                    "images_in_file": len(extracted_data["images"]),
                    "file_structure_detected": extracted_data["extraction_stats"]["structure_detected"],
                    "extraction_method": "enhanced"
                })

            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {file_path}")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue

    # Extract images from first PDF for viewer
    if files:
        try:
            pdf_images = extract_all_pages_as_images(files[0])
        except Exception as e:
            logger.error(f"Error extracting images for viewer: {e}")
            pdf_images = []

    logger.info(f"Total processing complete: {len(all_chunks)} chunks from {len(files)} files")
    return all_chunks, pdf_images