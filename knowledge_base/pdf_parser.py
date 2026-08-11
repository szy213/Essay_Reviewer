import re
import zlib
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFParser:
    """Extract text from PDF files with page-level metadata.

    Falls back to raw stream decompression for damaged PDFs.
    """

    @staticmethod
    def parse(file_path: str | Path) -> list[dict]:
        """Parse a PDF file and return a list of page dicts."""
        file_path = Path(file_path)

        # Try standard pypdf parsing first
        try:
            pages = PDFParser._parse_with_pypdf(file_path)
            if pages and any(p["text"] for p in pages):
                return pages
        except (PdfReadError, Exception):
            pass

        # Fall back to raw stream extraction for damaged PDFs
        return PDFParser._parse_raw(file_path)

    @staticmethod
    def _parse_with_pypdf(file_path: Path) -> list[dict]:
        reader = PdfReader(str(file_path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page": i,
                    "text": text.strip(),
                    "metadata": {
                        "source": str(file_path.resolve()),
                        "filename": file_path.name,
                        "page": i,
                    },
                })
        return pages

    @staticmethod
    def _parse_raw(file_path: Path) -> list[dict]:
        """Extract text by decompressing FlateDecode streams directly."""
        raw_bytes = file_path.read_bytes()

        # Find page objects and their content stream references
        page_refs = PDFParser._find_page_content_refs(raw_bytes)

        # Decompress all streams indexed by object number
        streams = PDFParser._decompress_all_streams(raw_bytes)

        pages = []
        for page_num, content_refs in enumerate(page_refs):
            page_text_parts = []
            for ref in content_refs:
                if ref in streams:
                    text = PDFParser._extract_text_from_stream(streams[ref])
                    if text:
                        page_text_parts.append(text)

            combined = "\n".join(page_text_parts).strip()
            if combined:
                pages.append({
                    "page": page_num,
                    "text": combined,
                    "metadata": {
                        "source": str(file_path.resolve()),
                        "filename": file_path.name,
                        "page": page_num,
                    },
                })

        return pages

    @staticmethod
    def _find_page_content_refs(raw: bytes) -> list[list[int]]:
        """Find each page's /Contents object references by parsing page objects."""
        # Match page objects: << ... /Type/Page ... /Contents X 0 R ... >>
        page_pattern = re.compile(
            rb'<<.*?/Type\s*/Page\s*.*?/Contents\s+(\d+(?:\s+\d+\s+R\s*)*).*?>>',
            re.DOTALL,
        )
        content_refs = []
        for match in page_pattern.finditer(raw):
            refs_block = match.group(1)
            # Extract all content object numbers
            refs = [int(n) for n in re.findall(rb'(\d+)\s+\d+\s+R', refs_block)]
            if refs:
                content_refs.append(refs)
        return content_refs

    @staticmethod
    def _decompress_all_streams(raw: bytes) -> dict[int, bytes]:
        """Decompress all FlateDecode streams, keyed by object number."""
        # Match: N 0 obj ... stream ... endstream ... endobj
        obj_pattern = re.compile(
            rb'(\d+)\s+\d+\s+obj.*?<<(.*?)>>.*?stream\r?\n(.*?)endstream',
            re.DOTALL,
        )
        streams = {}
        for match in obj_pattern.finditer(raw):
            obj_num = int(match.group(1))
            header = match.group(2)
            stream_data = match.group(3)

            # Only process FlateDecode streams
            if b'FlateDecode' not in header:
                continue

            try:
                decompressed = zlib.decompress(stream_data)
                streams[obj_num] = decompressed
            except zlib.error:
                pass

        return streams

    @staticmethod
    def _extract_text_from_stream(data: bytes) -> str:
        """Extract text from BT...ET blocks in decompressed content stream."""
        bt_blocks = re.findall(rb'BT(.*?)ET', data, re.DOTALL)
        texts = []
        for block in bt_blocks:
            # Extract text from (string) Tj operations
            tj_matches = re.findall(rb'\((.*?)\)\s*Tj', block, re.DOTALL)
            for t in tj_matches:
                decoded = t.decode('utf-8', errors='replace')
                texts.append(decoded)
        return "".join(texts)

    @staticmethod
    def parse_directory(dir_path: str | Path) -> list[dict]:
        """Parse all PDF files in a directory."""
        dir_path = Path(dir_path)
        all_pages = []
        for pdf_file in sorted(dir_path.glob("*.pdf")):
            all_pages.extend(PDFParser.parse(pdf_file))
        return all_pages
