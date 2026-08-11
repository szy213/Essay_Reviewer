from .pdf_parser import PDFParser
from .text_splitter import TextSplitter
from .vector_store import VectorStore


class Retriever:
    """High-level RAG pipeline: parse → split → index → search."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        collection_name: str = "knowledge_base",
        persist_dir: str = "data/chroma_db",
        model_name: str = "BAAI/bge-small-zh-v1.5",
    ):
        self.splitter = TextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
            model_name=model_name,
        )

    def index_pdf(self, pdf_path: str) -> int:
        """Parse a PDF, split into chunks, and index in vector store."""
        pages = PDFParser.parse(pdf_path)
        chunks = self.splitter.split_pages(pages)
        if not chunks:
            return 0
        return self.vector_store.add_chunks(chunks)

    def index_directory(self, dir_path: str) -> int:
        """Index all PDFs in a directory."""
        pages = PDFParser.parse_directory(dir_path)
        chunks = self.splitter.split_pages(pages)
        if not chunks:
            return 0
        return self.vector_store.add_chunks(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the knowledge base for relevant chunks."""
        return self.vector_store.search(query, top_k=top_k)

    def clear(self):
        self.vector_store.clear()

    @property
    def chunk_count(self) -> int:
        return self.vector_store.count()
