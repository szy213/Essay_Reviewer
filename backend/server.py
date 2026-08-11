"""FastAPI server for RAG Agent — chat and file upload."""

import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agent import RAGAgent
from knowledge_base.pdf_parser import PDFParser
from knowledge_base.text_splitter import TextSplitter
from knowledge_base.vector_store import VectorStore

app = FastAPI(title="Essay Reviewer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "maogai_knowledge"
PERSIST_DIR = "data/chroma_db"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

agent = RAGAgent(collection_name=COLLECTION_NAME)
vector_store = VectorStore(collection_name=COLLECTION_NAME, persist_dir=str(PROJECT_ROOT / PERSIST_DIR))
text_splitter = TextSplitter(chunk_size=500, chunk_overlap=50)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    total_indexed: int


class FileInfo(BaseModel):
    filename: str
    chunks: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    response = agent.run(req.message)
    return ChatResponse(answer=response)


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF, parse / split / embed / index into ChromaDB."""

    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pages = PDFParser.parse(tmp_path)
        chunks = text_splitter.split_pages(pages)
        count = vector_store.add_chunks(chunks)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadResponse(
        filename=file.filename or "unknown",
        pages=len(pages),
        chunks=len(chunks),
        total_indexed=vector_store.count(),
    )


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": vector_store.count()}


@app.get("/api/files", response_model=list[FileInfo])
def list_files():
    """List uploaded PDFs with chunk counts from ChromaDB metadata."""
    if vector_store.count() == 0:
        return []

    data = vector_store.collection.get(include=["metadatas"])
    file_stats: dict[str, int] = {}
    for meta in data["metadatas"]:
        fname = meta.get("filename", "unknown")
        file_stats[fname] = file_stats.get(fname, 0) + 1

    return [FileInfo(filename=k, chunks=v) for k, v in file_stats.items()]


# Serve frontend static files
frontend_dir = PROJECT_ROOT / "frontend"
frontend_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
