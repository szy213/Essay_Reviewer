import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Embedding generation and ChromaDB vector storage."""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_dir: str = "data/chroma_db",
        model_name: str = "BAAI/bge-small-zh-v1.5",
    ):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.model = SentenceTransformer(model_name)

    def add_chunks(self, chunks: list[dict], batch_size: int = 32) -> int:
        """Embed and insert chunks into the vector store.

        Returns the total number of chunks added.
        """
        total = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.model.encode(texts, normalize_embeddings=True).tolist()

            ids = []
            metadatas = []
            for j, chunk in enumerate(batch):
                chunk_id = self._make_id(chunk["metadata"], i + j)
                ids.append(chunk_id)
                metadatas.append(chunk["metadata"])

            # Upsert to handle re-indexing gracefully
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            total += len(batch)

        return total

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for the most relevant chunks to a query."""
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine distance → similarity
            })

        return hits

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        """Remove all chunks from the collection."""
        if self.count() > 0:
            ids = self.collection.get()["ids"]
            self.collection.delete(ids=ids)

    @staticmethod
    def _make_id(metadata: dict, idx: int) -> str:
        source = metadata.get("filename", "unknown")
        page = metadata.get("page", 0)
        chunk_idx = metadata.get("chunk_index", idx)
        return f"{source}_p{page}_c{chunk_idx}"
