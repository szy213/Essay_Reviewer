import re


class TextSplitter:
    """Split text into overlapping chunks with configurable size and separators."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",
            "\n",
            "。",
            "；",
            "，",
            " ",
            "",
        ]

    def split(self, text: str) -> list[str]:
        """Split text into chunks recursively using separators."""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        separator = separators[0]
        next_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        chunks = []
        current = ""

        for part in splits:
            piece = part if not current else separator + part

            if len(current) + len(piece) > self.chunk_size and current:
                chunks.append(self._clean(current))
                # Keep overlap: rewind by removing from the front
                if self.chunk_overlap > 0 and next_separators:
                    current = self._trim_to_overlap(current, piece)
                else:
                    current = ""

            current += piece

        if current.strip():
            chunks.append(self._clean(current))

        # If any chunk is still too large, recurse with next separator
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size and next_separators:
                final_chunks.extend(self._split_text(chunk, next_separators))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _trim_to_overlap(self, prev: str, next_part: str) -> str:
        """Trim previous text to maintain overlap context."""
        target_len = max(0, self.chunk_overlap - len(next_part))
        return prev[-target_len:] if target_len > 0 else ""

    @staticmethod
    def _clean(text: str) -> str:
        """Clean whitespace from chunk."""
        return re.sub(r"\s+", " ", text).strip()

    def split_pages(self, pages: list[dict]) -> list[dict]:
        """Split parsed PDF pages into chunks, preserving metadata."""
        chunks = []

        for page_info in pages:
            text_chunks = self.split(page_info["text"])
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **page_info["metadata"],
                        "chunk_index": i,
                        "chunk_page": page_info["page"],
                    },
                })

        return chunks
