from pathlib import Path
from typing import Any

from .base import BaseDocumentExtractor, ExtractedChunk


class TXTExtractor(BaseDocumentExtractor):
    def __init__(self, chunk_word_count = 100, overlap_ratio = 0.2):
        super().__init__(source_type="txt")
        self.chunk_word_count = chunk_word_count
        self.overlap_ratio = overlap_ratio

    def extract(self, file_path: str | Path) -> list[ExtractedChunk]:
        text = Path(file_path).read_text(encoding="utf-8")
        words = text.split()
        
        chunks: list[ExtractedChunk] = []
        step_size: int = max(1, int(self.chunk_word_count * (1.0 - self.overlap_ratio)))
        for i in range(0, len(words), step_size):
            window = words[i : i + self.chunk_word_count]

            if not window:
                break

            content = " ".join(window)

            chunks.append(
                ExtractedChunk(
                    content=content,
                    locator={},
                    source_type=self.source_type,
                )
            )

        return chunks

    def get_useful_metadata(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        text = file_path.read_text(encoding="utf-8")
        return {
            "line_count": len(text.splitlines()),
            "word_count": len(text.split()),
        } | self._get_base_meta_data(file_path)
