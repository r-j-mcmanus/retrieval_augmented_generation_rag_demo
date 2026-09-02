from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .base import BaseDocumentExtractor, ExtractedChunk


class PDFExtractor(BaseDocumentExtractor):
    def __init__(self, chunk_word_count = 100, overlap_ratio = 0.2):
        super().__init__(source_type="pdf")
        self.chunk_word_count = chunk_word_count
        self.overlap_ratio = overlap_ratio

    def extract(self, file_path: str | Path) -> list[ExtractedChunk]:
        reader = PdfReader(str(file_path))

        words = []
        for page in reader.pages:
            text = page.extract_text() or ""
            words.extend(text.split())
        
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
        reader = PdfReader(file_path)
        metadata = reader.metadata or {}
        return {
            "page_count": len(reader.pages),
            "title": getattr(metadata, "/Title", None),
            "author": getattr(metadata, "/Author", None),
            "subject": getattr(metadata, "/Subject", None),
        } | self._get_base_meta_data(file_path)
