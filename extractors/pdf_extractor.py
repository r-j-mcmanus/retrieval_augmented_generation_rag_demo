from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .base import BaseDocumentExtractor, ExtractedChunk


class PDFExtractor(BaseDocumentExtractor):
    def __init__(self):
        super().__init__(source_type="pdf")

    def extract(self, file_path: str | Path) -> list[ExtractedChunk]:
        reader = PdfReader(str(file_path))

        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")

        text = "\n\n".join(pages)
        return [
            ExtractedChunk(content=content, source_type=self.source_type)
            for content in self._chunker(text)
        ]

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
