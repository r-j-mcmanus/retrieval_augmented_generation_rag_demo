from pathlib import Path
from typing import Any

from .base import BaseDocumentExtractor, ExtractedChunk


class TXTExtractor(BaseDocumentExtractor):
    def __init__(self):
        super().__init__(source_type="txt")

    def extract(self, file_path: str | Path) -> list[ExtractedChunk]:
        text = Path(file_path).read_text(encoding="utf-8")
        return [
            ExtractedChunk(content=content, source_type=self.source_type)
            for content in self._chunker(text)
        ]

    def get_useful_metadata(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        text = file_path.read_text(encoding="utf-8")
        return {
            "line_count": len(text.splitlines()),
            "word_count": len(text.split()),
        } | self._get_base_meta_data(file_path)
