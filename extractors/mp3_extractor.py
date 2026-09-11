from pathlib import Path
from typing import Any, Literal

from faster_whisper import WhisperModel

from .base import BaseDocumentExtractor, ExtractedChunk


class MP3Extractor(BaseDocumentExtractor):
    def __init__(self):
        super().__init__(source_type="mp3")
        self.model = WhisperModel("base", device="cuda", compute_type="int8")

    def extract(self, file_path: str | Path) -> list[ExtractedChunk]:
        segments, info = self.model.transcribe(str(file_path))

        decorated_words: list[dict[str, str | int | float]] = []

        for segment in segments:
            clean_text = " ".join(str(segment.text).split())
            if not clean_text:
                continue

            words = clean_text.split()

            for word in words:
                decorated_words.append({
                    "word": word,
                    "start": segment.start
                })

        if not decorated_words:
            return []
        
        text = " ".join(str(decorated_word["word"]) for decorated_word in decorated_words)
        return [
            ExtractedChunk(content=content, source_type=self.source_type)
            for content in self._chunker(text)
        ]


    def get_useful_metadata(self, file_path: str | Path) -> dict[str, Any]:
        return {} | self._get_base_meta_data(Path(file_path))
