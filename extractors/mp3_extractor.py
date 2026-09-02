from pathlib import Path
from typing import Any, Literal

from faster_whisper import WhisperModel

from .base import BaseDocumentExtractor, ExtractedChunk


class MP3Extractor(BaseDocumentExtractor):
    def __init__(self, chunk_word_count = 100, overlap_ratio = 0.2):
        super().__init__(source_type="mp3")
        self.chunk_word_count = chunk_word_count
        self.overlap_ratio = overlap_ratio
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
        
        step_size = max(1, int(self.chunk_word_count * (1.0 - self.overlap_ratio)))
        chunks: list[ExtractedChunk] = []
        for i in range(0, len(decorated_words), step_size):
            window = decorated_words[i : i + self.chunk_word_count]

            if not window:
                break

            combined_text = " ".join([decorated_word["word"] for decorated_word in window]) # type: ignore
            start_time = window[0]["start"]

            chunks.append(
                ExtractedChunk(
                    content=combined_text,
                    locator={
                        "start": start_time,
                        "word_count": len(window),
                    },
                    source_type=self.source_type
                )
            )
        return chunks


    def get_useful_metadata(self, file_path: str | Path) -> dict[str, Any]:
        return {} | self._get_base_meta_data(Path(file_path))
