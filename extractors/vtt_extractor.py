from pathlib import Path
from typing import Any

import webvtt

from .base import BaseDocumentExtractor, ExtractedChunk


class VTTExtractor(BaseDocumentExtractor):
    def __init__(self, chunk_word_count = 100, overlap_ratio = 0.2):
        super().__init__(source_type="vtt")
        self.chunk_word_count = chunk_word_count
        self.overlap_ratio = overlap_ratio

    def extract(self, file_path: str | Path) -> list[ExtractedChunk]:
        decorated_words: list[dict[str, str]] = []

        for cue in webvtt.read(str(file_path)):
            clean_text = " ".join(str(cue.text).split())
            if not clean_text:
                continue

            voice = getattr(cue, "voice") or "Unknown"
            words = clean_text.split()

            for word in words:
                decorated_words.append({
                    "word": word,
                    "start": cue.start,
                    "end": cue.end,
                    "voice": voice
                })

        if not decorated_words:
            return []

        step_size = max(1, int(self.chunk_word_count * (1.0 - self.overlap_ratio)))
        chunks: list[ExtractedChunk] = []
        for i in range(0, len(decorated_words), step_size):
            window = decorated_words[i : i + self.chunk_word_count]

            if not window:
                break

            combined_text = " ".join(item["word"] for item in window)
            start_time = window[0]["start"]
            end_time = window[-1]["end"]

            voices_list = list({item["voice"] for item in window if item["voice"]})
            voices = ", ".join(voices_list) if voices_list else "Unknown"

            chunks.append(
                ExtractedChunk(
                    content=combined_text,
                    locator={
                        "start": start_time,
                        "end": end_time,
                        "voices": voices,
                        "word_count": len(window),
                    },
                    source_type=self.source_type
                )
            )
        return chunks

    def get_useful_metadata(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        cues = list(webvtt.read(file_path)) # type: ignore
        return {
            "cue_count": len(cues),
            "first_start": str(cues[0].start) if cues else None,
            "last_end": str(cues[-1].end) if cues else None,
        } | self._get_base_meta_data(file_path)
