from pathlib import Path
from typing import Any

import webvtt

from .base import BaseDocumentExtractor, ExtractedChunk


class VTTExtractor(BaseDocumentExtractor):
    def __init__(self):
        super().__init__(source_type="vtt")

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

        text = " ".join(item["word"] for item in decorated_words)
        return [
            ExtractedChunk(content=content, source_type=self.source_type)
            for content in self._chunker(text)
        ]

    def get_useful_metadata(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        cues = list(webvtt.read(file_path)) # type: ignore
        return {
            "cue_count": len(cues),
            "first_start": str(cues[0].start) if cues else None,
            "last_end": str(cues[-1].end) if cues else None,
        } | self._get_base_meta_data(file_path)
