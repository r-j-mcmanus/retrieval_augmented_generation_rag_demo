from pathlib import Path
from typing import Any
from dataclasses import dataclass

import webvtt

from .base import BaseDocumentExtractor, ExtractedChunk


@dataclass(frozen=True)
class _SpeakerRange:
    start: int
    end: int
    speaker: str
    
@dataclass(frozen=True)
class _DecoratedWord:
    word: str
    start: str
    end: str
    voice: str


class VTTExtractor(BaseDocumentExtractor):
    def __init__(self):
        super().__init__(source_type="vtt")

    def extract(self, file_path: str | Path) -> list[ExtractedChunk]:
        decorated_words = self._get_decorated_words(file_path)

        if not decorated_words:
            return []

        text, speaker_ranges = self._get_text(decorated_words)

        document = self._chunk_pipeline.run(text)
        assert not isinstance(document, list)

        chunks = self._get_chunks(document, speaker_ranges)
        return chunks

    def _get_decorated_words(self, file_path) -> list[_DecoratedWord]:
        decorated_words: list[_DecoratedWord] = []
        
        for cue in webvtt.read(str(file_path)):
            clean_text = " ".join(str(cue.text).split())
            if not clean_text:
                continue

            voice = getattr(cue, "voice") or "Unknown"
            words = clean_text.split()

            for word in words:
                decorated_words.append(_DecoratedWord(
                    word = word,
                    start = cue.start,
                    end = cue.end,
                    voice = voice
                ))

        return decorated_words

    def _get_text(self, decorated_words: list[_DecoratedWord]) -> tuple[str, list[_SpeakerRange]]:

        transcript_words: list[str] = []
        speaker_ranges: list[_SpeakerRange] = []
        current_voice: str | None = None
        current_voice_start = 0
        text_position = 0

        for item in decorated_words:
            voice = item.voice
            if voice != current_voice:
                if current_voice is not None:
                    speaker_ranges.append(
                        _SpeakerRange(current_voice_start, text_position, current_voice)
                    )
                current_voice = voice
                current_voice_start = text_position

            word = item.word
            transcript_words.append(word)
            text_position += len(word) + 1

        text = " ".join(transcript_words)
        speaker_ranges.append(
            _SpeakerRange(current_voice_start, len(text), current_voice or "Unknown")
        )

        return text, speaker_ranges

    def _get_chunks(
        self,
        document,
        speaker_ranges: list[_SpeakerRange],
    ) -> list[ExtractedChunk]:
        chunks: list[ExtractedChunk] = []
        for chunk in document.chunks:
            speaker = next(
                speaker_range.speaker
                for speaker_range in speaker_ranges
                if speaker_range.start <= chunk.start_index < speaker_range.end
            )
            chunks.append(
                ExtractedChunk(
                    content=f"**{speaker}:** {chunk.text}",
                    source_type=self.source_type,
                    metadata={"speaker": speaker},
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
