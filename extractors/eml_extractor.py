import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser

from .base import BaseDocumentExtractor, ExtractedChunk

# 100% vibe coded, I dont stand by the reliability of it

class EMLExtractor(BaseDocumentExtractor):
    def __init__(self, chunk_word_count = 100, overlap_ratio = 0.2):
        super().__init__(source_type="eml")
        self.chunk_word_count = chunk_word_count
        self.overlap_ratio = overlap_ratio

    def _parse_email_text(self, file_path: str | Path) -> str:
        with open(file_path, "rb") as eml_file:
            message = BytesParser(policy=policy.default).parse(eml_file)

        text_parts: list[str] = []
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue

            if part.get_content_type() not in {"text/plain", "text/html"}:
                continue

            payload = part.get_payload(decode=True)
            if payload is None:
                continue

            charset = part.get_content_charset() or "utf-8"
            try:
                decoded = payload.decode(charset, errors="replace") # type: ignore
            except (LookupError, UnicodeDecodeError):
                decoded = payload.decode("utf-8", errors="replace") # type: ignore

            if part.get_content_type() == "text/html":
                decoded = BeautifulSoup(decoded, "html.parser").get_text(" ", strip=True)

            cleaned = re.sub(r"\s+", " ", decoded).strip()
            if cleaned:
                text_parts.append(cleaned)

        return "\n\n".join(text_parts)

    def extract(self, file_path: str | Path) -> list[ExtractedChunk]:
        text = self._parse_email_text(file_path)
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
        with open(file_path, "rb") as eml_file:
            message = BytesParser(policy=policy.default).parse(eml_file)

        text = self._parse_email_text(file_path)
        return {
            "subject": message.get("Subject", ""),
            "from": message.get("From", ""),
            "to": message.get("To", ""),
            "date": message.get("Date", ""),
            "line_count": len(text.splitlines()),
            "word_count": len(text.split()),
        } | self._get_base_meta_data(file_path)
