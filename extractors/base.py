from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from datetime import datetime

import win32security


@dataclass
class ExtractedChunk:
    """Contains a row of data for a text chunk"""
    content: str
    locator: dict[str, Any] = field(default_factory=dict)
    entity_id: int | None = None
    source_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDocumentExtractor(ABC):
    """Interface for file-format specific extraction and metadata generation."""

    def __init__(self, source_type: str):
        self.source_type = source_type

    @abstractmethod
    def extract(self, file_path: str | Path) -> list[ExtractedChunk]:
        """Return text chunks and locator information for a given file."""

    @abstractmethod
    def get_useful_metadata(self, file_path: str | Path) -> dict[str, Any]:
        """Return metadata that is useful for this file type."""

    def _get_created_at(self, file_path: Path) -> str:
        return datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    
    def _get_file_owner_windows(self, file_path: str | Path) -> str:
        path = str(Path(file_path).resolve())
        sd = win32security.GetFileSecurity(path, win32security.OWNER_SECURITY_INFORMATION)
        owner_sid = sd.GetSecurityDescriptorOwner()
        name, domain, _type = win32security.LookupAccountSid(None, owner_sid)
        return f"{domain}\\{name}"

    def _get_base_meta_data(self, file_path: Path):
        return {
            "created_at": self._get_created_at(file_path),
            "created_by": self._get_file_owner_windows(file_path),
            "file_format": self.source_type,
        }
