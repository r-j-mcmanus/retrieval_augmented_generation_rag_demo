from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from .base import BaseDocumentExtractor, ExtractedChunk


class HTMLExtractor(BaseDocumentExtractor):
    def __init__(self):
        super().__init__(source_type="html")

    def _clean_text(self, text: str) -> str:
        return " ".join(text.split())

    def _extract_content_blocks(self, file_path: str | Path) -> tuple[list[str], str]:
        file_path = Path(file_path)
        html_text = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html_text, "html.parser")

        for tag_name in ["script", "style", "noscript", "svg", "iframe"]:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        article = soup.find("article") or soup.find("main") or soup.body or soup

        title = ""
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            title = self._clean_text(title_tag.get_text(" "))

        def render_inline(node: Any) -> str:
            if not getattr(node, "name", None):
                return str(node)

            tag_name = node.name
            if tag_name in {"ul", "ol"}:
                return ""
            if tag_name == "br":
                return "\n"

            content = "".join(render_inline(child) for child in node.children)
            content = self._clean_text(content)
            if not content:
                return ""
            if tag_name in {"strong", "b"}:
                return f"**{content}**"
            if tag_name in {"em", "i"}:
                return f"*{content}*"
            if tag_name == "code":
                return f"`{content}`"
            if tag_name == "a" and node.get("href"):
                return f"[{content}]({node['href']})"
            return content

        blocks: list[str] = []
        block_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div"}
        semantic_block_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
        for node in article.find_all(list(block_tags)):
            if node.name == "div" and node.find(list(block_tags)):
                continue
            if node.find_parent(semantic_block_tags):
                continue

            text = render_inline(node)
            text = self._clean_text(text)
            if not text:
                continue
            if len(text) < 25 and node.name in {"div", "li"}:
                continue
            if node.name.startswith("h"):
                text = f"{'#' * int(node.name[1:])} {text}"
            elif node.name == "li":
                text = f"- {text}"
            blocks.append(text)

        if not blocks:
            full_text = self._clean_text(soup.get_text(" ", strip=True))
            blocks = [full_text] if full_text else []

        return blocks, title

    def extract(self, file_path: str | Path) -> list[ExtractedChunk]:
        blocks, _ = self._extract_content_blocks(file_path)
        return [
            ExtractedChunk(content=content, source_type=self.source_type)
            for content in self._chunker("\n\n".join(blocks))
        ]

    def get_useful_metadata(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        blocks, title = self._extract_content_blocks(file_path)

        return {
            "title": title or file_path.stem,
            "paragraph_count": len(blocks),
        } | self._get_base_meta_data(file_path)
