import html
import re
import unicodedata


_ABBREVIATIONS = {
    "asap": "as soon as possible",
    "dept": "department",
    "e.g.": "for example",
    "etc.": "et cetera",
    "i.e.": "that is",
    "misc.": "miscellaneous",
    "no.": "number",
    "p.o.": "post office",
    "ph.": "phone",
    "approx.": "approximately",
}

_ABBREVIATION_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(item) for item in _ABBREVIATIONS) + r")(?!\w)",
    re.IGNORECASE,
)
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?])\1{2,}")

class TextPreprocessor:
    """Normalize text consistently before it is embedded or searched."""

    def __call__(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("TextPreprocessor expects a string")

        query = self._basic_clean(query)
        query = self._abbreviation_expansion(query)
        return query

    def _basic_clean(self, query: str) -> str:
        query = html.unescape(query)
        query = unicodedata.normalize("NFKC", query)
        query = query.replace("\u00a0", " ")
        query = query.replace("\u200b", "")
        query = _CONTROL_CHARACTER_PATTERN.sub(" ", query)
        query = _REPEATED_PUNCTUATION_PATTERN.sub(r"\1\1", query)
        return _WHITESPACE_PATTERN.sub(" ", query).strip().lower()

    def _abbreviation_expansion(self, query: str) -> str:
        return _ABBREVIATION_PATTERN.sub(
            lambda match: _ABBREVIATIONS[match.group(0).lower()],
            query,
        )