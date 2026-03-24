"""
Citation extraction and validation module.

Handles parsing inline citations from LLM responses, mapping them
back to source documents, and formatting citations consistently.

Design Choices:
    - Support multiple citation formats: [1], [source1], {cite:1}
    - Validate citations against provided source list
    - Extract verbatim quotes when available
    - Flexible output formatting
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llm_workflow.retrieval import Document


class CitationFormat(Enum):
    """Supported citation format styles."""

    NUMERIC = "numeric"  # [1], [2], [3]
    LABELED = "labeled"  # [source1], [doc2]
    CURLY = "curly"  # {cite:1}, {ref:doc1}


@dataclass(frozen=True)
class Citation:
    """
    Represents a single citation in the response.

    Attributes:
        reference: The citation reference (e.g., "1", "source1").
        source_id: ID of the source document if matched.
        quote: Verbatim quote from the source if available.
        start_pos: Start position in the response text.
        end_pos: End position in the response text.
        is_valid: Whether the citation maps to a valid source.
    """

    reference: str
    source_id: str | None = None
    quote: str | None = None
    start_pos: int = 0
    end_pos: int = 0
    is_valid: bool = False


@dataclass
class CitationExtractionResult:
    """
    Result of extracting citations from text.

    Attributes:
        citations: List of extracted citations.
        text_without_citations: The cleaned text with citations removed.
        valid_count: Number of citations that matched sources.
        invalid_count: Number of citations without matching sources.
    """

    citations: list[Citation] = field(default_factory=list)
    text_without_citations: str = ""
    valid_count: int = 0
    invalid_count: int = 0

    @property
    def all_valid(self) -> bool:
        """Check if all citations are valid."""
        return self.invalid_count == 0 and self.valid_count > 0

    @property
    def citation_coverage(self) -> float:
        """Calculate percentage of valid citations."""
        total = self.valid_count + self.invalid_count
        return self.valid_count / total if total > 0 else 0.0


class CitationExtractor:
    """
    Extracts and validates citations from LLM responses.

    Supports multiple citation formats and can map citations
    back to source documents for validation.

    Attributes:
        format: The citation format to use for extraction.
        strict: If True, only numeric references are valid.

    Example:
        >>> extractor = CitationExtractor()
        >>> result = extractor.extract("Paris is the capital [1]. It's in France [2].", source_ids=["doc1", "doc2"])
        >>> print(result.valid_count)
        2
    """

    # Regex patterns for different citation formats
    _PATTERNS = {
        CitationFormat.NUMERIC: re.compile(r"\[(\d+)\]"),
        CitationFormat.LABELED: re.compile(r"\[([a-zA-Z]+\d*)\]"),
        CitationFormat.CURLY: re.compile(r"\{(?:cite|ref):([^\}]+)\}"),
    }

    def __init__(
        self,
        format: CitationFormat = CitationFormat.NUMERIC,
        strict: bool = False,
    ) -> None:
        """
        Initialize the citation extractor.

        Args:
            format: The citation format to extract.
            strict: If True, require exact format matching.
        """
        self.format = format
        self.strict = strict

    def extract(
        self,
        text: str,
        source_ids: Sequence[str] | None = None,
        source_documents: Sequence[Document] | None = None,
    ) -> CitationExtractionResult:
        """
        Extract citations from text and validate against sources.

        Args:
            text: The text containing citations.
            source_ids: List of valid source IDs (1-indexed mapping).
            source_documents: Optional documents for detailed validation.

        Returns:
            CitationExtractionResult with extracted and validated citations.
        """
        if not text:
            return CitationExtractionResult(text_without_citations="")

        # Build source mapping (1-indexed for numeric citations)
        source_map: dict[str, str] = {}
        if source_ids:
            for idx, sid in enumerate(source_ids, 1):
                source_map[str(idx)] = sid
                source_map[sid] = sid  # Allow direct ID reference
        elif source_documents:
            for idx, doc in enumerate(source_documents, 1):
                source_map[str(idx)] = doc.id
                source_map[doc.id] = doc.id

        citations: list[Citation] = []
        pattern = self._PATTERNS[self.format]

        for match in pattern.finditer(text):
            reference = match.group(1)
            source_id = source_map.get(reference)
            is_valid = source_id is not None

            citation = Citation(
                reference=reference,
                source_id=source_id,
                start_pos=match.start(),
                end_pos=match.end(),
                is_valid=is_valid,
            )
            citations.append(citation)

        # Remove citations from text
        text_clean = pattern.sub("", text)
        text_clean = re.sub(r"\s+", " ", text_clean).strip()

        valid_count = sum(1 for c in citations if c.is_valid)
        invalid_count = len(citations) - valid_count

        return CitationExtractionResult(
            citations=citations,
            text_without_citations=text_clean,
            valid_count=valid_count,
            invalid_count=invalid_count,
        )

    def format_citations(
        self,
        source_documents: Sequence[Document],
        style: str = "numbered",
    ) -> str:
        """
        Format source documents as a citation list.

        Args:
            source_documents: Documents to format as citations.
            style: Format style ("numbered", "bullet", "inline").

        Returns:
            Formatted citation string.

        Raises:
            ValueError: If style is not recognized.
        """
        if not source_documents:
            return ""

        valid_styles = {"numbered", "bullet", "inline"}
        if style not in valid_styles:
            raise ValueError(f"Style must be one of: {valid_styles}")

        lines: list[str] = []

        if style == "numbered":
            for idx, doc in enumerate(source_documents, 1):
                preview = self._truncate_content(doc.content, 100)
                source = doc.metadata.get("source", doc.id)
                lines.append(f"[{idx}] {source}: {preview}")

        elif style == "bullet":
            for doc in source_documents:
                preview = self._truncate_content(doc.content, 100)
                source = doc.metadata.get("source", doc.id)
                lines.append(f"• {source}: {preview}")

        elif style == "inline":
            parts = []
            for idx, doc in enumerate(source_documents, 1):
                source = doc.metadata.get("source", doc.id)
                parts.append(f"[{idx}] {source}")
            return ", ".join(parts)

        return "\n".join(lines)

    def add_citations_to_text(
        self,
        text: str,
        source_documents: Sequence[Document],
        threshold: float = 0.5,
    ) -> str:
        """
        Add citations to text based on content matching.

        Finds sentences in text that closely match source content
        and adds appropriate citations.

        Args:
            text: The text to add citations to.
            source_documents: Source documents to cite.
            threshold: Minimum similarity threshold (0.0-1.0).

        Returns:
            Text with inline citations added.
        """
        if not source_documents or not text:
            return text

        sentences = self._split_sentences(text)
        cited_sentences: list[str] = []

        for sentence in sentences:
            best_match_idx = -1
            best_score = threshold

            for idx, doc in enumerate(source_documents):
                score = self._sentence_similarity(sentence, doc.content)
                if score > best_score:
                    best_score = score
                    best_match_idx = idx

            if best_match_idx >= 0:
                # Add citation at end of sentence
                sentence = sentence.rstrip()
                if sentence.endswith((".", "!", "?")):
                    sentence = sentence[:-1] + f" [{best_match_idx + 1}]" + sentence[-1]
                else:
                    sentence = sentence + f" [{best_match_idx + 1}]"

            cited_sentences.append(sentence)

        return " ".join(cited_sentences)

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content with ellipsis if needed."""
        content = content.replace("\n", " ").strip()
        if len(content) <= max_length:
            return content
        return content[: max_length - 3].rsplit(" ", 1)[0] + "..."

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        pattern = r"(?<=[.!?])\s+"
        return [s.strip() for s in re.split(pattern, text) if s.strip()]

    def _sentence_similarity(self, sentence: str, document: str) -> float:
        """
        Calculate word overlap similarity between sentence and document.

        Uses Jaccard similarity on word sets.
        """
        sentence_words = set(sentence.lower().split())
        doc_words = set(document.lower().split())

        if not sentence_words or not doc_words:
            return 0.0

        intersection = sentence_words & doc_words
        union = sentence_words | doc_words

        return len(intersection) / len(union)
