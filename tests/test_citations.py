"""
Tests for the citations module.

Tests cover:
    - Citation extraction from text
    - Multiple citation formats
    - Citation validation against sources
    - Citation formatting
    - Edge cases
"""

import pytest

from llm_workflow.citations import (
    Citation,
    CitationExtractionResult,
    CitationExtractor,
    CitationFormat,
)
from llm_workflow.retrieval import Document


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self) -> None:
        """Test basic citation creation."""
        citation = Citation(
            reference="1",
            source_id="doc1",
            start_pos=10,
            end_pos=13,
            is_valid=True,
        )

        assert citation.reference == "1"
        assert citation.source_id == "doc1"
        assert citation.is_valid is True

    def test_citation_defaults(self) -> None:
        """Test citation default values."""
        citation = Citation(reference="2")

        assert citation.source_id is None
        assert citation.quote is None
        assert citation.start_pos == 0
        assert citation.end_pos == 0
        assert citation.is_valid is False


class TestCitationExtractionResult:
    """Tests for CitationExtractionResult."""

    def test_all_valid_true(self) -> None:
        """Test all_valid property when all citations are valid."""
        result = CitationExtractionResult(
            citations=[
                Citation(reference="1", is_valid=True),
                Citation(reference="2", is_valid=True),
            ],
            valid_count=2,
            invalid_count=0,
        )

        assert result.all_valid is True

    def test_all_valid_false_with_invalid(self) -> None:
        """Test all_valid property with invalid citations."""
        result = CitationExtractionResult(
            citations=[
                Citation(reference="1", is_valid=True),
                Citation(reference="5", is_valid=False),
            ],
            valid_count=1,
            invalid_count=1,
        )

        assert result.all_valid is False

    def test_all_valid_false_no_citations(self) -> None:
        """Test all_valid is False when no citations."""
        result = CitationExtractionResult(citations=[], valid_count=0, invalid_count=0)

        assert result.all_valid is False

    def test_citation_coverage_full(self) -> None:
        """Test citation coverage when all are valid."""
        result = CitationExtractionResult(valid_count=5, invalid_count=0)

        assert result.citation_coverage == 1.0

    def test_citation_coverage_partial(self) -> None:
        """Test citation coverage with mixed validity."""
        result = CitationExtractionResult(valid_count=3, invalid_count=2)

        assert result.citation_coverage == 0.6

    def test_citation_coverage_none(self) -> None:
        """Test citation coverage with no citations."""
        result = CitationExtractionResult(valid_count=0, invalid_count=0)

        assert result.citation_coverage == 0.0


class TestCitationExtractor:
    """Tests for CitationExtractor."""

    @pytest.fixture
    def extractor(self) -> CitationExtractor:
        """Create a citation extractor for testing."""
        return CitationExtractor()

    @pytest.fixture
    def source_ids(self) -> list[str]:
        """Sample source IDs for validation."""
        return ["doc1", "doc2", "doc3"]

    def test_extract_numeric_citations(
        self,
        extractor: CitationExtractor,
        source_ids: list[str],
    ) -> None:
        """Test extraction of numeric citations like [1], [2]."""
        text = "Paris is the capital of France [1]. It is in Europe [2]."
        result = extractor.extract(text, source_ids=source_ids)

        assert result.valid_count == 2
        assert result.invalid_count == 0
        assert len(result.citations) == 2
        assert result.citations[0].reference == "1"
        assert result.citations[1].reference == "2"

    def test_extract_validates_against_sources(
        self,
        extractor: CitationExtractor,
        source_ids: list[str],
    ) -> None:
        """Test that citations are validated against source list."""
        text = "Valid citation [1]. Invalid citation [10]."
        result = extractor.extract(text, source_ids=source_ids)

        assert result.valid_count == 1
        assert result.invalid_count == 1
        assert result.citations[0].is_valid is True
        assert result.citations[1].is_valid is False

    def test_extract_removes_citations_from_text(
        self,
        extractor: CitationExtractor,
    ) -> None:
        """Test that citations are removed from cleaned text."""
        text = "Paris is the capital [1] of France [2]."
        result = extractor.extract(text)

        assert "[1]" not in result.text_without_citations
        assert "[2]" not in result.text_without_citations
        # Note: cleanup normalizes whitespace, may leave trailing space before period
        assert "Paris is the capital of France" in result.text_without_citations

    def test_extract_empty_text(self, extractor: CitationExtractor) -> None:
        """Test extraction from empty text."""
        result = extractor.extract("")

        assert result.citations == []
        assert result.text_without_citations == ""
        assert result.valid_count == 0

    def test_extract_no_citations(self, extractor: CitationExtractor) -> None:
        """Test extraction from text without citations."""
        text = "This is a sentence without any citations."
        result = extractor.extract(text)

        assert result.citations == []
        assert result.valid_count == 0
        assert result.text_without_citations == text

    def test_extract_multiple_same_citation(
        self,
        extractor: CitationExtractor,
        source_ids: list[str],
    ) -> None:
        """Test handling of repeated citations."""
        text = "Fact A [1]. Fact B [1]. Fact C [2]."
        result = extractor.extract(text, source_ids=source_ids)

        assert len(result.citations) == 3
        assert result.valid_count == 3

    def test_extract_with_documents(self, extractor: CitationExtractor) -> None:
        """Test extraction with Document objects."""
        documents = [
            Document(id="alpha", content="Content A"),
            Document(id="beta", content="Content B"),
        ]
        text = "Statement [1] and another [2]."
        result = extractor.extract(text, source_documents=documents)

        assert result.valid_count == 2
        assert result.citations[0].source_id == "alpha"
        assert result.citations[1].source_id == "beta"

    def test_extract_labeled_format(self) -> None:
        """Test extraction of labeled citations like [source1]."""
        extractor = CitationExtractor(format=CitationFormat.LABELED)
        text = "Information from [docA] and [docB]."
        result = extractor.extract(text, source_ids=["docA", "docB"])

        assert len(result.citations) == 2
        assert result.valid_count == 2

    def test_extract_curly_format(self) -> None:
        """Test extraction of curly brace citations."""
        extractor = CitationExtractor(format=CitationFormat.CURLY)
        text = "Data {cite:1} and more {ref:2}."
        result = extractor.extract(text, source_ids=["1", "2"])

        assert len(result.citations) == 2


class TestCitationFormatting:
    """Tests for citation formatting methods."""

    @pytest.fixture
    def extractor(self) -> CitationExtractor:
        """Create a citation extractor for testing."""
        return CitationExtractor()

    @pytest.fixture
    def documents(self) -> list[Document]:
        """Sample documents for formatting."""
        return [
            Document(
                id="doc1",
                content="This is the first document content.",
                metadata={"source": "file1.pdf"},
            ),
            Document(
                id="doc2",
                content="This is the second document content.",
                metadata={"source": "file2.pdf"},
            ),
        ]

    def test_format_citations_numbered(
        self,
        extractor: CitationExtractor,
        documents: list[Document],
    ) -> None:
        """Test numbered citation formatting."""
        formatted = extractor.format_citations(documents, style="numbered")

        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "file1.pdf" in formatted
        assert "file2.pdf" in formatted

    def test_format_citations_bullet(
        self,
        extractor: CitationExtractor,
        documents: list[Document],
    ) -> None:
        """Test bullet point citation formatting."""
        formatted = extractor.format_citations(documents, style="bullet")

        assert "•" in formatted
        assert "file1.pdf" in formatted

    def test_format_citations_inline(
        self,
        extractor: CitationExtractor,
        documents: list[Document],
    ) -> None:
        """Test inline citation formatting."""
        formatted = extractor.format_citations(documents, style="inline")

        assert "[1] file1.pdf" in formatted
        assert "[2] file2.pdf" in formatted
        assert ", " in formatted

    def test_format_citations_empty(self, extractor: CitationExtractor) -> None:
        """Test formatting empty document list."""
        formatted = extractor.format_citations([], style="numbered")

        assert formatted == ""

    def test_format_citations_invalid_style(
        self,
        extractor: CitationExtractor,
        documents: list[Document],
    ) -> None:
        """Test that invalid style raises ValueError."""
        with pytest.raises(ValueError, match="Style must be one of"):
            extractor.format_citations(documents, style="invalid")


class TestAddCitationsToText:
    """Tests for automatic citation addition."""

    @pytest.fixture
    def extractor(self) -> CitationExtractor:
        """Create a citation extractor for testing."""
        return CitationExtractor()

    def test_add_citations_matching_content(
        self,
        extractor: CitationExtractor,
    ) -> None:
        """Test adding citations when content matches."""
        documents = [
            Document(id="doc1", content="Paris is the capital of France."),
            Document(id="doc2", content="Berlin is the capital of Germany."),
        ]
        text = "Paris is the capital of France."

        result = extractor.add_citations_to_text(text, documents, threshold=0.3)

        assert "[1]" in result

    def test_add_citations_no_match(self, extractor: CitationExtractor) -> None:
        """Test no citations added when content doesn't match."""
        documents = [
            Document(id="doc1", content="Completely different content here."),
        ]
        text = "Python is a programming language."

        result = extractor.add_citations_to_text(text, documents, threshold=0.8)

        assert "[1]" not in result

    def test_add_citations_empty_text(self, extractor: CitationExtractor) -> None:
        """Test with empty text."""
        documents = [Document(id="doc1", content="Some content")]

        result = extractor.add_citations_to_text("", documents)

        assert result == ""

    def test_add_citations_empty_documents(
        self,
        extractor: CitationExtractor,
    ) -> None:
        """Test with no documents."""
        result = extractor.add_citations_to_text("Some text.", [])

        assert result == "Some text."
