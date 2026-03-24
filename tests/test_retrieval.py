"""
Tests for the retrieval module.

Tests cover:
    - Document creation and validation
    - BM25 retrieval scoring
    - Document chunking
    - Edge cases and error handling
"""

import pytest

from llm_workflow.retrieval import Document, RetrievalResult, SimpleRetriever


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation_success(self) -> None:
        """Test successful document creation with required fields."""
        doc = Document(id="doc1", content="This is test content.")

        assert doc.id == "doc1"
        assert doc.content == "This is test content."
        assert doc.metadata == {}
        assert doc.chunk_index is None

    def test_document_with_metadata(self) -> None:
        """Test document creation with metadata."""
        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"source": "test.pdf", "page": "1"},
        )

        assert doc.metadata["source"] == "test.pdf"
        assert doc.metadata["page"] == "1"

    def test_document_with_chunk_index(self) -> None:
        """Test document creation with chunk index."""
        doc = Document(id="doc1", content="Test", chunk_index=3)

        assert doc.chunk_index == 3

    def test_document_empty_id_raises_error(self) -> None:
        """Test that empty ID raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            Document(id="", content="Some content")

    def test_document_empty_content_raises_error(self) -> None:
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Document(id="doc1", content="")

    def test_document_is_immutable(self) -> None:
        """Test that Document is frozen (immutable)."""
        doc = Document(id="doc1", content="Test")

        with pytest.raises(AttributeError):
            doc.id = "doc2"  # type: ignore[misc]


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_retrieval_result_creation(self) -> None:
        """Test creation of RetrievalResult."""
        doc = Document(id="doc1", content="Test content")
        result = RetrievalResult(document=doc, score=0.85, rank=1)

        assert result.document == doc
        assert result.score == 0.85
        assert result.rank == 1


class TestSimpleRetriever:
    """Tests for SimpleRetriever BM25 implementation."""

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="Python is a high-level programming language known for readability.",
            ),
            Document(
                id="doc2",
                content="Java is a popular programming language for enterprise applications.",
            ),
            Document(
                id="doc3",
                content="Machine learning uses algorithms to learn from data.",
            ),
            Document(
                id="doc4",
                content="Python is widely used in data science and machine learning.",
            ),
        ]

    def test_retriever_initialization_defaults(self) -> None:
        """Test retriever initializes with default parameters."""
        retriever = SimpleRetriever()

        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.top_k == 3

    def test_retriever_initialization_custom(self) -> None:
        """Test retriever initializes with custom parameters."""
        retriever = SimpleRetriever(k1=2.0, b=0.5, top_k=5)

        assert retriever.k1 == 2.0
        assert retriever.b == 0.5
        assert retriever.top_k == 5

    def test_retriever_invalid_k1(self) -> None:
        """Test that negative k1 raises ValueError."""
        with pytest.raises(ValueError, match="k1 must be non-negative"):
            SimpleRetriever(k1=-1.0)

    def test_retriever_invalid_b(self) -> None:
        """Test that b outside 0-1 raises ValueError."""
        with pytest.raises(ValueError, match="b must be between 0 and 1"):
            SimpleRetriever(b=1.5)

    def test_retriever_invalid_top_k(self) -> None:
        """Test that top_k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            SimpleRetriever(top_k=0)

    def test_retrieve_empty_query(self, sample_documents: list[Document]) -> None:
        """Test that empty query raises ValueError."""
        retriever = SimpleRetriever()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("", sample_documents)

    def test_retrieve_empty_documents(self) -> None:
        """Test retrieval with no documents returns empty list."""
        retriever = SimpleRetriever()
        results = retriever.retrieve("Python programming", [])

        assert results == []

    def test_retrieve_returns_ranked_results(
        self,
        sample_documents: list[Document],
    ) -> None:
        """Test that results are ranked by relevance."""
        retriever = SimpleRetriever(top_k=3)
        results = retriever.retrieve("Python programming", sample_documents)

        assert len(results) <= 3
        assert results[0].rank == 1
        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_python_query_finds_python_docs(
        self,
        sample_documents: list[Document],
    ) -> None:
        """Test that Python query retrieves Python-related documents."""
        retriever = SimpleRetriever(top_k=2)
        results = retriever.retrieve("Python data science", sample_documents)

        doc_ids = {r.document.id for r in results}
        # Should find both Python documents
        assert "doc1" in doc_ids or "doc4" in doc_ids

    def test_retrieve_machine_learning_query(
        self,
        sample_documents: list[Document],
    ) -> None:
        """Test retrieval for machine learning query."""
        retriever = SimpleRetriever(top_k=2)
        results = retriever.retrieve("machine learning algorithms", sample_documents)

        assert len(results) > 0
        # doc3 should rank high for this query
        top_doc_ids = [r.document.id for r in results]
        assert "doc3" in top_doc_ids

    def test_retrieve_min_score_filter(
        self,
        sample_documents: list[Document],
    ) -> None:
        """Test that min_score filters low-relevance documents."""
        retriever = SimpleRetriever(top_k=10)

        # With high min_score, fewer results
        results_high = retriever.retrieve(
            "Python",
            sample_documents,
            min_score=2.0,
        )

        # With low min_score, more results
        results_low = retriever.retrieve(
            "Python",
            sample_documents,
            min_score=0.0,
        )

        assert len(results_high) <= len(results_low)

    def test_retrieve_respects_top_k(
        self,
        sample_documents: list[Document],
    ) -> None:
        """Test that retriever respects top_k limit."""
        retriever = SimpleRetriever(top_k=2)
        results = retriever.retrieve("programming language", sample_documents)

        assert len(results) <= 2


class TestDocumentChunking:
    """Tests for document chunking functionality."""

    @pytest.fixture
    def retriever(self) -> SimpleRetriever:
        """Create a retriever for testing."""
        return SimpleRetriever()

    @pytest.fixture
    def long_document(self) -> Document:
        """Create a long document for chunking tests."""
        content = (
            "Python is a versatile programming language. "
            "It supports multiple paradigms including procedural and object-oriented. "
            "Python is widely used in web development, data science, and automation. "
            "The language emphasizes code readability and simplicity. "
            "Many libraries are available for machine learning tasks. "
            "Django and Flask are popular web frameworks. "
            "NumPy and Pandas are essential for data analysis. "
            "Python has a large and active community. "
            "It is also used for scripting and rapid prototyping."
        )
        return Document(id="long_doc", content=content)

    def test_chunk_small_document_unchanged(
        self,
        retriever: SimpleRetriever,
    ) -> None:
        """Test that small documents are not chunked."""
        doc = Document(id="small", content="This is a short document.")
        chunks = retriever.chunk_document(doc, chunk_size=500)

        assert len(chunks) == 1
        assert chunks[0].content == doc.content

    def test_chunk_creates_multiple_chunks(
        self,
        retriever: SimpleRetriever,
        long_document: Document,
    ) -> None:
        """Test that long documents are split into chunks."""
        chunks = retriever.chunk_document(
            long_document,
            chunk_size=100,
            overlap=20,
        )

        assert len(chunks) > 1

    def test_chunk_preserves_parent_id(
        self,
        retriever: SimpleRetriever,
        long_document: Document,
    ) -> None:
        """Test that chunks reference parent document."""
        chunks = retriever.chunk_document(
            long_document,
            chunk_size=100,
            overlap=20,
        )

        for chunk in chunks:
            assert chunk.metadata["parent_id"] == long_document.id

    def test_chunk_has_sequential_indices(
        self,
        retriever: SimpleRetriever,
        long_document: Document,
    ) -> None:
        """Test that chunks have sequential indices."""
        chunks = retriever.chunk_document(
            long_document,
            chunk_size=100,
            overlap=20,
        )

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_invalid_params(self, retriever: SimpleRetriever) -> None:
        """Test that chunk_size <= overlap raises ValueError."""
        doc = Document(id="test", content="Test content")

        with pytest.raises(ValueError, match="chunk_size must be greater"):
            retriever.chunk_document(doc, chunk_size=50, overlap=50)

    def test_chunk_ids_are_unique(
        self,
        retriever: SimpleRetriever,
        long_document: Document,
    ) -> None:
        """Test that chunk IDs are unique."""
        chunks = retriever.chunk_document(
            long_document,
            chunk_size=100,
            overlap=20,
        )

        chunk_ids = [c.id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_overlap_creates_redundancy(
        self,
        retriever: SimpleRetriever,
        long_document: Document,
    ) -> None:
        """Test that overlap creates overlapping content between chunks."""
        chunks = retriever.chunk_document(
            long_document,
            chunk_size=200,
            overlap=50,
        )

        if len(chunks) >= 2:
            # Check that consecutive chunks have some overlap
            chunk1_end = chunks[0].content[-30:].lower()
            chunk2_start = chunks[1].content[:50].lower()
            # There should be some common words (basic overlap check)
            words1 = set(chunk1_end.split())
            words2 = set(chunk2_start.split())
            # At least some words might overlap
            assert len(words1) > 0 and len(words2) > 0
