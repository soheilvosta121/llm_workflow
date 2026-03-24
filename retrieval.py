"""
Retrieval module for document search and ranking.

This module provides a simple BM25-based retrieval system for finding
relevant document chunks given a query. It's designed to be lightweight
and production-ready with proper error handling.

Design Choices:
    - BM25 algorithm for relevance scoring (no external dependencies)
    - Configurable top-k retrieval
    - Support for document chunking with overlap
    - Immutable data classes for thread safety
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class Document:
    """
    Represents a document with metadata for retrieval.

    Attributes:
        id: Unique identifier for the document.
        content: The text content of the document.
        metadata: Optional key-value metadata (e.g., source, date).
        chunk_index: Index if this is part of a chunked document.
    """

    id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    chunk_index: int | None = None

    def __post_init__(self) -> None:
        """Validate document fields."""
        if not self.id:
            raise ValueError("Document id cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")


@dataclass(frozen=True)
class RetrievalResult:
    """
    Result of a retrieval operation.

    Attributes:
        document: The retrieved document.
        score: Relevance score (higher is more relevant).
        rank: Position in the result list (1-indexed).
    """

    document: Document
    score: float
    rank: int


class SimpleRetriever:
    """
    BM25-based document retriever.

    Implements the BM25 ranking function without external dependencies.
    Suitable for small to medium document collections (< 10k documents).

    Attributes:
        k1: Term frequency saturation parameter. Default 1.5.
        b: Document length normalization parameter. Default 0.75.
        top_k: Number of results to return. Default 3.

    Example:
        >>> retriever = SimpleRetriever(top_k=2)
        >>> docs = [
        ...     Document(id="1", content="Python is a programming language"),
        ...     Document(id="2", content="Java is also a programming language"),
        ... ]
        >>> results = retriever.retrieve("Python programming", docs)
        >>> print(results[0].document.id)
        '1'
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 3,
    ) -> None:
        """
        Initialize the retriever with BM25 parameters.

        Args:
            k1: Term frequency saturation parameter (0.0-3.0 typical).
            b: Length normalization (0.0 = no normalization, 1.0 = full).
            top_k: Maximum number of results to return.

        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.k1 = k1
        self.b = b
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        documents: Sequence[Document],
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant documents for a query.

        Args:
            query: The search query string.
            documents: Collection of documents to search.
            min_score: Minimum score threshold for results.

        Returns:
            List of RetrievalResult objects, sorted by relevance.

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Pre-compute document frequencies and lengths
        doc_tokens = [self._tokenize(doc.content) for doc in documents]
        avg_doc_len = sum(len(tokens) for tokens in doc_tokens) / len(documents)

        # Compute IDF for query terms
        idf_scores = self._compute_idf(query_terms, doc_tokens, len(documents))

        # Score each document
        scored_docs: list[tuple[Document, float]] = []
        for doc, tokens in zip(documents, doc_tokens, strict=True):
            score = self._bm25_score(query_terms, tokens, idf_scores, avg_doc_len)
            if score >= min_score:
                scored_docs.append((doc, score))

        # Sort by score (descending) and take top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[: self.top_k]

        # Build results with rank
        return [RetrievalResult(document=doc, score=score, rank=rank + 1) for rank, (doc, score) in enumerate(top_docs)]

    def chunk_document(
        self,
        document: Document,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[Document]:
        """
        Split a document into overlapping chunks.

        Args:
            document: The document to chunk.
            chunk_size: Maximum characters per chunk.
            overlap: Character overlap between chunks.

        Returns:
            List of Document objects representing chunks.

        Raises:
            ValueError: If chunk_size <= overlap.
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")

        content = document.content
        if len(content) <= chunk_size:
            return [document]

        chunks: list[Document] = []
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + chunk_size

            # Try to break at sentence or word boundary
            if end < len(content):
                # Look for sentence end
                sentence_end = content.rfind(". ", start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Fall back to word boundary
                    word_end = content.rfind(" ", start, end)
                    if word_end > start + chunk_size // 2:
                        end = word_end

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk = Document(
                    id=f"{document.id}_chunk_{chunk_idx}",
                    content=chunk_content,
                    metadata={
                        **document.metadata,
                        "parent_id": document.id,
                        "chunk_start": str(start),
                        "chunk_end": str(end),
                    },
                    chunk_index=chunk_idx,
                )
                chunks.append(chunk)
                chunk_idx += 1

            start = end - overlap

        return chunks

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into lowercase words.

        Simple tokenization: lowercase, split on non-alphanumeric,
        filter short tokens.
        """
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        # Filter stop words and very short tokens
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "and",
            "but",
            "or",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "not",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
        }
        return [t for t in tokens if t not in stop_words and len(t) > 1]

    def _compute_idf(
        self,
        query_terms: list[str],
        doc_tokens: list[list[str]],
        n_docs: int,
    ) -> dict[str, float]:
        """Compute inverse document frequency for query terms."""
        idf: dict[str, float] = {}
        for term in set(query_terms):
            doc_freq = sum(1 for tokens in doc_tokens if term in tokens)
            # BM25 IDF formula with smoothing
            idf[term] = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return idf

    def _bm25_score(
        self,
        query_terms: list[str],
        doc_tokens: list[str],
        idf_scores: dict[str, float],
        avg_doc_len: float,
    ) -> float:
        """Compute BM25 score for a document."""
        score = 0.0
        doc_len = len(doc_tokens)
        term_freqs = {}
        for token in doc_tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1

        for term in query_terms:
            if term not in term_freqs:
                continue
            tf = term_freqs[term]
            idf = idf_scores.get(term, 0)
            # BM25 term frequency component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len)
            score += idf * numerator / denominator

        return score
