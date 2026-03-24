"""
Main LLM workflow orchestration.

Combines retrieval, prompt construction, LLM invocation, and citation
extraction into a cohesive document Q&A pipeline.

Design Choices:
    - Clean separation between retrieval and generation
    - Pluggable LLM backend (defaults to Azure OpenAI)
    - Comprehensive error handling with fallbacks
    - Immutable result objects for safety
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from llm_workflow.citations import CitationExtractor, CitationExtractionResult
from llm_workflow.prompts import PromptBuilder, PromptConfig, create_qa_messages
from llm_workflow.retrieval import Document, RetrievalResult, SimpleRetriever

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generate a completion for the given messages.

        Args:
            messages: List of chat messages.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text response.
        """
        ...


@dataclass(frozen=True)
class WorkflowConfig:
    """
    Configuration for the document Q&A workflow.

    Attributes:
        top_k: Number of documents to retrieve.
        min_retrieval_score: Minimum relevance score for retrieval.
        require_citations: Require citations in responses.
        max_response_length: Maximum response length hint.
        temperature: LLM temperature (0.0-1.0).
        chunk_size: Size for document chunking.
        chunk_overlap: Overlap between chunks.
    """

    top_k: int = 3
    min_retrieval_score: float = 0.1
    require_citations: bool = True
    max_response_length: int = 500
    temperature: float = 0.3
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass(frozen=True)
class WorkflowResult:
    """
    Result of running the document Q&A workflow.

    Attributes:
        answer: The generated answer text.
        citations: Extracted citation information.
        retrieved_documents: Documents used for context.
        raw_response: Unprocessed LLM response.
        success: Whether the workflow completed successfully.
        error: Error message if success is False.
    """

    answer: str
    citations: CitationExtractionResult
    retrieved_documents: list[RetrievalResult] = field(default_factory=list)
    raw_response: str = ""
    success: bool = True
    error: str | None = None

    @property
    def has_citations(self) -> bool:
        """Check if the response contains any citations."""
        return self.citations.valid_count > 0

    @property
    def citation_quality(self) -> float:
        """Get citation coverage as a quality metric."""
        return self.citations.citation_coverage


class AzureOpenAIClient:
    """
    Azure OpenAI client for LLM completions.

    Reads configuration from environment variables:
        - AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_ENDPOINT
        - OPENAI_CHAT_MODEL (defaults to gpt-4)

    Example:
        >>> client = AzureOpenAIClient()
        >>> response = client.complete([{"role": "user", "content": "Hello!"}])
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Initialize the Azure OpenAI client.

        Args:
            api_key: Azure OpenAI API key (or env AZURE_OPENAI_API_KEY).
            endpoint: Azure endpoint URL (or env AZURE_OPENAI_ENDPOINT).
            model: Model deployment name (or env OPENAI_CHAT_MODEL).

        Raises:
            ValueError: If required configuration is missing.
        """
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4")

        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required. Set AZURE_OPENAI_API_KEY environment variable.")
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint is required. Set AZURE_OPENAI_ENDPOINT environment variable.")

        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy initialization of the OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.endpoint,
                    api_version="2024-02-01",
                )
            except ImportError as e:
                raise ImportError("openai package is required. Install with: pip install openai") from e
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """
        Generate a completion using Azure OpenAI.

        Args:
            messages: Chat messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional API parameters.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If API call fails.
        """
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.exception("Azure OpenAI API call failed")
            raise RuntimeError(f"LLM completion failed: {e}") from e


class MockLLMClient:
    """
    Mock LLM client for testing.

    Returns configurable responses without making API calls.
    Useful for unit tests and development.

    Example:
        >>> client = MockLLMClient(response="Test response [1].")
        >>> result = client.complete([{"role": "user", "content": "Test"}])
        >>> print(result)
        'Test response [1].'
    """

    def __init__(
        self,
        response: str = "This is a mock response [1].",
        raise_exception: bool = False,
    ) -> None:
        """
        Initialize the mock client.

        Args:
            response: The response to return.
            raise_exception: If True, raise an error on complete().
        """
        self.response = response
        self.raise_exception = raise_exception
        self.call_count = 0
        self.last_messages: list[dict[str, str]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Return the configured mock response."""
        self.call_count += 1
        self.last_messages = messages

        if self.raise_exception:
            raise RuntimeError("Mock LLM error")

        return self.response


class DocumentQAWorkflow:
    """
    End-to-end document Q&A workflow with retrieval and citations.

    Orchestrates:
    1. Document retrieval using BM25
    2. Prompt construction with context
    3. LLM response generation
    4. Citation extraction and validation

    Example:
        >>> workflow = DocumentQAWorkflow()
        >>> documents = [
        ...     {"id": "doc1", "content": "Paris is the capital of France."},
        ...     {"id": "doc2", "content": "Berlin is the capital of Germany."},
        ... ]
        >>> result = workflow.run(question="What is the capital of France?", documents=documents)
        >>> print(result.answer)
        'Paris is the capital of France [1].'
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: WorkflowConfig | None = None,
    ) -> None:
        """
        Initialize the workflow.

        Args:
            llm_client: LLM client for completions. If None, uses Azure OpenAI.
            config: Workflow configuration. Uses defaults if None.
        """
        self.config = config or WorkflowConfig()
        self._llm_client = llm_client
        self._retriever = SimpleRetriever(top_k=self.config.top_k)
        self._citation_extractor = CitationExtractor()
        self._prompt_builder = PromptBuilder(
            PromptConfig(
                max_response_length=self.config.max_response_length,
                require_citations=self.config.require_citations,
            )
        )

    @property
    def llm_client(self) -> LLMClient:
        """Get the LLM client, initializing if needed."""
        if self._llm_client is None:
            self._llm_client = AzureOpenAIClient()
        return self._llm_client

    def run(
        self,
        question: str,
        documents: Sequence[dict[str, Any] | Document],
        chunk_documents: bool = False,
    ) -> WorkflowResult:
        """
        Run the document Q&A workflow.

        Args:
            question: The user's question.
            documents: Documents to search (dicts or Document objects).
            chunk_documents: If True, chunk large documents before retrieval.

        Returns:
            WorkflowResult with answer, citations, and metadata.
        """
        try:
            # Normalize documents
            normalized_docs = self._normalize_documents(documents)

            if not normalized_docs:
                return WorkflowResult(
                    answer="No documents provided to search.",
                    citations=CitationExtractionResult(),
                    success=False,
                    error="no-documents",
                )

            # Optionally chunk documents
            if chunk_documents:
                normalized_docs = self._chunk_documents(normalized_docs)

            # Retrieve relevant documents
            retrieved = self._retriever.retrieve(
                query=question,
                documents=normalized_docs,
                min_score=self.config.min_retrieval_score,
            )

            if not retrieved:
                return WorkflowResult(
                    answer="I could not find relevant information in the provided documents.",
                    citations=CitationExtractionResult(),
                    success=True,
                    retrieved_documents=[],
                )

            # Build prompts
            context_docs = [r.document for r in retrieved]
            messages = create_qa_messages(
                question=question,
                documents=context_docs,
                config=PromptConfig(
                    max_response_length=self.config.max_response_length,
                    require_citations=self.config.require_citations,
                ),
            )

            # Generate response
            raw_response = self.llm_client.complete(
                messages=messages,
                temperature=self.config.temperature,
            )

            # Extract and validate citations
            source_ids = [doc.id for doc in context_docs]
            citations = self._citation_extractor.extract(
                text=raw_response,
                source_ids=source_ids,
            )

            return WorkflowResult(
                answer=raw_response,
                citations=citations,
                retrieved_documents=retrieved,
                raw_response=raw_response,
                success=True,
            )

        except Exception as e:
            logger.exception("Workflow execution failed")
            return WorkflowResult(
                answer="An error occurred while processing your question.",
                citations=CitationExtractionResult(),
                success=False,
                error=str(e),
            )

    def _normalize_documents(
        self,
        documents: Sequence[dict[str, Any] | Document],
    ) -> list[Document]:
        """Convert input documents to Document objects."""
        normalized: list[Document] = []

        for doc in documents:
            if isinstance(doc, Document):
                normalized.append(doc)
            elif isinstance(doc, dict):
                normalized.append(
                    Document(
                        id=str(doc.get("id", len(normalized) + 1)),
                        content=str(doc.get("content", "")),
                        metadata=doc.get("metadata", {}),
                    )
                )

        return normalized

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk documents if they exceed chunk_size."""
        chunked: list[Document] = []
        for doc in documents:
            if len(doc.content) > self.config.chunk_size:
                chunks = self._retriever.chunk_document(
                    doc,
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.chunk_overlap,
                )
                chunked.extend(chunks)
            else:
                chunked.append(doc)
        return chunked
