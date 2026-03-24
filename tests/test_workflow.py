"""
Tests for the workflow module.

Tests cover:
    - End-to-end workflow execution
    - Mock LLM client behavior
    - Document normalization
    - Error handling
    - Configuration options
"""

import pytest

from llm_workflow.retrieval import Document
from llm_workflow.workflow import (
    DocumentQAWorkflow,
    MockLLMClient,
    WorkflowConfig,
    WorkflowResult,
)


class TestWorkflowConfig:
    """Tests for WorkflowConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = WorkflowConfig()

        assert config.top_k == 3
        assert config.min_retrieval_score == 0.1
        assert config.require_citations is True
        assert config.max_response_length == 500
        assert config.temperature == 0.3

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = WorkflowConfig(
            top_k=5,
            min_retrieval_score=0.5,
            require_citations=False,
            temperature=0.7,
        )

        assert config.top_k == 5
        assert config.min_retrieval_score == 0.5
        assert config.require_citations is False
        assert config.temperature == 0.7


class TestWorkflowResult:
    """Tests for WorkflowResult."""

    def test_result_has_citations_true(self) -> None:
        """Test has_citations when citations exist."""
        from llm_workflow.citations import CitationExtractionResult

        result = WorkflowResult(
            answer="Test [1]",
            citations=CitationExtractionResult(valid_count=1),
            success=True,
        )

        assert result.has_citations is True

    def test_result_has_citations_false(self) -> None:
        """Test has_citations when no citations."""
        from llm_workflow.citations import CitationExtractionResult

        result = WorkflowResult(
            answer="Test without citations",
            citations=CitationExtractionResult(valid_count=0),
            success=True,
        )

        assert result.has_citations is False

    def test_citation_quality(self) -> None:
        """Test citation_quality calculation."""
        from llm_workflow.citations import CitationExtractionResult

        result = WorkflowResult(
            answer="Test [1] [2]",
            citations=CitationExtractionResult(valid_count=2, invalid_count=1),
            success=True,
        )

        assert result.citation_quality == pytest.approx(2 / 3)


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    def test_mock_returns_configured_response(self) -> None:
        """Test mock client returns configured response."""
        client = MockLLMClient(response="Custom response [1].")
        result = client.complete([{"role": "user", "content": "Test"}])

        assert result == "Custom response [1]."

    def test_mock_tracks_call_count(self) -> None:
        """Test mock client tracks call count."""
        client = MockLLMClient()
        assert client.call_count == 0

        client.complete([])
        assert client.call_count == 1

        client.complete([])
        assert client.call_count == 2

    def test_mock_stores_last_messages(self) -> None:
        """Test mock client stores last messages."""
        client = MockLLMClient()
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ]

        client.complete(messages)

        assert client.last_messages == messages

    def test_mock_raises_exception_when_configured(self) -> None:
        """Test mock client raises exception when configured."""
        client = MockLLMClient(raise_exception=True)

        with pytest.raises(RuntimeError, match="Mock LLM error"):
            client.complete([])


class TestDocumentQAWorkflow:
    """Tests for DocumentQAWorkflow."""

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create a mock LLM client."""
        return MockLLMClient(response="Paris is the capital of France [1]. It has many landmarks [2].")

    @pytest.fixture
    def documents(self) -> list[dict]:
        """Sample documents for testing."""
        return [
            {
                "id": "doc1",
                "content": "Paris is the capital and largest city of France.",
            },
            {
                "id": "doc2",
                "content": "Paris has many famous landmarks including the Eiffel Tower.",
            },
            {
                "id": "doc3",
                "content": "Berlin is the capital of Germany.",
            },
        ]

    def test_workflow_initialization(self, mock_client: MockLLMClient) -> None:
        """Test workflow initialization."""
        workflow = DocumentQAWorkflow(llm_client=mock_client)

        assert workflow.llm_client == mock_client
        assert workflow.config.top_k == 3

    def test_workflow_run_success(
        self,
        mock_client: MockLLMClient,
        documents: list[dict],
    ) -> None:
        """Test successful workflow execution."""
        workflow = DocumentQAWorkflow(llm_client=mock_client)
        result = workflow.run(
            question="What is the capital of France?",
            documents=documents,
        )

        assert result.success is True
        assert result.error is None
        assert "Paris" in result.answer
        assert result.has_citations is True

    def test_workflow_normalizes_dict_documents(
        self,
        mock_client: MockLLMClient,
    ) -> None:
        """Test that dict documents are normalized to Document objects."""
        workflow = DocumentQAWorkflow(llm_client=mock_client)
        result = workflow.run(
            question="Test question",
            documents=[
                {"id": "test1", "content": "Test content here."},
            ],
        )

        assert result.success is True
        assert len(result.retrieved_documents) > 0

    def test_workflow_accepts_document_objects(
        self,
        mock_client: MockLLMClient,
    ) -> None:
        """Test that Document objects are accepted directly."""
        workflow = DocumentQAWorkflow(llm_client=mock_client)
        documents = [
            Document(id="doc1", content="Test document content."),
        ]

        result = workflow.run(
            question="Test question",
            documents=documents,
        )

        assert result.success is True

    def test_workflow_no_documents(self, mock_client: MockLLMClient) -> None:
        """Test workflow with no documents."""
        workflow = DocumentQAWorkflow(llm_client=mock_client)
        result = workflow.run(
            question="What is Python?",
            documents=[],
        )

        assert result.success is False
        assert result.error == "no-documents"

    def test_workflow_no_relevant_documents(
        self,
        mock_client: MockLLMClient,
    ) -> None:
        """Test workflow when query doesn't match any documents."""
        workflow = DocumentQAWorkflow(
            llm_client=mock_client,
            config=WorkflowConfig(min_retrieval_score=100.0),  # Very high threshold
        )
        documents = [
            {"id": "doc1", "content": "Completely unrelated content."},
        ]

        result = workflow.run(
            question="What is the capital of France?",
            documents=documents,
        )

        assert result.success is True
        assert "could not find" in result.answer.lower()

    def test_workflow_handles_llm_error(
        self,
        documents: list[dict],
    ) -> None:
        """Test workflow handles LLM errors gracefully."""
        error_client = MockLLMClient(raise_exception=True)
        workflow = DocumentQAWorkflow(llm_client=error_client)

        result = workflow.run(
            question="What is the capital of France?",
            documents=documents,
        )

        assert result.success is False
        assert result.error is not None

    def test_workflow_with_chunking(
        self,
        mock_client: MockLLMClient,
    ) -> None:
        """Test workflow with document chunking enabled."""
        long_content = "Python is a programming language. " * 50
        documents = [
            {"id": "long_doc", "content": long_content},
        ]

        workflow = DocumentQAWorkflow(
            llm_client=mock_client,
            config=WorkflowConfig(chunk_size=200, chunk_overlap=50),
        )

        result = workflow.run(
            question="What is Python?",
            documents=documents,
            chunk_documents=True,
        )

        assert result.success is True

    def test_workflow_respects_top_k(
        self,
        mock_client: MockLLMClient,
        documents: list[dict],
    ) -> None:
        """Test that workflow respects top_k configuration."""
        workflow = DocumentQAWorkflow(
            llm_client=mock_client,
            config=WorkflowConfig(top_k=1),
        )

        result = workflow.run(
            question="What is Paris?",
            documents=documents,
        )

        assert len(result.retrieved_documents) <= 1

    def test_workflow_extracts_citations(
        self,
        mock_client: MockLLMClient,
        documents: list[dict],
    ) -> None:
        """Test that workflow extracts and validates citations."""
        workflow = DocumentQAWorkflow(llm_client=mock_client)
        result = workflow.run(
            question="What is the capital of France?",
            documents=documents,
        )

        assert result.citations.valid_count >= 0
        assert len(result.citations.citations) > 0

    def test_workflow_custom_temperature(
        self,
        mock_client: MockLLMClient,
        documents: list[dict],
    ) -> None:
        """Test workflow with custom temperature."""
        workflow = DocumentQAWorkflow(
            llm_client=mock_client,
            config=WorkflowConfig(temperature=0.9),
        )

        result = workflow.run(
            question="Test question",
            documents=documents,
        )

        assert result.success is True


class TestPromptIntegration:
    """Tests for prompt building integration."""

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create mock client that returns what was asked."""
        return MockLLMClient(response="Response based on documents [1].")

    def test_prompts_include_documents(self, mock_client: MockLLMClient) -> None:
        """Test that prompts include document content."""
        workflow = DocumentQAWorkflow(llm_client=mock_client)
        documents = [
            {"id": "doc1", "content": "Unique test content ABC123."},
        ]

        workflow.run(
            question="What is the content?",
            documents=documents,
        )

        # Check that documents were included in the messages
        messages = mock_client.last_messages
        user_message = next(m for m in messages if m["role"] == "user")
        assert "Unique test content ABC123" in user_message["content"]

    def test_prompts_include_question(self, mock_client: MockLLMClient) -> None:
        """Test that prompts include the question."""
        workflow = DocumentQAWorkflow(llm_client=mock_client)

        workflow.run(
            question="What is the meaning of XYZ789?",
            documents=[{"id": "d1", "content": "XYZ789 means something."}],
        )

        messages = mock_client.last_messages
        user_message = next(m for m in messages if m["role"] == "user")
        assert "XYZ789" in user_message["content"]

    def test_prompts_have_system_message(self, mock_client: MockLLMClient) -> None:
        """Test that prompts include system message."""
        workflow = DocumentQAWorkflow(llm_client=mock_client)

        workflow.run(
            question="Test?",
            documents=[{"id": "d1", "content": "Test content."}],
        )

        messages = mock_client.last_messages
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles
