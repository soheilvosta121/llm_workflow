"""
Tests for the prompts module.

Tests cover:
    - Prompt configuration
    - System prompt generation
    - User prompt generation
    - Chat message formatting
    - Edge cases
"""

import pytest

from llm_workflow.prompts import (
    PromptBuilder,
    PromptConfig,
    create_qa_messages,
)
from llm_workflow.retrieval import Document


class TestPromptConfig:
    """Tests for PromptConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PromptConfig()

        assert config.max_response_length == 500
        assert config.require_citations is True
        assert config.allow_no_answer is True
        assert config.language == "English"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PromptConfig(
            max_response_length=200,
            require_citations=False,
            allow_no_answer=False,
            language="Spanish",
        )

        assert config.max_response_length == 200
        assert config.require_citations is False
        assert config.language == "Spanish"


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    @pytest.fixture
    def builder(self) -> PromptBuilder:
        """Create a prompt builder for testing."""
        return PromptBuilder()

    @pytest.fixture
    def documents(self) -> list[Document]:
        """Create sample documents."""
        return [
            Document(id="doc1", content="Paris is the capital of France."),
            Document(id="doc2", content="The Eiffel Tower is in Paris."),
        ]

    def test_build_prompts_returns_tuple(
        self,
        builder: PromptBuilder,
        documents: list[Document],
    ) -> None:
        """Test that build_prompts returns system and user prompts."""
        system_prompt, user_prompt = builder.build_prompts(
            question="What is the capital of France?",
            documents=documents,
        )

        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)
        assert len(system_prompt) > 0
        assert len(user_prompt) > 0

    def test_system_prompt_includes_instructions(
        self,
        builder: PromptBuilder,
        documents: list[Document],
    ) -> None:
        """Test that system prompt includes citation instructions."""
        system_prompt, _ = builder.build_prompts(
            question="Test?",
            documents=documents,
        )

        assert "citation" in system_prompt.lower()
        assert "[1]" in system_prompt or "[2]" in system_prompt

    def test_user_prompt_includes_documents(
        self,
        builder: PromptBuilder,
        documents: list[Document],
    ) -> None:
        """Test that user prompt includes document content."""
        _, user_prompt = builder.build_prompts(
            question="Test?",
            documents=documents,
        )

        assert "Paris is the capital" in user_prompt
        assert "Eiffel Tower" in user_prompt

    def test_user_prompt_includes_question(
        self,
        builder: PromptBuilder,
        documents: list[Document],
    ) -> None:
        """Test that user prompt includes the question."""
        _, user_prompt = builder.build_prompts(
            question="What is the capital of France?",
            documents=documents,
        )

        assert "What is the capital of France?" in user_prompt

    def test_empty_question_raises_error(
        self,
        builder: PromptBuilder,
        documents: list[Document],
    ) -> None:
        """Test that empty question raises ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            builder.build_prompts(question="", documents=documents)

    def test_empty_documents_raises_error(
        self,
        builder: PromptBuilder,
    ) -> None:
        """Test that empty documents raises ValueError."""
        with pytest.raises(ValueError, match="At least one document"):
            builder.build_prompts(question="Test?", documents=[])

    def test_custom_config_affects_prompt(self, documents: list[Document]) -> None:
        """Test that custom config affects generated prompts."""
        config = PromptConfig(
            max_response_length=100,
            language="Spanish",
        )
        builder = PromptBuilder(config=config)

        system_prompt, _ = builder.build_prompts(
            question="Test?",
            documents=documents,
        )

        assert "100" in system_prompt or "Spanish" in system_prompt

    def test_build_single_prompt(
        self,
        builder: PromptBuilder,
        documents: list[Document],
    ) -> None:
        """Test single combined prompt generation."""
        prompt = builder.build_single_prompt(
            question="What is Paris known for?",
            documents=documents,
        )

        assert isinstance(prompt, str)
        # Should contain both system and user parts
        assert "citation" in prompt.lower()
        assert "Paris" in prompt

    def test_long_document_truncation(self, builder: PromptBuilder) -> None:
        """Test that very long documents are truncated."""
        long_content = "x" * 3000  # Longer than 2000 char limit
        documents = [Document(id="long", content=long_content)]

        _, user_prompt = builder.build_prompts(
            question="Test?",
            documents=documents,
        )

        # Should be truncated with ellipsis
        assert "..." in user_prompt
        assert len(user_prompt) < 3000


class TestCreateQAMessages:
    """Tests for create_qa_messages helper function."""

    @pytest.fixture
    def documents(self) -> list[Document]:
        """Create sample documents."""
        return [
            Document(id="doc1", content="Sample document content."),
        ]

    def test_returns_message_list(self, documents: list[Document]) -> None:
        """Test that function returns list of messages."""
        messages = create_qa_messages(
            question="Test question?",
            documents=documents,
        )

        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_messages_have_correct_roles(self, documents: list[Document]) -> None:
        """Test that messages have system and user roles."""
        messages = create_qa_messages(
            question="Test?",
            documents=documents,
        )

        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_messages_have_content(self, documents: list[Document]) -> None:
        """Test that messages have content."""
        messages = create_qa_messages(
            question="Test?",
            documents=documents,
        )

        for message in messages:
            assert "role" in message
            assert "content" in message
            assert len(message["content"]) > 0

    def test_custom_config_passed(self, documents: list[Document]) -> None:
        """Test that custom config is used."""
        config = PromptConfig(require_citations=False)
        messages = create_qa_messages(
            question="Test?",
            documents=documents,
            config=config,
        )

        system_message = next(m for m in messages if m["role"] == "system")
        # Should have optional citation instruction
        assert "Optionally cite" in system_message["content"]

    def test_question_in_user_message(self, documents: list[Document]) -> None:
        """Test that question appears in user message."""
        messages = create_qa_messages(
            question="What is the meaning of life?",
            documents=documents,
        )

        user_message = next(m for m in messages if m["role"] == "user")
        assert "What is the meaning of life?" in user_message["content"]
