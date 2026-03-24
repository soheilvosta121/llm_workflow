"""
Prompt templates for the LLM workflow.

Contains structured prompts for document Q&A with citation requirements.
Templates are designed to elicit well-structured, cited responses.

Design Choices:
    - Jinja2-style templates for flexibility
    - Clear citation instructions to improve compliance
    - System/user role separation for chat models
    - Configurable response constraints
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llm_workflow.retrieval import Document


@dataclass(frozen=True)
class PromptConfig:
    """
    Configuration for prompt generation.

    Attributes:
        max_response_length: Soft limit for response length.
        require_citations: Whether to require inline citations.
        allow_no_answer: Allow "I don't know" responses.
        language: Response language (e.g., "English", "Spanish").
    """

    max_response_length: int = 500
    require_citations: bool = True
    allow_no_answer: bool = True
    language: str = "English"


# System prompt template
SYSTEM_PROMPT_TEMPLATE = Template("""You are a helpful assistant that answers questions based on provided documents.

IMPORTANT INSTRUCTIONS:
1. Only use information from the provided documents to answer questions.
2. ${citation_instruction}
3. ${no_answer_instruction}
4. Keep your response concise (under ${max_length} words).
5. Respond in ${language}.

Citation Format:
- Use [1], [2], etc. to cite specific documents.
- Place citations immediately after the relevant statement.
- You may cite multiple sources for a single statement: [1][2]

Example Response:
"The capital of France is Paris [1]. It has been the capital since 987 CE [2]."
""")

# User prompt template
USER_PROMPT_TEMPLATE = Template("""DOCUMENTS:
${documents}

QUESTION: ${question}

Please answer the question based only on the documents above. Remember to cite your sources using [1], [2], etc.""")

# Document formatting template
DOCUMENT_TEMPLATE = Template("""[Document ${index}]
${content}
---""")


class PromptBuilder:
    """
    Builds prompts for the document Q&A workflow.

    Generates system and user prompts with proper formatting
    and citation instructions.

    Example:
        >>> builder = PromptBuilder()
        >>> system, user = builder.build_prompts(
        ...     question="What is Python?", documents=[Document(id="1", content="Python is a language.")]
        ... )
    """

    def __init__(self, config: PromptConfig | None = None) -> None:
        """
        Initialize the prompt builder.

        Args:
            config: Prompt configuration. Uses defaults if None.
        """
        self.config = config or PromptConfig()

    def build_prompts(
        self,
        question: str,
        documents: Sequence[Document],
    ) -> tuple[str, str]:
        """
        Build system and user prompts for the Q&A task.

        Args:
            question: The user's question.
            documents: Retrieved documents to use as context.

        Returns:
            Tuple of (system_prompt, user_prompt).

        Raises:
            ValueError: If question is empty or no documents provided.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        if not documents:
            raise ValueError("At least one document is required")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, documents)

        return system_prompt, user_prompt

    def build_single_prompt(
        self,
        question: str,
        documents: Sequence[Document],
    ) -> str:
        """
        Build a single combined prompt (for non-chat models).

        Args:
            question: The user's question.
            documents: Retrieved documents to use as context.

        Returns:
            Combined prompt string.
        """
        system_prompt, user_prompt = self.build_prompts(question, documents)
        return f"{system_prompt}\n\n{user_prompt}"

    def _build_system_prompt(self) -> str:
        """Build the system prompt with configuration."""
        citation_instruction = (
            "Always cite your sources using [1], [2], etc. format"
            if self.config.require_citations
            else "Optionally cite sources using [1], [2], etc."
        )

        no_answer_instruction = (
            "If the documents don't contain enough information, say 'I cannot answer this based on the provided documents.'"
            if self.config.allow_no_answer
            else "Always provide an answer based on the documents, even if partial."
        )

        return SYSTEM_PROMPT_TEMPLATE.substitute(
            citation_instruction=citation_instruction,
            no_answer_instruction=no_answer_instruction,
            max_length=self.config.max_response_length,
            language=self.config.language,
        )

    def _build_user_prompt(
        self,
        question: str,
        documents: Sequence[Document],
    ) -> str:
        """Build the user prompt with documents and question."""
        formatted_docs = self._format_documents(documents)
        return USER_PROMPT_TEMPLATE.substitute(
            documents=formatted_docs,
            question=question.strip(),
        )

    def _format_documents(self, documents: Sequence[Document]) -> str:
        """Format documents for inclusion in prompt."""
        formatted: list[str] = []
        for idx, doc in enumerate(documents, 1):
            # Truncate very long documents
            content = doc.content
            if len(content) > 2000:
                content = content[:1997] + "..."

            formatted.append(
                DOCUMENT_TEMPLATE.substitute(
                    index=idx,
                    content=content,
                )
            )
        return "\n\n".join(formatted)


def create_qa_messages(
    question: str,
    documents: Sequence[Document],
    config: PromptConfig | None = None,
) -> list[dict[str, str]]:
    """
    Create chat messages for OpenAI-compatible APIs.

    Convenience function that returns properly formatted messages
    for chat completion APIs.

    Args:
        question: The user's question.
        documents: Retrieved documents.
        config: Optional prompt configuration.

    Returns:
        List of message dictionaries with 'role' and 'content'.

    Example:
        >>> messages = create_qa_messages(
        ...     "What is Python?", [Document(id="1", content="Python is a programming language.")]
        ... )
        >>> print(messages[0]["role"])
        'system'
    """
    builder = PromptBuilder(config)
    system_prompt, user_prompt = builder.build_prompts(question, documents)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
