"""
Evaluation module for LLM response quality.

Provides lightweight evaluation metrics for citation accuracy,
response faithfulness, and answer relevance without requiring
external LLM calls.

Design Choices:
    - Rule-based evaluation (no LLM judge needed)
    - Fast enough for inline quality checks
    - Configurable thresholds for pass/fail
    - Detailed breakdown of quality dimensions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llm_workflow.citations import CitationExtractionResult
    from llm_workflow.retrieval import Document
    from llm_workflow.workflow import WorkflowResult


class QualityLevel(Enum):
    """Quality assessment levels."""

    EXCELLENT = "excellent"  # Score >= 0.8
    GOOD = "good"  # Score >= 0.6
    FAIR = "fair"  # Score >= 0.4
    POOR = "poor"  # Score < 0.4


@dataclass(frozen=True)
class EvaluationResult:
    """
    Comprehensive evaluation result.

    Attributes:
        overall_score: Combined quality score (0.0-1.0).
        citation_score: Citation accuracy score (0.0-1.0).
        faithfulness_score: Response faithfulness to sources (0.0-1.0).
        relevance_score: Answer relevance to question (0.0-1.0).
        quality_level: Overall quality classification.
        passed: Whether evaluation passed thresholds.
        issues: List of identified quality issues.
        details: Additional evaluation details.
    """

    overall_score: float
    citation_score: float
    faithfulness_score: float
    relevance_score: float
    quality_level: QualityLevel
    passed: bool
    issues: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """
    Configuration for response evaluation.

    Attributes:
        min_overall_score: Minimum overall score to pass.
        min_citation_score: Minimum citation score to pass.
        citation_weight: Weight of citation score in overall.
        faithfulness_weight: Weight of faithfulness in overall.
        relevance_weight: Weight of relevance in overall.
        require_citations: Whether citations are required.
    """

    min_overall_score: float = 0.5
    min_citation_score: float = 0.3
    citation_weight: float = 0.3
    faithfulness_weight: float = 0.4
    relevance_weight: float = 0.3
    require_citations: bool = True

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        total = self.citation_weight + self.faithfulness_weight + self.relevance_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class ResponseEvaluator:
    """
    Evaluates LLM response quality without external API calls.

    Performs rule-based assessment of:
    - Citation accuracy: Do citations match sources?
    - Faithfulness: Is response grounded in sources?
    - Relevance: Does response address the question?

    Example:
        >>> evaluator = ResponseEvaluator()
        >>> from llm_workflow import DocumentQAWorkflow
        >>> workflow = DocumentQAWorkflow()
        >>> result = workflow.run("What is Python?", documents)
        >>> eval_result = evaluator.evaluate(result, question="What is Python?", source_documents=documents)
        >>> print(eval_result.passed)
        True
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        """
        Initialize the evaluator.

        Args:
            config: Evaluation configuration. Uses defaults if None.
        """
        self.config = config or EvaluationConfig()

    def evaluate(
        self,
        result: WorkflowResult,
        question: str,
        source_documents: Sequence[Document] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a workflow result.

        Args:
            result: The workflow result to evaluate.
            question: The original question asked.
            source_documents: Source documents for faithfulness check.

        Returns:
            EvaluationResult with scores and quality assessment.
        """
        issues: list[str] = []
        details: dict[str, float] = {}

        # Extract source documents from result if not provided
        if source_documents is None and result.retrieved_documents:
            source_documents = [r.document for r in result.retrieved_documents]

        # Score citations
        citation_score, citation_issues = self._evaluate_citations(result.citations)
        issues.extend(citation_issues)
        details["citation_coverage"] = result.citations.citation_coverage

        # Score faithfulness
        faithfulness_score, faith_issues = self._evaluate_faithfulness(
            result.answer,
            source_documents or [],
        )
        issues.extend(faith_issues)

        # Score relevance
        relevance_score, rel_issues = self._evaluate_relevance(
            result.answer,
            question,
        )
        issues.extend(rel_issues)
        details["question_term_overlap"] = relevance_score

        # Calculate overall score
        overall_score = (
            self.config.citation_weight * citation_score
            + self.config.faithfulness_weight * faithfulness_score
            + self.config.relevance_weight * relevance_score
        )
        details["weighted_overall"] = overall_score

        # Determine quality level
        if overall_score >= 0.8:
            quality_level = QualityLevel.EXCELLENT
        elif overall_score >= 0.6:
            quality_level = QualityLevel.GOOD
        elif overall_score >= 0.4:
            quality_level = QualityLevel.FAIR
        else:
            quality_level = QualityLevel.POOR

        # Check pass/fail
        passed = overall_score >= self.config.min_overall_score and (
            not self.config.require_citations or citation_score >= self.config.min_citation_score
        )

        return EvaluationResult(
            overall_score=round(overall_score, 3),
            citation_score=round(citation_score, 3),
            faithfulness_score=round(faithfulness_score, 3),
            relevance_score=round(relevance_score, 3),
            quality_level=quality_level,
            passed=passed,
            issues=issues,
            details=details,
        )

    def quick_check(
        self,
        result: WorkflowResult,
        question: str,
    ) -> bool:
        """
        Perform a quick pass/fail check without detailed scoring.

        Faster than full evaluate() for runtime quality gates.

        Args:
            result: The workflow result.
            question: The original question.

        Returns:
            True if response passes basic quality checks.
        """
        # Check for error response
        if not result.success:
            return False

        # Check for empty or very short response
        if len(result.answer.strip()) < 10:
            return False

        # Check for refusal patterns (acceptable)
        refusal_patterns = [
            r"cannot answer",
            r"don't have enough information",
            r"no information",
            r"not mentioned",
        ]
        is_refusal = any(re.search(pattern, result.answer.lower()) for pattern in refusal_patterns)

        # If it's a refusal, that's okay
        if is_refusal:
            return True

        # Non-refusal should have citations if required
        if self.config.require_citations:
            if result.citations.valid_count == 0:
                return False

        # Check basic relevance (question terms in answer)
        question_terms = set(question.lower().split())
        answer_terms = set(result.answer.lower().split())
        overlap = len(question_terms & answer_terms)

        return overlap >= 1

    def _evaluate_citations(
        self,
        citations: CitationExtractionResult,
    ) -> tuple[float, list[str]]:
        """
        Evaluate citation quality.

        Returns:
            Tuple of (score, list of issues).
        """
        issues: list[str] = []

        if citations.valid_count == 0 and citations.invalid_count == 0:
            # No citations at all
            if self.config.require_citations:
                issues.append("No citations provided")
                return 0.0, issues
            return 1.0, issues  # Citations not required

        # Check for invalid citations
        if citations.invalid_count > 0:
            issues.append(f"{citations.invalid_count} citation(s) reference non-existent sources")

        # Calculate coverage
        coverage = citations.citation_coverage
        if coverage < 0.5:
            issues.append("Low citation coverage (less than 50% valid)")

        return coverage, issues

    def _evaluate_faithfulness(
        self,
        answer: str,
        source_documents: Sequence[Document],
    ) -> tuple[float, list[str]]:
        """
        Evaluate how faithful the answer is to source documents.

        Uses n-gram overlap as a proxy for faithfulness.

        Returns:
            Tuple of (score, list of issues).
        """
        issues: list[str] = []

        if not source_documents:
            # Can't evaluate without sources
            return 0.5, ["No source documents for faithfulness check"]

        # Extract content words from answer
        answer_words = self._extract_content_words(answer)
        if not answer_words:
            return 0.0, ["Answer contains no content words"]

        # Combine source content
        source_text = " ".join(doc.content for doc in source_documents)
        source_words = set(self._extract_content_words(source_text))

        # Calculate overlap
        overlap = len(answer_words & source_words)
        total = len(answer_words)

        score = overlap / total if total > 0 else 0.0

        # Generate issues
        if score < 0.3:
            issues.append("Response may contain information not in sources")
        elif score < 0.5:
            issues.append("Low overlap with source documents")

        return score, issues

    def _evaluate_relevance(
        self,
        answer: str,
        question: str,
    ) -> tuple[float, list[str]]:
        """
        Evaluate how relevant the answer is to the question.

        Uses term overlap and simple heuristics.

        Returns:
            Tuple of (score, list of issues).
        """
        issues: list[str] = []

        # Extract question terms (focus on nouns and content words)
        question_terms = self._extract_content_words(question)
        answer_terms = self._extract_content_words(answer)

        if not question_terms:
            return 0.5, ["Could not extract question terms"]

        if not answer_terms:
            return 0.0, ["Answer contains no content"]

        # Calculate overlap
        overlap = len(question_terms & answer_terms)
        max_possible = min(len(question_terms), len(answer_terms))

        score = overlap / max_possible if max_possible > 0 else 0.0

        # Boost score if answer is reasonable length
        answer_len = len(answer.split())
        if 20 <= answer_len <= 300:
            score = min(1.0, score + 0.1)
        elif answer_len < 10:
            score = max(0.0, score - 0.2)
            issues.append("Answer is very short")
        elif answer_len > 500:
            issues.append("Answer may be too verbose")

        if score < 0.3:
            issues.append("Answer may not address the question directly")

        return score, issues

    def _extract_content_words(self, text: str) -> set[str]:
        """
        Extract meaningful content words from text.

        Filters out stop words and very short words.
        """
        # Simple stop words
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
            "what",
            "which",
            "who",
            "whom",
            "how",
            "when",
            "where",
            "why",
            "all",
            "each",
            "every",
            "any",
            "some",
            "no",
            "if",
        }

        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        return {w for w in words if w not in stop_words and len(w) > 2}


def evaluate_batch(
    results: Sequence[tuple[WorkflowResult, str]],
    evaluator: ResponseEvaluator | None = None,
) -> dict[str, float]:
    """
    Evaluate a batch of results and return aggregate metrics.

    Useful for testing and monitoring.

    Args:
        results: List of (WorkflowResult, question) tuples.
        evaluator: Evaluator to use. Creates default if None.

    Returns:
        Dictionary with aggregate metrics.

    Example:
        >>> results = [(result1, "Q1"), (result2, "Q2")]
        >>> metrics = evaluate_batch(results)
        >>> print(metrics["pass_rate"])
        0.8
    """
    if not results:
        return {
            "count": 0,
            "pass_rate": 0.0,
            "avg_overall_score": 0.0,
            "avg_citation_score": 0.0,
            "avg_faithfulness_score": 0.0,
            "avg_relevance_score": 0.0,
        }

    evaluator = evaluator or ResponseEvaluator()
    evaluations = [evaluator.evaluate(r, q) for r, q in results]

    pass_count = sum(1 for e in evaluations if e.passed)

    return {
        "count": len(results),
        "pass_rate": pass_count / len(results),
        "avg_overall_score": sum(e.overall_score for e in evaluations) / len(evaluations),
        "avg_citation_score": sum(e.citation_score for e in evaluations) / len(evaluations),
        "avg_faithfulness_score": sum(e.faithfulness_score for e in evaluations) / len(evaluations),
        "avg_relevance_score": sum(e.relevance_score for e in evaluations) / len(evaluations),
    }
