"""
Tests for the evaluation module.

Tests cover:
    - Response evaluation scoring
    - Citation quality assessment
    - Faithfulness checks
    - Relevance scoring
    - Batch evaluation
    - Quick check functionality
"""

import pytest

from llm_workflow.citations import CitationExtractionResult
from llm_workflow.evaluation import (
    EvaluationConfig,
    QualityLevel,
    ResponseEvaluator,
    evaluate_batch,
)
from llm_workflow.retrieval import Document
from llm_workflow.workflow import WorkflowResult


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = EvaluationConfig()

        assert config.min_overall_score == 0.5
        assert config.min_citation_score == 0.3
        assert config.require_citations is True

    def test_weights_sum_to_one(self) -> None:
        """Test that default weights sum to 1.0."""
        config = EvaluationConfig()
        total = config.citation_weight + config.faithfulness_weight + config.relevance_weight

        assert abs(total - 1.0) < 0.01

    def test_invalid_weights_raise_error(self) -> None:
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            EvaluationConfig(
                citation_weight=0.5,
                faithfulness_weight=0.5,
                relevance_weight=0.5,
            )


class TestQualityLevel:
    """Tests for QualityLevel enum."""

    def test_quality_levels_exist(self) -> None:
        """Test all quality levels are defined."""
        assert QualityLevel.EXCELLENT.value == "excellent"
        assert QualityLevel.GOOD.value == "good"
        assert QualityLevel.FAIR.value == "fair"
        assert QualityLevel.POOR.value == "poor"


class TestResponseEvaluator:
    """Tests for ResponseEvaluator."""

    @pytest.fixture
    def evaluator(self) -> ResponseEvaluator:
        """Create an evaluator for testing."""
        return ResponseEvaluator()

    @pytest.fixture
    def good_result(self) -> WorkflowResult:
        """Create a good workflow result."""
        return WorkflowResult(
            answer="Python is a programming language used for web development and data science [1]. It supports multiple paradigms [2].",
            citations=CitationExtractionResult(valid_count=2, invalid_count=0),
            success=True,
        )

    @pytest.fixture
    def source_documents(self) -> list[Document]:
        """Create source documents for testing."""
        return [
            Document(
                id="doc1",
                content="Python is a high-level programming language used for web development, data science, and automation.",
            ),
            Document(
                id="doc2",
                content="Python supports multiple programming paradigms including procedural and object-oriented.",
            ),
        ]

    def test_evaluate_good_response(
        self,
        evaluator: ResponseEvaluator,
        good_result: WorkflowResult,
        source_documents: list[Document],
    ) -> None:
        """Test evaluation of a high-quality response."""
        result = evaluator.evaluate(
            good_result,
            question="What is Python?",
            source_documents=source_documents,
        )

        assert result.overall_score > 0.5
        assert result.passed is True
        assert result.citation_score == 1.0  # All citations valid

    def test_evaluate_no_citations(
        self,
        evaluator: ResponseEvaluator,
        source_documents: list[Document],
    ) -> None:
        """Test evaluation when citations are missing."""
        result_no_cite = WorkflowResult(
            answer="Python is a programming language.",
            citations=CitationExtractionResult(valid_count=0, invalid_count=0),
            success=True,
        )

        eval_result = evaluator.evaluate(
            result_no_cite,
            question="What is Python?",
            source_documents=source_documents,
        )

        assert eval_result.citation_score == 0.0
        assert "No citations provided" in eval_result.issues

    def test_evaluate_invalid_citations(
        self,
        evaluator: ResponseEvaluator,
        source_documents: list[Document],
    ) -> None:
        """Test evaluation with invalid citations."""
        result_invalid = WorkflowResult(
            answer="Answer with bad citation [5].",
            citations=CitationExtractionResult(valid_count=0, invalid_count=1),
            success=True,
        )

        eval_result = evaluator.evaluate(
            result_invalid,
            question="Test question?",
            source_documents=source_documents,
        )

        assert eval_result.citation_score < 1.0
        assert "non-existent sources" in str(eval_result.issues)

    def test_evaluate_quality_levels(
        self,
        evaluator: ResponseEvaluator,
    ) -> None:
        """Test quality level assignment based on scores."""
        # Excellent (>= 0.8)
        result_excellent = WorkflowResult(
            answer="Python is a great programming language for data science and machine learning [1]. It has many libraries [2].",
            citations=CitationExtractionResult(valid_count=2, invalid_count=0),
            success=True,
        )
        docs = [
            Document(id="d1", content="Python is great for data science and machine learning."),
            Document(id="d2", content="Python has many libraries available."),
        ]
        eval_result = evaluator.evaluate(
            result_excellent,
            question="What is Python good for?",
            source_documents=docs,
        )

        assert eval_result.quality_level in [
            QualityLevel.EXCELLENT,
            QualityLevel.GOOD,
        ]

    def test_evaluate_poor_quality(self, evaluator: ResponseEvaluator) -> None:
        """Test detection of poor quality response."""
        result_poor = WorkflowResult(
            answer="X.",  # Very short, no citations
            citations=CitationExtractionResult(valid_count=0, invalid_count=0),
            success=True,
        )

        eval_result = evaluator.evaluate(
            result_poor,
            question="What is the capital of France?",
            source_documents=[Document(id="d1", content="Paris is the capital of France.")],
        )

        assert eval_result.quality_level in [QualityLevel.POOR, QualityLevel.FAIR]
        assert eval_result.passed is False

    def test_evaluate_extracts_documents_from_result(
        self,
        evaluator: ResponseEvaluator,
    ) -> None:
        """Test that evaluator can use documents from result."""
        from llm_workflow.retrieval import RetrievalResult

        doc = Document(id="d1", content="Paris is the capital of France.")
        result = WorkflowResult(
            answer="Paris is the capital of France [1].",
            citations=CitationExtractionResult(valid_count=1, invalid_count=0),
            retrieved_documents=[RetrievalResult(document=doc, score=1.0, rank=1)],
            success=True,
        )

        # Don't pass source_documents, let it extract from result
        eval_result = evaluator.evaluate(
            result,
            question="What is the capital of France?",
        )

        assert eval_result.faithfulness_score > 0


class TestQuickCheck:
    """Tests for quick_check method."""

    @pytest.fixture
    def evaluator(self) -> ResponseEvaluator:
        """Create evaluator for testing."""
        return ResponseEvaluator()

    def test_quick_check_success(self, evaluator: ResponseEvaluator) -> None:
        """Test quick check passes for good response."""
        result = WorkflowResult(
            answer="Python is a programming language [1].",
            citations=CitationExtractionResult(valid_count=1, invalid_count=0),
            success=True,
        )

        assert evaluator.quick_check(result, "What is Python?") is True

    def test_quick_check_fails_empty_answer(
        self,
        evaluator: ResponseEvaluator,
    ) -> None:
        """Test quick check fails for empty answer."""
        result = WorkflowResult(
            answer="",
            citations=CitationExtractionResult(),
            success=True,
        )

        assert evaluator.quick_check(result, "Question?") is False

    def test_quick_check_fails_error_result(
        self,
        evaluator: ResponseEvaluator,
    ) -> None:
        """Test quick check fails for error result."""
        result = WorkflowResult(
            answer="Error occurred.",
            citations=CitationExtractionResult(),
            success=False,
            error="Some error",
        )

        assert evaluator.quick_check(result, "Question?") is False

    def test_quick_check_accepts_refusal(self, evaluator: ResponseEvaluator) -> None:
        """Test quick check accepts legitimate refusal to answer."""
        result = WorkflowResult(
            answer="I cannot answer this based on the provided documents.",
            citations=CitationExtractionResult(valid_count=0, invalid_count=0),
            success=True,
        )

        assert evaluator.quick_check(result, "Unrelated question?") is True

    def test_quick_check_fails_no_citations_when_required(
        self,
        evaluator: ResponseEvaluator,
    ) -> None:
        """Test quick check fails when citations required but missing."""
        result = WorkflowResult(
            answer="Python is a programming language. It is used for many things.",
            citations=CitationExtractionResult(valid_count=0, invalid_count=0),
            success=True,
        )

        assert evaluator.quick_check(result, "What is Python?") is False


class TestBatchEvaluation:
    """Tests for batch evaluation function."""

    def test_evaluate_batch_empty(self) -> None:
        """Test batch evaluation with empty list."""
        metrics = evaluate_batch([])

        assert metrics["count"] == 0
        assert metrics["pass_rate"] == 0.0

    def test_evaluate_batch_multiple_results(self) -> None:
        """Test batch evaluation with multiple results."""
        results = [
            (
                WorkflowResult(
                    answer="Good answer with citation [1].",
                    citations=CitationExtractionResult(valid_count=1, invalid_count=0),
                    success=True,
                ),
                "Question 1?",
            ),
            (
                WorkflowResult(
                    answer="Another good answer [1] [2].",
                    citations=CitationExtractionResult(valid_count=2, invalid_count=0),
                    success=True,
                ),
                "Question 2?",
            ),
        ]

        metrics = evaluate_batch(results)

        assert metrics["count"] == 2
        assert metrics["pass_rate"] >= 0
        assert "avg_overall_score" in metrics
        assert "avg_citation_score" in metrics

    def test_evaluate_batch_mixed_results(self) -> None:
        """Test batch evaluation with mixed quality results."""
        results = [
            (
                WorkflowResult(
                    answer="Good answer [1].",
                    citations=CitationExtractionResult(valid_count=1, invalid_count=0),
                    success=True,
                ),
                "Q1",
            ),
            (
                WorkflowResult(
                    answer="Bad.",  # Too short
                    citations=CitationExtractionResult(valid_count=0, invalid_count=0),
                    success=True,
                ),
                "Q2",
            ),
        ]

        metrics = evaluate_batch(results)

        assert metrics["count"] == 2
        assert 0 <= metrics["pass_rate"] <= 1.0


class TestFaithfulnessEvaluation:
    """Tests for faithfulness scoring."""

    @pytest.fixture
    def evaluator(self) -> ResponseEvaluator:
        """Create evaluator."""
        return ResponseEvaluator()

    def test_high_faithfulness_score(self, evaluator: ResponseEvaluator) -> None:
        """Test high faithfulness when answer matches source."""
        source = Document(
            id="d1",
            content="The Eiffel Tower is located in Paris, France. It was built in 1889.",
        )
        result = WorkflowResult(
            answer="The Eiffel Tower was built in 1889 in Paris, France [1].",
            citations=CitationExtractionResult(valid_count=1, invalid_count=0),
            success=True,
        )

        eval_result = evaluator.evaluate(
            result,
            question="When was the Eiffel Tower built?",
            source_documents=[source],
        )

        assert eval_result.faithfulness_score > 0.3

    def test_low_faithfulness_score(self, evaluator: ResponseEvaluator) -> None:
        """Test low faithfulness when answer doesn't match source."""
        source = Document(
            id="d1",
            content="The weather in London is rainy.",
        )
        result = WorkflowResult(
            answer="Python is a programming language for AI [1].",
            citations=CitationExtractionResult(valid_count=1, invalid_count=0),
            success=True,
        )

        eval_result = evaluator.evaluate(
            result,
            question="What is Python?",
            source_documents=[source],
        )

        assert eval_result.faithfulness_score < 0.5


class TestRelevanceEvaluation:
    """Tests for relevance scoring."""

    @pytest.fixture
    def evaluator(self) -> ResponseEvaluator:
        """Create evaluator."""
        return ResponseEvaluator()

    def test_relevant_answer(self, evaluator: ResponseEvaluator) -> None:
        """Test high relevance when answer addresses question."""
        result = WorkflowResult(
            answer="Python is a high-level programming language created by Guido van Rossum [1].",
            citations=CitationExtractionResult(valid_count=1, invalid_count=0),
            success=True,
        )

        eval_result = evaluator.evaluate(
            result,
            question="What is Python programming language?",
            source_documents=[Document(id="d1", content="Python was created by Guido.")],
        )

        assert eval_result.relevance_score > 0.2

    def test_irrelevant_answer(self, evaluator: ResponseEvaluator) -> None:
        """Test low relevance when answer doesn't address question."""
        result = WorkflowResult(
            answer="The weather forecast shows rain tomorrow [1].",
            citations=CitationExtractionResult(valid_count=1, invalid_count=0),
            success=True,
        )

        eval_result = evaluator.evaluate(
            result,
            question="What is Python?",
            source_documents=[Document(id="d1", content="Rain expected.")],
        )

        # Relevance should be low since answer doesn't address Python
        assert eval_result.relevance_score < 0.5
