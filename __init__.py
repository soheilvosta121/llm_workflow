"""
LLM Workflow Module

A production-grade LLM workflow with minimal retrieval, inline citations,
and evaluation checks. Designed for document Q&A use cases.

Example usage:
    >>> from llm_workflow import DocumentQAWorkflow
    >>> workflow = DocumentQAWorkflow()
    >>> result = workflow.run(
    ...     question="What is the capital of France?",
    ...     documents=[{"id": "doc1", "content": "Paris is the capital of France."}],
    ... )
    >>> print(result.answer)
    Paris is the capital of France [1].
"""

from llm_workflow.workflow import DocumentQAWorkflow, WorkflowResult
from llm_workflow.retrieval import SimpleRetriever, Document, RetrievalResult
from llm_workflow.citations import CitationExtractor, Citation
from llm_workflow.evaluation import ResponseEvaluator, EvaluationResult

__all__ = [
    "DocumentQAWorkflow",
    "WorkflowResult",
    "SimpleRetriever",
    "Document",
    "RetrievalResult",
    "CitationExtractor",
    "Citation",
    "ResponseEvaluator",
    "EvaluationResult",
]

__version__ = "1.0.0"
