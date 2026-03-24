# LLM Workflow Module

A production-grade LLM workflow module for document Q&A with minimal retrieval, inline citations, and evaluation checks.

## Overview

This module provides a complete pipeline for:
1. **Retrieval** - BM25-based document search
2. **Prompt Construction** - Structured prompts with citation instructions
3. **LLM Generation** - Azure OpenAI integration (or mock for testing)
4. **Citation Extraction** - Parse and validate inline citations
5. **Evaluation** - Rule-based quality assessment

## Quick Start

```python
from llm_workflow import DocumentQAWorkflow, MockLLMClient

# Create workflow with mock client for testing
workflow = DocumentQAWorkflow(llm_client=MockLLMClient())

# Define documents
documents = [
    {"id": "doc1", "content": "Paris is the capital of France."},
    {"id": "doc2", "content": "The Eiffel Tower is located in Paris."},
]

# Run the workflow
result = workflow.run(
    question="What is the capital of France?",
    documents=documents
)

print(result.answer)       # "Paris is the capital of France [1]."
print(result.has_citations)  # True
print(result.success)      # True
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DocumentQAWorkflow                          │
├─────────────────────────────────────────────────────────────────┤
│  Input: question + documents                                    │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────┐                          │
│  │        SimpleRetriever           │  BM25 scoring            │
│  │    (top-k relevant documents)    │  Document chunking       │
│  └──────────────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────┐                          │
│  │        PromptBuilder             │  System + User prompts   │
│  │    (citation instructions)       │  Document formatting     │
│  └──────────────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────┐                          │
│  │         LLM Client               │  Azure OpenAI or Mock    │
│  │    (chat completion API)         │  Configurable temp       │
│  └──────────────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────┐                          │
│  │      CitationExtractor           │  Parse [1], [2], etc.    │
│  │    (validate against sources)    │  Track valid/invalid     │
│  └──────────────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│  Output: WorkflowResult (answer, citations, metadata)          │
└─────────────────────────────────────────────────────────────────┘
```

## Design Choices & Trade-offs

### 1. BM25 Retrieval (No External Dependencies)

**Choice:** Implemented BM25 from scratch instead of using dense embeddings.

**Trade-offs:**
- ✅ No ML model dependencies (no GPU, no large downloads)
- ✅ Deterministic, testable results
- ✅ Fast for small-medium collections (< 10k docs)
- ❌ Less semantic understanding than embedding-based retrieval
- ❌ May miss paraphrased content

**When to change:** If semantic similarity is critical, integrate an embedding model and vector store.

### 2. Rule-Based Evaluation (No LLM Judge)

**Choice:** Evaluation uses word overlap and heuristics instead of LLM-as-judge.

**Trade-offs:**
- ✅ No API costs for evaluation
- ✅ Fast, deterministic scores
- ✅ Works offline
- ❌ Less nuanced than LLM evaluation
- ❌ May miss subtle quality issues

**When to change:** For production monitoring, consider adding an LLM judge for a sample of responses.

### 3. Immutable Data Classes

**Choice:** All data classes (`Document`, `WorkflowResult`, etc.) are frozen.

**Trade-offs:**
- ✅ Thread-safe by default
- ✅ Clear data flow (no hidden mutations)
- ✅ Easier to debug
- ❌ Slightly more memory for modifications (need new objects)

### 4. Mock LLM Client for Testing

**Choice:** Built-in `MockLLMClient` for unit tests.

**Trade-offs:**
- ✅ Tests don't require API keys
- ✅ Fast, deterministic testing
- ✅ Can simulate errors
- ❌ Won't catch real LLM behavior issues

**Recommendation:** Use mock for unit tests, real client for integration tests.

### 5. Numeric Citations [1], [2], [3]

**Choice:** Default to numeric citation format.

**Trade-offs:**
- ✅ Concise, standard format
- ✅ Easy to parse and validate
- ✅ Works well with most LLMs
- ❌ Less readable than named citations for long documents

## Module Structure

```
llm_workflow/
├── __init__.py          # Public API exports
├── retrieval.py         # Document, RetrievalResult, SimpleRetriever
├── citations.py         # Citation, CitationExtractor
├── prompts.py           # PromptConfig, PromptBuilder
├── workflow.py          # DocumentQAWorkflow, WorkflowResult
├── evaluation.py        # ResponseEvaluator, EvaluationResult
├── README.md            # This file
└── tests/
    ├── __init__.py
    ├── test_retrieval.py
    ├── test_citations.py
    ├── test_prompts.py
    ├── test_workflow.py
    └── test_evaluation.py
```

## Configuration

### Workflow Configuration

```python
from llm_workflow import DocumentQAWorkflow
from llm_workflow.workflow import WorkflowConfig

config = WorkflowConfig(
    top_k=5,                    # Number of documents to retrieve
    min_retrieval_score=0.2,    # Minimum BM25 score threshold
    require_citations=True,      # Require citations in prompt
    max_response_length=300,    # Response length hint
    temperature=0.3,            # LLM temperature
    chunk_size=500,             # Max chars per chunk
    chunk_overlap=50,           # Overlap between chunks
)

workflow = DocumentQAWorkflow(config=config)
```

### Evaluation Configuration

```python
from llm_workflow import ResponseEvaluator
from llm_workflow.evaluation import EvaluationConfig

config = EvaluationConfig(
    min_overall_score=0.5,      # Minimum score to pass
    min_citation_score=0.3,     # Minimum citation score
    citation_weight=0.3,        # Weight in overall score
    faithfulness_weight=0.4,    # Weight in overall score
    relevance_weight=0.3,       # Weight in overall score
    require_citations=True,     # Fail if no citations
)

evaluator = ResponseEvaluator(config=config)
```

## Using with Azure OpenAI

```python
import os
from llm_workflow import DocumentQAWorkflow

# Set environment variables
os.environ["AZURE_OPENAI_API_KEY"] = "your-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-endpoint.openai.azure.com/"
os.environ["OPENAI_CHAT_MODEL"] = "gpt-4"

# Create workflow (will use Azure OpenAI automatically)
workflow = DocumentQAWorkflow()

result = workflow.run(
    question="What is machine learning?",
    documents=[{"id": "ml", "content": "Machine learning is..."}]
)
```

## Evaluation Example

```python
from llm_workflow import DocumentQAWorkflow, ResponseEvaluator, MockLLMClient

# Run workflow
workflow = DocumentQAWorkflow(llm_client=MockLLMClient())
result = workflow.run(
    question="What is Python?",
    documents=[{"id": "py", "content": "Python is a programming language."}]
)

# Evaluate response
evaluator = ResponseEvaluator()
eval_result = evaluator.evaluate(
    result,
    question="What is Python?"
)

print(f"Overall Score: {eval_result.overall_score}")
print(f"Citation Score: {eval_result.citation_score}")
print(f"Quality Level: {eval_result.quality_level.value}")
print(f"Passed: {eval_result.passed}")
print(f"Issues: {eval_result.issues}")
```

## Running Tests

```bash
# Run all llm_workflow tests
pytest llm_workflow/tests/ -v

# Run specific test file
pytest llm_workflow/tests/test_retrieval.py -v

# Run with coverage
pytest llm_workflow/tests/ --cov=llm_workflow --cov-report=html
```

## Key Classes

| Class | Description |
|-------|-------------|
| `DocumentQAWorkflow` | Main orchestrator for the Q&A pipeline |
| `SimpleRetriever` | BM25-based document retrieval |
| `CitationExtractor` | Parse and validate inline citations |
| `PromptBuilder` | Construct system and user prompts |
| `ResponseEvaluator` | Assess response quality |
| `MockLLMClient` | Testing utility for deterministic responses |

## Future Improvements

1. **Embedding-based retrieval** - Add optional vector search for better semantic matching
2. **Streaming responses** - Support streaming for real-time UX
3. **Multi-turn conversation** - Extend for conversational Q&A
4. **LLM-as-judge** - Optional LLM evaluation for production monitoring
5. **Caching** - Cache retrieval and LLM results for repeated queries

## License

Internal use only. Part of the SmrteDjango project.
