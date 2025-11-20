"""
Uniform-K Fractal Summarization - Core Framework

This package provides the core algorithm for multi-layer document summarization
using uniform random sampling with LLM compression.
"""

from .batch_interface import BatchInterface
from .commentary_manager import CommentaryManager
from .config import AnalysisConfig, FrameworkConfig, PreprocessingConfig
from .document import Document, Tokenizer, create_document_with_tokens
from .fractal_summarizer import FractalSummarizer, RunMetadata
from .k_calibrator import find_optimal_K
from .layer_executor import LayerStats, run_layer
from .llm_interface import APIUsage, LLMInterface
from .ollama_interface import OllamaInterface, OllamaUsage, is_ollama_available

__all__ = [
    # Document handling
    "Document",
    "Tokenizer",
    "create_document_with_tokens",

    # Configuration
    "FrameworkConfig",
    "AnalysisConfig",
    "PreprocessingConfig",

    # Main interface
    "FractalSummarizer",
    "RunMetadata",

    # LLM interfaces
    "LLMInterface",
    "OllamaInterface",
    "BatchInterface",
    "APIUsage",
    "OllamaUsage",
    "is_ollama_available",

    # Layer execution
    "run_layer",
    "LayerStats",
    "find_optimal_K",

    # Iterative refinement
    "CommentaryManager",
]
