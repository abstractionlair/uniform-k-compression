"""
Uniform-K Fractal Summarization - Core Framework

This package provides the core algorithm for multi-layer document summarization
using uniform random sampling with LLM compression.
"""

from .document import Document, Tokenizer, create_document_with_tokens
from .config import FrameworkConfig, AnalysisConfig, PreprocessingConfig
from .fractal_summarizer import FractalSummarizer, RunMetadata
from .llm_interface import LLMInterface, APIUsage
from .batch_interface import BatchInterface
from .ollama_interface import OllamaInterface, OllamaUsage, is_ollama_available
from .k_calibrator import find_optimal_K
from .layer_executor import run_layer, LayerStats
from .commentary_manager import CommentaryManager

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
