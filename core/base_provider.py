#!/usr/bin/env python3
"""
Base provider interface for LLM inference.

Defines the common contract that all providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class UsageStats:
    """
    Unified usage tracking across providers.

    All providers normalize their usage to this format.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0  # Tokens written to cache
    cache_read_tokens: int = 0      # Tokens read from cache

    def __add__(self, other: 'UsageStats') -> 'UsageStats':
        """Add two usage objects."""
        return UsageStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens
        )


@dataclass
class BatchRequest:
    """A single request in a batch."""
    custom_id: str
    prompt: str
    max_tokens: int = 50_000
    context_size: str = "small"  # "small" or "large"


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement this interface to be compatible
    with the fractal summarization framework.
    """

    @abstractmethod
    def __init__(
        self,
        model: str,
        large_context_model: Optional[str] = None,
        temperature: float = 1.0,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize provider.

        Args:
            model: Model name for small contexts
            large_context_model: Model for large contexts (if provider supports)
            temperature: Sampling temperature
            api_key: API key (provider-specific env var if None)
            **kwargs: Provider-specific parameters
        """
        pass

    @abstractmethod
    def call(
        self,
        prompt: str,
        context_size: str,
        max_tokens: int = 50_000,
        timeout: float = 600.0
    ) -> Tuple[str, int, UsageStats]:
        """
        Call LLM with prompt.

        Args:
            prompt: Input prompt
            context_size: "small" or "large" (provider interprets as needed)
            max_tokens: Maximum output tokens
            timeout: Timeout in seconds

        Returns:
            Tuple of (output_text, output_token_count, usage_stats)
        """
        pass

    @abstractmethod
    def supports_batch(self) -> bool:
        """Return True if provider supports batch API."""
        pass

    @abstractmethod
    def supports_caching(self) -> bool:
        """Return True if provider supports prompt caching."""
        pass

    def submit_batch(
        self,
        requests: List[BatchRequest]
    ) -> str:
        """
        Submit a batch of requests.

        Args:
            requests: List of BatchRequest objects

        Returns:
            Batch ID for tracking

        Raises:
            NotImplementedError: If provider doesn't support batch API
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batch API"
        )

    def check_batch_status(self, batch_id: str) -> dict:
        """
        Check batch status.

        Args:
            batch_id: Batch ID from submission

        Returns:
            Status dict with processing_status and request_counts

        Raises:
            NotImplementedError: If provider doesn't support batch API
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batch API"
        )

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 60,
        verbose: bool = True
    ) -> bool:
        """
        Wait for batch to complete.

        Args:
            batch_id: Batch ID
            poll_interval: Seconds between status checks
            verbose: Print progress

        Returns:
            True if completed successfully, False otherwise

        Raises:
            NotImplementedError: If provider doesn't support batch API
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batch API"
        )

    def get_batch_results(
        self,
        batch_id: str
    ) -> List[Tuple[str, str, int]]:
        """
        Retrieve results from completed batch.

        Args:
            batch_id: Batch ID

        Returns:
            List of (custom_id, output_text, output_tokens) tuples

        Raises:
            NotImplementedError: If provider doesn't support batch API
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batch API"
        )

    def submit_and_wait(
        self,
        requests: List[BatchRequest],
        poll_interval: int = 60
    ) -> List[Tuple[str, str, int]]:
        """
        Submit batch and wait for completion (convenience method).

        Args:
            requests: List of BatchRequest objects
            poll_interval: Seconds between status checks

        Returns:
            List of (custom_id, output_text, output_tokens) tuples

        Raises:
            NotImplementedError: If provider doesn't support batch API
            RuntimeError: If batch fails
        """
        batch_id = self.submit_batch(requests)
        success = self.wait_for_batch(batch_id, poll_interval)

        if not success:
            raise RuntimeError(f"Batch {batch_id} failed")

        return self.get_batch_results(batch_id)

    @abstractmethod
    def get_total_usage(self) -> UsageStats:
        """Get cumulative usage stats."""
        pass

    @abstractmethod
    def reset_usage(self):
        """Reset usage tracking."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name (e.g., 'anthropic', 'openai')."""
        pass

    @abstractmethod
    def calculate_cost(self, usage: UsageStats, model: str) -> float:
        """
        Calculate cost in USD for given usage and model.

        Args:
            usage: Usage statistics
            model: Model name

        Returns:
            Cost in USD
        """
        pass
