#!/usr/bin/env python3
"""
OpenAI provider implementation.

Supports GPT-5.1, GPT-4o, o1 models with automatic prompt caching and batch API.
"""

import os
import time
from typing import List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI SDK not installed. Install with: pip install openai"
    )

from ..base_provider import BaseProvider, BatchRequest, UsageStats

# Model name resolution
MODEL_MAP = {
    # GPT-5 family (Aug 2025)
    'gpt-5': 'gpt-5',
    'gpt-5-mini': 'gpt-5-mini',
    'gpt-5-nano': 'gpt-5-nano',
    # Legacy names (backwards compat)
    'gpt-5.1': 'gpt-5',
    'gpt-5.1-instant': 'gpt-5-mini',
    'gpt-5.1-thinking': 'gpt-5',
    # GPT-4o
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    # o1 reasoning models
    'o1': 'o1',
    'o1-mini': 'o1-mini',
    'o1-preview': 'o1-preview',
}

# Pricing per MTok (as of Nov 2025)
# Source: https://openai.com/api/pricing/ (Aug 2025 GPT-5 release)
PRICING = {
    # GPT-5 family
    'gpt-5': {
        'input': 1.25,
        'output': 10.0,
        'cache_read': 0.125,  # 90% discount
    },
    'gpt-5-mini': {
        'input': 0.25,
        'output': 2.0,
        'cache_read': 0.025,  # 90% discount
    },
    'gpt-5-nano': {
        'input': 0.05,
        'output': 0.40,
        'cache_read': 0.005,  # 90% discount
    },
    # GPT-4o
    'gpt-4o': {
        'input': 2.5,
        'output': 10.0,
        'cache_read': 1.25,
    },
    'gpt-4o-mini': {
        'input': 0.15,
        'output': 0.60,
        'cache_read': 0.075,
    },
    # o1 reasoning models
    'o1': {
        'input': 15.0,
        'output': 60.0,
        'cache_read': 7.5,
    },
    'o1-mini': {
        'input': 3.0,
        'output': 12.0,
        'cache_read': 1.5,
    },
}


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider.

    Supports:
    - Prompt caching (automatic on prompts >1,024 tokens, 50% discount)
    - Batch API (50% discount on both input and output)
    - GPT-5.1, GPT-4o, o1 models
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        large_context_model: Optional[str] = None,
        temperature: float = 1.0,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (default: "gpt-5.1")
            large_context_model: Not used (OpenAI models have large context by default)
            temperature: Sampling temperature
            api_key: API key (uses OPENAI_API_KEY env var if None)
        """
        self.model = model
        self.temperature = temperature

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=api_key)
        self.total_usage = UsageStats()

    def _resolve_model(self, model: str) -> str:
        """Resolve simplified model name to full API name."""
        return MODEL_MAP.get(model, model)

    def call(
        self,
        prompt: str,
        context_size: str,
        max_tokens: int = 50_000,
        timeout: float = 600.0
    ) -> Tuple[str, int, UsageStats]:
        """
        Call OpenAI with prompt.

        Automatic prompt caching on prompts >1,024 tokens.

        Args:
            prompt: Input prompt
            context_size: Ignored (OpenAI models have large context)
            max_tokens: Maximum output tokens
            timeout: Timeout in seconds

        Returns:
            Tuple of (output_text, output_token_count, usage_stats)
        """
        model = self._resolve_model(self.model)

        # GPT-5 models use max_completion_tokens instead of max_tokens
        if model.startswith('gpt-5') or model.startswith('o1'):
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=self.temperature,
                timeout=timeout,
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature,
                timeout=timeout,
            )

        # Extract output
        output_text = response.choices[0].message.content or ""

        # Extract usage with caching info
        usage_obj = response.usage

        # Check for cached tokens (only available on some models)
        cached_tokens = 0
        if hasattr(usage_obj, 'prompt_tokens_details'):
            details = usage_obj.prompt_tokens_details
            if hasattr(details, 'cached_tokens'):
                cached_tokens = details.cached_tokens or 0

        usage = UsageStats(
            input_tokens=usage_obj.prompt_tokens,
            output_tokens=usage_obj.completion_tokens,
            cache_creation_tokens=0,  # OpenAI doesn't expose cache writes
            cache_read_tokens=cached_tokens
        )

        self.total_usage = self.total_usage + usage

        return output_text, usage.output_tokens, usage

    def supports_batch(self) -> bool:
        """OpenAI supports batch API."""
        return True

    def supports_caching(self) -> bool:
        """OpenAI supports automatic prompt caching."""
        return True

    def submit_batch(self, requests: List[BatchRequest]) -> str:
        """
        Submit batch to OpenAI Batch API.

        Provides 50% discount on both input and output tokens.

        Args:
            requests: List of BatchRequest objects

        Returns:
            Batch ID for tracking

        Note: OpenAI batch API requires uploading a JSONL file.
        This is a simplified implementation - production use may need
        file-based batching.
        """
        # For OpenAI, we'd need to create a JSONL file and upload it
        # This is a placeholder - full implementation would use files
        raise NotImplementedError(
            "OpenAI batch API requires file-based implementation. "
            "See: https://platform.openai.com/docs/guides/batch"
        )

    def check_batch_status(self, batch_id: str) -> dict:
        """Check OpenAI batch status."""
        batch = self.client.batches.retrieve(batch_id)

        return {
            "batch_id": batch_id,
            "processing_status": batch.status,
            "request_counts": {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed,
            }
        }

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 60,
        verbose: bool = True
    ) -> bool:
        """Wait for OpenAI batch to complete."""
        if verbose:
            print(f"Waiting for batch {batch_id}...")

        while True:
            status = self.check_batch_status(batch_id)

            if verbose:
                counts = status['request_counts']
                print(f"  Status: {status['processing_status']}")
                print(f"    Completed: {counts['completed']}, Failed: {counts['failed']}")

            if status['processing_status'] in ['completed', 'failed', 'cancelled']:
                if verbose:
                    print("  ✓ Batch complete!")
                return status['processing_status'] == 'completed'

            time.sleep(poll_interval)

    def get_batch_results(self, batch_id: str) -> List[Tuple[str, str, int]]:
        """
        Retrieve results from completed OpenAI batch.

        Note: This is a placeholder - full implementation would
        download and parse the results file.
        """
        raise NotImplementedError(
            "OpenAI batch results require file download. "
            "See: https://platform.openai.com/docs/guides/batch"
        )

    def get_total_usage(self) -> UsageStats:
        """Get cumulative usage."""
        return self.total_usage

    def reset_usage(self):
        """Reset usage tracking."""
        self.total_usage = UsageStats()

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "openai"

    def calculate_cost(self, usage: UsageStats, model: str) -> float:
        """
        Calculate cost for OpenAI usage.

        Args:
            usage: Usage statistics
            model: Model name (simplified or full)

        Returns:
            Cost in USD
        """
        model = self._resolve_model(model)

        # Get pricing for this model (default to GPT-5 if unknown)
        pricing = PRICING.get(model, PRICING['gpt-5'])

        input_cost = (usage.input_tokens / 1_000_000) * pricing['input']
        output_cost = (usage.output_tokens / 1_000_000) * pricing['output']
        cache_read_cost = (usage.cache_read_tokens / 1_000_000) * pricing['cache_read']

        return input_cost + output_cost + cache_read_cost
