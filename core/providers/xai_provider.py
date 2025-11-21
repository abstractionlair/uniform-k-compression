#!/usr/bin/env python3
"""
xAI Grok provider implementation.

Supports Grok 4, Grok 4.1 Fast with automatic caching (>90% hit rates).
OpenAI SDK compatible.
"""

import os
from typing import List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI SDK not installed (needed for xAI). Install with: pip install openai"
    )

from ..base_provider import BaseProvider, UsageStats, BatchRequest


# Model name resolution
MODEL_MAP = {
    'grok-4': 'grok-4',
    'grok-4.1-fast': 'grok-4-1-fast-reasoning',
    'grok-4.1-fast-reasoning': 'grok-4-1-fast-reasoning',
    'grok-4.1-fast-non-reasoning': 'grok-4-1-fast-non-reasoning',
    'grok-3': 'grok-3',
    'grok-3-mini': 'grok-3-mini',
    'grok-code-fast': 'grok-code-fast-1',
}

# Pricing per MTok (as of Nov 2025)
# Source: https://x.ai/news/grok-4-1
PRICING = {
    'grok-4': {
        'input': 0.20,  # Same as 4.1 Fast
        'output': 0.50,
        'cache_read': 0.05,
    },
    'grok-4-1-fast-reasoning': {
        'input': 0.20,
        'output': 0.50,
        'cache_read': 0.05,
    },
    'grok-4-1-fast-non-reasoning': {
        'input': 0.20,
        'output': 0.50,
        'cache_read': 0.05,
    },
    'grok-3': {
        'input': 3.0,
        'output': 15.0,
        'cache_read': 0.30,  # Estimated at 10%
    },
    'grok-3-mini': {
        'input': 0.30,
        'output': 0.50,
        'cache_read': 0.03,  # Estimated at 10%
    },
    'grok-code-fast-1': {
        'input': 0.20,
        'output': 0.50,
        'cache_read': 0.05,
    },
}


class XAIProvider(BaseProvider):
    """
    xAI Grok provider.

    Supports:
    - Automatic prompt caching (>90% hit rates on code model)
    - Very cheap cached tokens ($0.02/MTok on code model)
    - OpenAI SDK compatible
    - Grok 4 (plain) for deep reasoning
    - Grok 4.1 Fast for production speed

    Note: Batch API not documented, so not implemented.
    """

    def __init__(
        self,
        model: str = "grok-4.1-fast",
        large_context_model: Optional[str] = None,
        temperature: float = 1.0,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize xAI provider.

        Args:
            model: Model name (default: "grok-4.1-fast")
            large_context_model: Not used (Grok models have large context)
            temperature: Sampling temperature
            api_key: API key (uses XAI_API_KEY env var if None)
        """
        self.model = model
        self.temperature = temperature

        if api_key is None:
            api_key = os.environ.get("XAI_API_KEY")
            if not api_key:
                raise ValueError("XAI_API_KEY not set")

        # xAI uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
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
        Call Grok with prompt.

        Automatic caching enabled on all requests.

        Args:
            prompt: Input prompt
            context_size: Ignored (Grok models have large context)
            max_tokens: Maximum output tokens (important for plain Grok 4)
            timeout: Timeout in seconds

        Returns:
            Tuple of (output_text, output_token_count, usage_stats)
        """
        model = self._resolve_model(self.model)

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

        # xAI exposes cached tokens in usage object
        cached_tokens = 0
        if hasattr(usage_obj, 'prompt_tokens_details'):
            details = usage_obj.prompt_tokens_details
            if hasattr(details, 'cached_tokens'):
                cached_tokens = details.cached_tokens or 0

        usage = UsageStats(
            input_tokens=usage_obj.prompt_tokens,
            output_tokens=usage_obj.completion_tokens,
            cache_creation_tokens=0,  # xAI doesn't expose cache writes
            cache_read_tokens=cached_tokens
        )

        self.total_usage = self.total_usage + usage

        return output_text, usage.output_tokens, usage

    def supports_batch(self) -> bool:
        """xAI batch API not documented."""
        return False

    def supports_caching(self) -> bool:
        """xAI supports automatic caching."""
        return True

    def get_total_usage(self) -> UsageStats:
        """Get cumulative usage."""
        return self.total_usage

    def reset_usage(self):
        """Reset usage tracking."""
        self.total_usage = UsageStats()

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "xai"

    def calculate_cost(self, usage: UsageStats, model: str) -> float:
        """
        Calculate cost for xAI usage.

        Args:
            usage: Usage statistics
            model: Model name (simplified or full)

        Returns:
            Cost in USD
        """
        model = self._resolve_model(model)

        # Get pricing for this model (default to Grok 4.1 Fast if unknown)
        pricing = PRICING.get(model, PRICING['grok-4-1-fast-reasoning'])

        input_cost = (usage.input_tokens / 1_000_000) * pricing['input']
        output_cost = (usage.output_tokens / 1_000_000) * pricing['output']
        cache_read_cost = (usage.cache_read_tokens / 1_000_000) * pricing['cache_read']

        return input_cost + output_cost + cache_read_cost
