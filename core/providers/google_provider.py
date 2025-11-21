#!/usr/bin/env python3
"""
Google Gemini provider implementation.

Supports Gemini 3 Pro, 2.5 Pro/Flash with implicit/explicit caching and batch API.
"""

import os
from typing import List, Optional, Tuple

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Google Generative AI SDK not installed. Install with: pip install google-generativeai"
    )

from ..base_provider import BaseProvider, BatchRequest, UsageStats

# Model name resolution
MODEL_MAP = {
    'gemini-3-pro': 'gemini-3-pro',
    'gemini-2.5-pro': 'gemini-2.5-pro',
    'gemini-2.5-flash': 'gemini-2.5-flash',
    'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite',
    'gemini-2-flash': 'gemini-2.0-flash',
}

# Pricing per MTok (as of Nov 2025)
# Source: https://ai.google.dev/gemini-api/docs/pricing
PRICING = {
    'gemini-3-pro-preview': {
        'input': 2.0,  # ≤200k context
        'output': 12.0,
        'cache_read': 0.20,  # 90% discount
    },
    'gemini-2.5-pro': {
        'input': 1.25,  # ≤200k context
        'output': 10.0,
        'cache_read': 0.125,  # 90% discount
    },
    'gemini-2.5-flash': {
        'input': 0.30,
        'output': 2.50,
        'cache_read': 0.03,  # 90% discount
    },
    'gemini-2.5-flash-lite': {
        'input': 0.10,
        'output': 0.40,
        'cache_read': 0.01,  # 90% discount
    },
    'gemini-2.0-flash': {
        'input': 0.10,
        'output': 0.40,
        'cache_read': 0.01,  # Estimated
    },
}


class GoogleProvider(BaseProvider):
    """
    Google Gemini provider.

    Supports:
    - Implicit caching (automatic, 90% discount on 2.5 models, 75% on 2.0)
    - Explicit caching (min 2,048 tokens)
    - Batch API (50% discount, but batch doesn't support explicit caching)
    - Up to 1M token context on 2.5 Pro
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        large_context_model: Optional[str] = "gemini-2.5-pro",
        temperature: float = 1.0,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Google Gemini provider.

        Args:
            model: Model name (default: "gemini-2.5-flash")
            large_context_model: Model for large contexts (default: "gemini-2.5-pro")
            temperature: Sampling temperature
            api_key: API key (uses GOOGLE_API_KEY env var if None)
        """
        self.model = model
        self.large_context_model = large_context_model
        self.temperature = temperature

        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
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
        Call Gemini with prompt.

        Implicit caching is automatic. For explicit caching with repeated
        content, you'd need to use the caching API separately.

        Args:
            prompt: Input prompt
            context_size: "small" (uses model) or "large" (uses large_context_model)
            max_tokens: Maximum output tokens
            timeout: Timeout in seconds

        Returns:
            Tuple of (output_text, output_token_count, usage_stats)
        """
        # Select model based on context size
        model_name = self.large_context_model if context_size == "large" else self.model
        model_name = self._resolve_model(model_name)

        # Create model
        model = genai.GenerativeModel(model_name)

        # Configure generation
        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=self.temperature,
        )

        # Generate
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={'timeout': timeout}
        )

        # Extract output
        output_text = response.text

        # Extract usage
        # Note: Gemini implicit caching discounts are applied automatically
        # but we can't distinguish cached vs non-cached in the response
        usage_obj = response.usage_metadata

        usage = UsageStats(
            input_tokens=usage_obj.prompt_token_count,
            output_tokens=usage_obj.candidates_token_count,
            cache_creation_tokens=0,  # Implicit caching, not exposed
            cache_read_tokens=getattr(usage_obj, 'cached_content_token_count', 0)
        )

        self.total_usage = self.total_usage + usage

        return output_text, usage.output_tokens, usage

    def supports_batch(self) -> bool:
        """Google supports batch API."""
        return True

    def supports_caching(self) -> bool:
        """Google supports implicit and explicit caching."""
        return True

    def submit_batch(self, requests: List[BatchRequest]) -> str:
        """
        Submit batch to Google Gemini Batch API.

        Provides 50% discount. Note: Batch doesn't support explicit caching,
        only implicit caching.

        Args:
            requests: List of BatchRequest objects

        Returns:
            Batch ID for tracking

        Note: This is a placeholder - full implementation would use
        Vertex AI batch prediction API.
        """
        raise NotImplementedError(
            "Google Gemini batch API requires Vertex AI setup. "
            "See: https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/batch-prediction-gemini"
        )

    def check_batch_status(self, batch_id: str) -> dict:
        """Check Google batch status."""
        raise NotImplementedError("Google batch API not fully implemented")

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 60,
        verbose: bool = True
    ) -> bool:
        """Wait for Google batch to complete."""
        raise NotImplementedError("Google batch API not fully implemented")

    def get_batch_results(self, batch_id: str) -> List[Tuple[str, str, int]]:
        """Retrieve results from completed Google batch."""
        raise NotImplementedError("Google batch API not fully implemented")

    def get_total_usage(self) -> UsageStats:
        """Get cumulative usage."""
        return self.total_usage

    def reset_usage(self):
        """Reset usage tracking."""
        self.total_usage = UsageStats()

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "google"

    def calculate_cost(self, usage: UsageStats, model: str) -> float:
        """
        Calculate cost for Google Gemini usage.

        Note: Implicit caching discounts are already applied in billing,
        so we estimate based on cache_read_tokens if available.

        Args:
            usage: Usage statistics
            model: Model name (simplified or full)

        Returns:
            Cost in USD
        """
        model = self._resolve_model(model)

        # Get pricing for this model (default to 2.5 Flash if unknown)
        pricing = PRICING.get(model, PRICING['gemini-2.5-flash'])

        # If we have cache_read_tokens, use discounted rate
        # Otherwise, use regular input rate
        if usage.cache_read_tokens > 0:
            input_cost = ((usage.input_tokens - usage.cache_read_tokens) / 1_000_000) * pricing['input']
            cache_cost = (usage.cache_read_tokens / 1_000_000) * pricing['cache_read']
        else:
            input_cost = (usage.input_tokens / 1_000_000) * pricing['input']
            cache_cost = 0

        output_cost = (usage.output_tokens / 1_000_000) * pricing['output']

        return input_cost + output_cost + cache_cost
