#!/usr/bin/env python3
"""
Anthropic Claude provider implementation.

Supports prompt caching and batch API with 50% cost savings.
"""

import os
import time
from typing import List, Optional, Tuple

import anthropic

from ..base_provider import BaseProvider, UsageStats, BatchRequest


# Model name resolution
MODEL_MAP = {
    'sonnet': 'claude-sonnet-4-5-20250929',
    'sonnet[1m]': 'claude-sonnet-4-5-20250929',
    'opus': 'claude-opus-4-20250514',
    'opus-4.1': 'claude-opus-4-20250514',
    'haiku': 'claude-3-5-haiku-20241022',
    'haiku-4.5': 'claude-3-5-haiku-20241022',
    'sonnet-3.7': 'claude-3-7-sonnet-20250219',
}

# Pricing per MTok (as of Nov 2025)
# Source: https://platform.claude.com/docs/en/about-claude/pricing
PRICING = {
    'claude-opus-4-20250514': {
        'input': 15.0,
        'output': 75.0,
        'cache_write': 18.75,
        'cache_read': 1.50,
    },
    'claude-sonnet-4-5-20250929': {
        'input': 3.0,
        'output': 15.0,
        'cache_write': 3.75,
        'cache_read': 0.30,
    },
    'claude-3-5-haiku-20241022': {  # Haiku 3.5
        'input': 0.80,
        'output': 4.0,
        'cache_write': 1.0,
        'cache_read': 0.08,
    },
    'claude-3-7-sonnet-20250219': {
        'input': 3.0,
        'output': 15.0,
        'cache_write': 3.75,
        'cache_read': 0.30,
    },
}


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude provider.

    Supports:
    - Prompt caching (automatic, up to 90% cost reduction)
    - Batch API (50% discount, can stack with caching)
    - Extended context (1M tokens via beta flag)
    """

    def __init__(
        self,
        model: str = "sonnet",
        large_context_model: str = "sonnet[1m]",
        temperature: float = 1.0,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Model for small contexts (default: "sonnet")
            large_context_model: Model for large contexts (default: "sonnet[1m]")
            temperature: Sampling temperature
            api_key: API key (uses ANTHROPIC_API_KEY env var if None)
        """
        self.model = model
        self.large_context_model = large_context_model
        self.temperature = temperature

        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=api_key)
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
        Call Claude with prompt.

        Automatically uses prompt caching where beneficial.

        Args:
            prompt: Input prompt
            context_size: "small" (200K) or "large" (1M with beta flag)
            max_tokens: Maximum output tokens
            timeout: Timeout in seconds

        Returns:
            Tuple of (output_text, output_token_count, usage_stats)
        """
        model = self._resolve_model(self.model)

        # Haiku has a lower max_tokens limit (8192)
        if 'haiku' in model.lower() and max_tokens > 8192:
            max_tokens = 8192

        # For large context, use beta flag
        if context_size == "large":
            response = self.client.beta.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                timeout=timeout,
                betas=["context-1m-2025-08-07"],
                messages=[{"role": "user", "content": prompt}]
            )
        else:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                timeout=timeout,
                messages=[{"role": "user", "content": prompt}]
            )

        # Extract output text
        output_text = ""
        for block in response.content:
            if block.type == "text":
                output_text += block.text

        # Extract usage with caching info
        usage_obj = response.usage
        usage = UsageStats(
            input_tokens=usage_obj.input_tokens,
            output_tokens=usage_obj.output_tokens,
            cache_creation_tokens=getattr(usage_obj, 'cache_creation_input_tokens', 0),
            cache_read_tokens=getattr(usage_obj, 'cache_read_input_tokens', 0)
        )

        self.total_usage = self.total_usage + usage

        return output_text, usage.output_tokens, usage

    def supports_batch(self) -> bool:
        """Anthropic supports batch API."""
        return True

    def supports_caching(self) -> bool:
        """Anthropic supports prompt caching."""
        return True

    def submit_batch(self, requests: List[BatchRequest]) -> str:
        """
        Submit batch to Anthropic Message Batches API.

        Provides 50% cost savings (can stack with caching).

        Args:
            requests: List of BatchRequest objects

        Returns:
            Batch ID for tracking
        """
        batch_requests = []
        for req in requests:
            # Select model based on context size
            model = self.large_context_model if req.context_size == "large" else self.model
            model = self._resolve_model(model)

            batch_requests.append({
                "custom_id": req.custom_id,
                "params": {
                    "model": model,
                    "max_tokens": req.max_tokens,
                    "messages": [{"role": "user", "content": req.prompt}]
                }
            })

        print(f"Submitting batch: {len(batch_requests)} requests...")
        response = self.client.beta.messages.batches.create(requests=batch_requests)

        print(f"  ✓ Batch submitted: {response.id}")
        print(f"    Status: {response.processing_status}")

        return response.id

    def check_batch_status(self, batch_id: str) -> dict:
        """Check Anthropic batch status."""
        batch = self.client.beta.messages.batches.retrieve(batch_id)

        return {
            "batch_id": batch_id,
            "processing_status": batch.processing_status,
            "request_counts": {
                "processing": batch.request_counts.processing,
                "succeeded": batch.request_counts.succeeded,
                "errored": batch.request_counts.errored,
                "canceled": batch.request_counts.canceled,
                "expired": batch.request_counts.expired
            }
        }

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 60,
        verbose: bool = True
    ) -> bool:
        """Wait for Anthropic batch to complete."""
        if verbose:
            print(f"Waiting for batch {batch_id}...")

        while True:
            status = self.check_batch_status(batch_id)

            if verbose:
                counts = status['request_counts']
                print(f"  Status: {status['processing_status']}")
                print(f"    Succeeded: {counts['succeeded']}, Processing: {counts['processing']}, "
                      f"Errored: {counts['errored']}")

            if status['processing_status'] == 'ended':
                if verbose:
                    print("  ✓ Batch complete!")
                return counts['succeeded'] > 0

            time.sleep(poll_interval)

    def get_batch_results(self, batch_id: str) -> List[Tuple[str, str, int]]:
        """Retrieve results from completed Anthropic batch."""
        results_iter = self.client.beta.messages.batches.results(batch_id)

        outputs = []
        for result in results_iter:
            if result.result.type == 'succeeded':
                custom_id = result.custom_id
                message = result.result.message

                # Extract text
                output_text = ""
                for content in message.content:
                    if content.type == 'text':
                        output_text += content.text

                output_tokens = message.usage.output_tokens
                outputs.append((custom_id, output_text, output_tokens))

            elif result.result.type == 'errored':
                error = result.result.error
                print(f"  ⚠️  Error in {result.custom_id}: {error.type}")

        return outputs

    def get_total_usage(self) -> UsageStats:
        """Get cumulative usage."""
        return self.total_usage

    def reset_usage(self):
        """Reset usage tracking."""
        self.total_usage = UsageStats()

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "anthropic"

    def calculate_cost(self, usage: UsageStats, model: str) -> float:
        """
        Calculate cost for Anthropic usage.

        Args:
            usage: Usage statistics
            model: Model name (simplified or full)

        Returns:
            Cost in USD
        """
        model = self._resolve_model(model)

        # Get pricing for this model (default to Sonnet if unknown)
        pricing = PRICING.get(model, PRICING['claude-sonnet-4-5-20250929'])

        input_cost = (usage.input_tokens / 1_000_000) * pricing['input']
        output_cost = (usage.output_tokens / 1_000_000) * pricing['output']
        cache_write_cost = (usage.cache_creation_tokens / 1_000_000) * pricing['cache_write']
        cache_read_cost = (usage.cache_read_tokens / 1_000_000) * pricing['cache_read']

        return input_cost + output_cost + cache_write_cost + cache_read_cost
