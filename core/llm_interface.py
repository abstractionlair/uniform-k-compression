#!/usr/bin/env python3
"""
LLM interface for fractal summarization.

Wrapper for Anthropic API calls with context size management and
token tracking.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import anthropic


# Import after other imports to avoid circular dependency
def resolve_model_name(model: str) -> str:
    """
    Resolve simplified model names - duplicated to avoid circular import.

    Extended context is accessed via beta flag, not a different model name.
    """
    model_map = {
        'sonnet': 'claude-sonnet-4-5-20250929',
        'sonnet[1m]': 'claude-sonnet-4-5-20250929',  # Same model, use beta flag for 1M
        'opus': 'claude-opus-4-20250514',
        'haiku': 'claude-3-5-haiku-20241022'
    }
    return model_map.get(model, model)


@dataclass
class APIUsage:
    """Track API usage for cost estimation."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        """
        Estimate cost in USD for Sonnet 4.5.

        Pricing (as of Oct 2025):
        - Input: $3 / MTok
        - Output: $15 / MTok
        - Cache write: $3.75 / MTok
        - Cache read: $0.30 / MTok
        """
        input_cost = (self.input_tokens / 1_000_000) * 3.0
        output_cost = (self.output_tokens / 1_000_000) * 15.0
        cache_write = (self.cache_creation_tokens / 1_000_000) * 3.75
        cache_read = (self.cache_read_tokens / 1_000_000) * 0.30

        return input_cost + output_cost + cache_write + cache_read

    def __add__(self, other: 'APIUsage') -> 'APIUsage':
        """Add two usage objects."""
        return APIUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens
        )


class LLMInterface:
    """
    Interface to Anthropic API for fractal summarization.

    Handles context size selection and token usage tracking.
    """

    def __init__(
        self,
        model: str = "sonnet",
        large_context_model: str = "sonnet[1m]",
        temperature: float = 1.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM interface.

        Args:
            model: Anthropic model name for small contexts (200K Sonnet)
            large_context_model: Model for large contexts (1M Sonnet)
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
        self.total_usage = APIUsage()

    def call(
        self,
        prompt: str,
        context_size: str,
        max_tokens: int = 50_000,
        timeout: float = 600.0
    ) -> Tuple[str, int, APIUsage]:
        """
        Call LLM with prompt.

        Args:
            prompt: Input prompt
            context_size: "small" (uses 200K model) or "large" (uses 1M model)
            max_tokens: Maximum output tokens
            timeout: Timeout in seconds

        Returns:
            Tuple of (output_text, output_token_count, usage_stats)
        """
        # Select model - use base model for both, but add beta flag for large context
        model = resolve_model_name(self.model)

        # For large context, use beta.messages.create with context-1m beta flag
        if context_size == "large":
            response = self.client.beta.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                timeout=timeout,
                betas=["context-1m-2025-08-07"],
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        else:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                timeout=timeout,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        # Debug: Print actual model used
        print(f"  API returned model: {response.model}")

        # Extract output text
        output_text = ""
        for block in response.content:
            if block.type == "text":
                output_text += block.text

        # Extract usage
        usage_obj = response.usage
        usage = APIUsage(
            input_tokens=usage_obj.input_tokens,
            output_tokens=usage_obj.output_tokens,
            cache_creation_tokens=getattr(usage_obj, 'cache_creation_input_tokens', 0),
            cache_read_tokens=getattr(usage_obj, 'cache_read_input_tokens', 0)
        )

        # Update total
        self.total_usage = self.total_usage + usage

        return output_text, usage.output_tokens, usage

    def get_total_usage(self) -> APIUsage:
        """Get cumulative API usage."""
        return self.total_usage

    def reset_usage(self):
        """Reset usage tracking."""
        self.total_usage = APIUsage()


if __name__ == "__main__":
    # Test with a simple call (requires API key)
    try:
        interface = LLMInterface()

        prompt = "What is 2+2? Answer briefly."
        output, tokens, usage = interface.call(prompt, "small", max_tokens=100)

        print("Test call:")
        print(f"  Output: {output}")
        print(f"  Tokens: {tokens}")
        print(f"  Cost: ${usage.cost_usd:.4f}")

    except ValueError as e:
        print(f"Skipping test (no API key): {e}")
