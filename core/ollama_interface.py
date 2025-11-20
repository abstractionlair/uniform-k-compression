#!/usr/bin/env python3
"""
Ollama interface for local LLM testing.

Provides a drop-in replacement for LLMInterface that uses local Ollama models
instead of the Anthropic API. Useful for testing and development without API costs.
"""

import json
import subprocess
from dataclasses import dataclass
from typing import Tuple


@dataclass
class OllamaUsage:
    """Track Ollama usage (similar API to APIUsage from llm_interface, but cost is always $0)."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        """Ollama is free, always returns $0."""
        return 0.0

    def __add__(self, other: 'OllamaUsage') -> 'OllamaUsage':
        """Add two usage objects."""
        return OllamaUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens
        )


class OllamaInterface:
    """
    Interface to local Ollama models for testing.

    Drop-in replacement for LLMInterface that uses local models.
    Matches the same API so it can be swapped in tests or development.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        temperature: float = 1.0,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama interface.

        Args:
            model: Ollama model name (default: qwen2.5:3b, fast and capable)
            temperature: Sampling temperature
            base_url: Ollama API base URL
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.total_usage = OllamaUsage()

        # Check if Ollama is available
        if not self._check_ollama_available():
            raise RuntimeError(
                f"Ollama not available at {base_url}. "
                "Please install and start Ollama: https://ollama.ai"
            )

        # Check if model is pulled
        if not self._check_model_available():
            raise RuntimeError(
                f"Model '{model}' not available. Pull it with: ollama pull {model}"
            )

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            result = subprocess.run(
                ["curl", "-s", f"{self.base_url}/api/tags"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_model_available(self) -> bool:
        """Check if the specified model is available."""
        try:
            result = subprocess.run(
                ["curl", "-s", f"{self.base_url}/api/tags"],
                capture_output=True,
                timeout=2,
                text=True
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                available_models = [m["name"] for m in data.get("models", [])]
                # Check exact match or base model match (e.g., "qwen2.5:3b" or "qwen2.5:latest")
                return any(
                    self.model == m or self.model.split(":")[0] == m.split(":")[0]
                    for m in available_models
                )
        except Exception:
            pass
        return False

    def call(
        self,
        prompt: str,
        context_size: str,
        max_tokens: int = 50_000,
        timeout: float = 600.0
    ) -> Tuple[str, int, OllamaUsage]:
        """
        Call Ollama with prompt.

        Args:
            prompt: Input prompt
            context_size: Ignored for Ollama (kept for API compatibility)
            max_tokens: Maximum output tokens
            timeout: Timeout in seconds

        Returns:
            Tuple of (output_text, output_token_count, usage_stats)
        """
        # Build Ollama API request
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            }
        }

        # Call Ollama API
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-X", "POST",
                    f"{self.base_url}/api/generate",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(request_data)
                ],
                capture_output=True,
                timeout=timeout,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Ollama API call failed: {result.stderr}")

            # Parse response
            response = json.loads(result.stdout)
            output_text = response.get("response", "")

            # Estimate token counts (Ollama doesn't always return them)
            input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            output_tokens = len(output_text.split()) * 1.3  # Rough estimate

            # Track usage
            usage = OllamaUsage(
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens)
            )

            self.total_usage = self.total_usage + usage

            return output_text, int(output_tokens), usage

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Ollama call timed out after {timeout}s")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Ollama response: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")

    def get_total_usage(self) -> OllamaUsage:
        """Get cumulative usage (always $0 cost)."""
        return self.total_usage

    def reset_usage(self):
        """Reset usage tracking."""
        self.total_usage = OllamaUsage()


# Convenience function to check if Ollama is available
def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible."""
    try:
        result = subprocess.run(
            ["curl", "-s", f"{base_url}/api/tags"],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except Exception:
        return False


def list_ollama_models(base_url: str = "http://localhost:11434") -> list:
    """List available Ollama models."""
    try:
        result = subprocess.run(
            ["curl", "-s", f"{base_url}/api/tags"],
            capture_output=True,
            timeout=2,
            text=True
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


if __name__ == "__main__":
    # Test Ollama availability
    print("Checking Ollama availability...")

    if not is_ollama_available():
        print("❌ Ollama not available")
        print("   Install from: https://ollama.ai")
        exit(1)

    print("✅ Ollama is running")

    models = list_ollama_models()
    print(f"\n📦 Available models: {', '.join(models) if models else 'none'}")

    if not models:
        print("\n💡 Pull a model with: ollama pull qwen2.5:3b")
        exit(1)

    # Test a simple call
    print("\n🧪 Testing simple call...")
    try:
        interface = OllamaInterface(model=models[0])
        output, tokens, usage = interface.call(
            "What is 2+2? Answer briefly.",
            context_size="small",
            max_tokens=100
        )

        print("✅ Test successful!")
        print(f"   Output: {output[:100]}")
        print(f"   Tokens: {tokens}")
        print(f"   Cost: ${usage.cost_usd:.2f}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        exit(1)
