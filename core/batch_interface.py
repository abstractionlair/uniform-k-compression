#!/usr/bin/env python3
"""
Batch API interface for fractal summarization.

For non-time-critical operations, use the batch API for 50% cost savings.
Batch processing takes hours but costs half as much.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import anthropic

from .llm_interface import APIUsage
from .config import resolve_model_name


@dataclass
class BatchRequest:
    """A single request in a batch."""
    custom_id: str
    prompt: str
    max_tokens: int = 50_000
    context_size: str = "small"  # "small" or "large"


class BatchInterface:
    """
    Interface to Anthropic Batch API.

    Use for operations where latency is not critical (hours OK).
    Provides 50% cost savings vs. real-time API.
    """

    def __init__(
        self,
        model: str = "sonnet",
        large_context_model: str = "sonnet[1m]",
        api_key: Optional[str] = None
    ):
        """
        Initialize batch interface.

        Args:
            model: Model for small contexts (200K Sonnet)
            large_context_model: Model for large contexts (1M Sonnet)
            api_key: API key (uses ANTHROPIC_API_KEY env var if None)
        """
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

        self.model = model
        self.large_context_model = large_context_model
        self.client = anthropic.Anthropic(api_key=api_key)

    def submit_batch(
        self,
        requests: List[BatchRequest]
    ) -> str:
        """
        Submit a batch of requests.

        Args:
            requests: List of BatchRequest objects (can have mixed context sizes)

        Returns:
            Batch ID for tracking
        """
        # Convert to batch API format, selecting model per request
        batch_requests = []
        for req in requests:
            # Select model based on context_size
            model = self.large_context_model if req.context_size == "large" else self.model

            # Resolve simplified names to full API names
            model = resolve_model_name(model)

            batch_requests.append({
                "custom_id": req.custom_id,
                "params": {
                    "model": model,
                    "max_tokens": req.max_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": req.prompt
                        }
                    ]
                }
            })

        # Submit batch
        print(f"Submitting batch: {len(batch_requests)} requests...")
        response = self.client.beta.messages.batches.create(requests=batch_requests)

        batch_id = response.id
        print(f"  ✓ Batch submitted: {batch_id}")
        print(f"    Status: {response.processing_status}")

        return batch_id

    def check_status(self, batch_id: str) -> dict:
        """
        Check batch status.

        Args:
            batch_id: Batch ID from submission

        Returns:
            Status dict with processing_status and request_counts
        """
        batch = self.client.beta.messages.batches.retrieve(batch_id)

        status = {
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

        return status

    def wait_for_completion(
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
        """
        if verbose:
            print(f"Waiting for batch {batch_id}...")

        while True:
            status = self.check_status(batch_id)

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

    def get_results(
        self,
        batch_id: str
    ) -> List[Tuple[str, str, int]]:
        """
        Retrieve results from completed batch.

        Args:
            batch_id: Batch ID

        Returns:
            List of (custom_id, output_text, output_tokens) tuples
        """
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

    def submit_and_wait(
        self,
        requests: List[BatchRequest],
        poll_interval: int = 60
    ) -> List[Tuple[str, str, int]]:
        """
        Submit batch and wait for completion.

        Convenience method that combines submit, wait, and retrieve.

        Args:
            requests: List of BatchRequest objects (can have mixed context sizes)
            poll_interval: Seconds between status checks

        Returns:
            List of (custom_id, output_text, output_tokens) tuples
        """
        batch_id = self.submit_batch(requests)
        success = self.wait_for_completion(batch_id, poll_interval)

        if not success:
            raise RuntimeError(f"Batch {batch_id} failed")

        return self.get_results(batch_id)


if __name__ == "__main__":
    # Test batch API
    try:
        interface = BatchInterface()

        # Create test requests
        requests = [
            BatchRequest(
                custom_id="test_1",
                prompt="What is 2+2? Answer briefly.",
                max_tokens=100
            ),
            BatchRequest(
                custom_id="test_2",
                prompt="What is the capital of France? Answer briefly.",
                max_tokens=100
            )
        ]

        print("Testing batch API...")
        batch_id = interface.submit_batch(requests)

        print(f"\nBatch submitted: {batch_id}")
        print("To check status later:")
        print(f"  python3 -c 'from batch_interface import BatchInterface; bi = BatchInterface(); print(bi.check_status(\"{batch_id}\"))'")

    except ValueError as e:
        print(f"Skipping test (no API key): {e}")
