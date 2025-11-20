#!/usr/bin/env python3
"""
Commentary manager for iterative fractal analysis refinement.

Allows incorporating feedback from previous runs to improve subsequent analyses.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class CommentaryManager:
    """
    Manages user feedback on previous analysis runs for iterative refinement.

    Enables an iterative workflow:
    1. Run initial analysis
    2. User reviews output and provides commentary (what was accurate, missed, or misunderstood)
    3. Re-run analysis with commentary incorporated into prompts
    4. Results improve with each iteration

    Commentary is incorporated into prompts at all layers to guide the LLM's analysis.
    """

    def __init__(self, commentary_file: Optional[str] = None):
        """
        Initialize commentary manager.

        Args:
            commentary_file: Path to markdown file with commentary
        """
        self.commentary_file = commentary_file
        self.commentary_text = None

        if commentary_file:
            self.load_commentary(commentary_file)

    def load_commentary(self, commentary_file: str) -> None:
        """
        Load commentary from markdown file.

        Args:
            commentary_file: Path to commentary file
        """
        path = Path(commentary_file)
        if not path.exists():
            raise FileNotFoundError(f"Commentary file not found: {commentary_file}")

        with open(path) as f:
            self.commentary_text = f.read().strip()

    def format_for_prompt(self, layer: str) -> str:
        """
        Format commentary for inclusion in prompts.

        Args:
            layer: Which layer this is being used for ("layer1", "layer2+", "final")

        Returns:
            Formatted commentary section for prompt
        """
        if not self.commentary_text:
            return ""

        # Different framing depending on layer
        if layer == "layer1":
            header = """

---

**USER COMMENTARY (Feedback from Previous Run):**

The user has reviewed a previous analysis run and provided the following commentary. This represents their feedback on what was accurate, what was missing, what was misunderstood, and what additional context matters. Use this to calibrate your analysis, but remember:
- User feedback adds important context but doesn't override evidence in the documents
- Where commentary and evidence conflict, note the discrepancy explicitly
- Commentary helps you understand what the user is looking for and how to frame insights

**User's Feedback:**

"""
        elif layer == "layer2+":
            header = """

---

**USER COMMENTARY (Feedback from Previous Run):**

The user reviewed a previous synthesis and provided feedback on accuracy, gaps, and misunderstandings. The Layer 1 summaries you're reading were created with this commentary in context. Your task is to synthesize their findings, which already incorporate the user's feedback.

**User's Feedback (for context):**

"""
        else:  # final synthesis
            header = """

---

**USER COMMENTARY (Feedback from Previous Run):**

The user provided detailed feedback on a previous final synthesis. All the Layer summaries you're reading were generated with this commentary in context, so their analyses should already reflect the user's corrections and clarifications.

Your synthesis should:
- Integrate the commentary-informed insights from all layers
- Address any gaps or misunderstandings noted in the feedback
- Be explicit about any remaining ambiguities or areas needing clarification

**User's Feedback (for your context):**

"""

        return header + self.commentary_text + "\n\n---\n"

    def augment_prompt(self, base_prompt: str, layer: str) -> str:
        """
        Add commentary section to a prompt.

        Args:
            base_prompt: The original prompt template
            layer: Which layer type ("layer1", "layer2+", "final")

        Returns:
            Augmented prompt with commentary
        """
        if not self.commentary_text:
            return base_prompt

        # Insert commentary before the documents section
        # Look for common markers of where documents begin
        markers = [
            "\n**Conversations:**\n",
            "\n**Summaries from Layer",
            "\n{documents}",
        ]

        commentary_section = self.format_for_prompt(layer)

        # Try to insert before documents section
        for marker in markers:
            if marker in base_prompt:
                return base_prompt.replace(marker, commentary_section + marker)

        # Fallback: append before end
        # Look for the final "Your analysis:" or similar
        final_markers = [
            "\n**Your analysis:**",
            "\n**Your comprehensive analysis:**",
            "\n**Your synthesis:**",
        ]

        for marker in final_markers:
            if marker in base_prompt:
                return base_prompt.replace(marker, commentary_section + marker)

        # Last resort: just append
        return base_prompt + "\n\n" + commentary_section


def load_commentary_config(config_file: str) -> Dict[str, Any]:
    """
    Load analysis config and check for commentary file reference.

    Args:
        config_file: Path to analysis config JSON

    Returns:
        Config dict with commentary_file key if present
    """
    with open(config_file) as f:
        config = json.load(f)

    # Check for commentary_file in config
    if "commentary_file" in config:
        return config

    return config


def create_commentary_template(output_path: str) -> None:
    """
    Create a template commentary file for user feedback.

    Args:
        output_path: Where to write the template
    """
    template = """# Commentary on Analysis Run

**Run being commented on:** [e.g., my_analysis_run1, 2025-11-19]

## What Was Accurate

[Note patterns, insights, or characterizations that resonated as true]

## What Was Missing

[Important aspects of your thinking, capabilities, or context that weren't captured]

## What Was Misunderstood

[Patterns that were identified but misinterpreted, or claims that don't match reality]

## Important Context

[Background information that would help calibrate interpretation]

## Specific Corrections

[Point-by-point corrections to specific claims or characterizations]

## Questions or Uncertainties

[Things you're not sure about in your own patterns - areas where external validation would help]

---

## Instructions for Effective Commentary

**Be specific:** Instead of "the analysis got my work patterns wrong," say "the analysis suggested I struggle with execution, but in my professional roles I consistently shipped major projects on deadline."

**Provide evidence:** When correcting misunderstandings, cite concrete examples.

**Note uncertainty:** If you're unsure whether a pattern is accurate, say so - that's valuable information.

**Separate perception from reality:** You can note "this feels wrong to me" even when you don't have contradicting evidence.

**Context matters:** Explain circumstances that might affect interpretation (e.g., "the conversations sampled were during a particularly stressful period").
"""

    with open(output_path, 'w') as f:
        f.write(template)

    print(f"Created commentary template: {output_path}")
    print("Fill this in with your feedback on the analysis.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "template":
        # Create template
        output = sys.argv[2] if len(sys.argv) > 2 else "commentary_template.md"
        create_commentary_template(output)
    else:
        # Test loading and formatting
        if len(sys.argv) < 2:
            print("Usage:")
            print("  python commentary_manager.py template [output.md]  # Create template")
            print("  python commentary_manager.py <commentary.md>       # Test formatting")
            sys.exit(1)

        cm = CommentaryManager(sys.argv[1])

        print("=== Layer 1 Format ===")
        print(cm.format_for_prompt("layer1")[:500])
        print("\n=== Layer 2+ Format ===")
        print(cm.format_for_prompt("layer2+")[:500])
        print("\n=== Final Format ===")
        print(cm.format_for_prompt("final")[:500])
