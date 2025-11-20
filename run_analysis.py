#!/usr/bin/env python3
"""
Run fractal summarization on a directory of text files.

This script serves as the main entry point for running the analysis.
"""

import argparse
import sys
from pathlib import Path

from core import AnalysisConfig, FractalSummarizer, FrameworkConfig
from utilities import load_documents


def main():
    parser = argparse.ArgumentParser(description="Run Fractal Summarization on a text corpus.")
    
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Input directory containing text files"
    )
    parser.add_argument(
        "--analysis-config", "-a", 
        required=True, 
        help="Path to analysis configuration JSON (prompts, etc.)"
    )
    parser.add_argument(
        "--framework-config", "-c", 
        default="configs/framework_default.json", 
        help="Path to framework configuration JSON (algorithm params)"
    )
    parser.add_argument(
        "--output", "-o", 
        default="output", 
        help="Output directory (overrides config if specified)"
    )
    parser.add_argument(
        "--pattern", "-p", 
        default="*.txt", 
        help="File glob pattern to match (default: *.txt)"
    )
    parser.add_argument(
        "--limit", "-l", 
        type=int, 
        help="Limit number of documents to load (for testing)"
    )

    args = parser.parse_args()

    # Validate paths
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    framework_config_path = Path(args.framework_config)
    if not framework_config_path.exists():
        print(f"Error: Framework config '{framework_config_path}' does not exist.")
        sys.exit(1)

    analysis_config_path = Path(args.analysis_config)
    if not analysis_config_path.exists():
        print(f"Error: Analysis config '{analysis_config_path}' does not exist.")
        sys.exit(1)

    # Load configurations
    print("Loading configs...")
    try:
        framework_config = FrameworkConfig.from_file(str(framework_config_path))
        analysis_config = AnalysisConfig.from_file(str(analysis_config_path))
    except Exception as e:
        print(f"Error loading configs: {e}")
        sys.exit(1)

    # Override output directory if specified in CLI
    if args.output:
        analysis_config.output_dir = args.output

    # Load documents
    print(f"Loading documents from {input_dir}/{args.pattern}...")
    try:
        documents = load_documents(
            directory_path=str(input_dir),
            pattern=args.pattern,
            limit=args.limit
        )
    except Exception as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)

    if not documents:
        print(f"No documents found matching '{args.pattern}' in '{input_dir}'.")
        sys.exit(1)

    print(f"Found {len(documents)} documents. Total tokens: {sum(d.token_count for d in documents):,}")

    # Run analysis
    print("\nStarting Fractal Summarization...")
    print("=" * 60)
    
    summarizer = FractalSummarizer(framework_config)
    
    try:
        result, metadata = summarizer.run(documents, analysis_config)
        
        print("=" * 60)
        print("\nAnalysis Complete!")
        print(f"Total Layers: {metadata.total_layers}")
        print(f"Total Cost: ${metadata.total_cost_usd:.2f}")
        print(f"Output written to: {analysis_config.output_dir}")
        
    except Exception as e:
        print(f"\nAnalysis Failed: {e}")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
