#!/usr/bin/env python3
"""
Quick start example for MIVA Studio.

This example demonstrates the basic usage of the MIVA Studio pipeline:
1. Initialize pipeline with configuration
2. Generate images with identity conditioning
3. Inspect results

Run with:
    python examples/quick_start.py --subject_id alice --output_dir ./outputs
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from miva.pipeline import MIVAPipeline
from miva.config import get_config


def main():
    parser = argparse.ArgumentParser(description="MIVA Studio Quick Start Example")
    parser.add_argument("--subject_id", default="demo_subject", help="Subject ID to generate for")
    parser.add_argument("--prompt", default="professional portrait, studio lighting", help="Generation prompt")
    parser.add_argument("--num_outputs", type=int, default=1, help="Number of outputs to generate")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--config", default="config/miva_default.yaml", help="Config file")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  MIVA Studio — Quick Start Example")
    print("="*60 + "\n")
    
    # Load configuration
    try:
        config = get_config()
        print(f"✓ Configuration loaded (system v{config.system_version})")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return 1
    
    # Initialize pipeline
    try:
        pipeline = MIVAPipeline(config=config)
        print(f"✓ Pipeline initialized")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return 1
    
    # Health check
    print("\nRunning health check...")
    if not pipeline.health_check():
        print("✗ Health check failed")
        return 1
    print("✓ Health check passed")
    
    # Generate
    print(f"\nGenerating {args.num_outputs} image(s) for subject '{args.subject_id}'...")
    print(f"  Prompt: {args.prompt}")
    
    try:
        results = pipeline.generate(
            subject_id=args.subject_id,
            prompt=args.prompt,
            num_outputs=args.num_outputs,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return 1
    
    # Report results
    print("\n" + "-"*60)
    print("Results:")
    print("-"*60)
    
    success_count = 0
    for result in results:
        print(f"\nSession: {result.session_id[:8]}...")
        if result.success:
            print(f"  Status: ✓ SUCCESS")
            print(f"  Output: {result.output_path}")
            print(f"  Identity score: {result.final_identity_score:.4f}")
            print(f"  Attempts: {result.attempts}")
            success_count += 1
        else:
            print(f"  Status: ✗ FAILED")
            print(f"  Reason: {result.failure_reason}")
            print(f"  Attempts: {result.attempts}")
    
    print("\n" + "="*60)
    print(f"Summary: {success_count}/{len(results)} successful")
    print("="*60 + "\n")
    
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
