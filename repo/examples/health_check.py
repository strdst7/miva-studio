#!/usr/bin/env python3
"""
Health check for MIVA Studio.

Validates that all components are properly installed and configured.

Run with:
    python examples/health_check.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from miva.config import get_config
from miva.pipeline import MIVAPipeline
from miva.guardrails import GuardrailEvaluator


def check_pytorch():
    """Check PyTorch installation."""
    try:
        print(f"PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU memory: {props.total_memory / 1e9:.1f} GB")
        return True
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        return False


def check_config():
    """Check configuration loading."""
    try:
        config = get_config()
        print(f"Configuration loaded successfully")
        print(f"  System version: {config.system_version}")
        print(f"  Pipeline version: {config.pipeline_version}")
        print(f"  Identity threshold: {config.guardrails.identity.threshold}")
        print(f"  Max attempts: {config.guardrails.agent.max_regeneration_attempts}")
        
        if not config.validate():
            print("✗ Configuration validation failed")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Configuration check failed: {e}")
        return False


def check_pipeline():
    """Check pipeline initialization."""
    try:
        config = get_config()
        pipeline = MIVAPipeline(config=config)
        print(f"Pipeline initialized successfully")
        
        if not pipeline.health_check():
            print("✗ Pipeline health check failed")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Pipeline check failed: {e}")
        return False


def check_guardrails():
    """Check guardrail evaluation."""
    try:
        config = get_config()
        evaluator = GuardrailEvaluator(config)
        
        # Test with dummy data
        dummy_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        dummy_embedding = np.random.randn(512).astype(np.float32)
        dummy_anchors = [np.random.randn(512).astype(np.float32) for _ in range(3)]
        
        decision = evaluator.evaluate(
            generated_image=dummy_image,
            generated_embedding=dummy_embedding,
            anchor_embeddings=dummy_anchors,
            attempt_number=1
        )
        
        print(f"Guardrail evaluation successful")
        print(f"  Decision: {decision.action.value}")
        print(f"  Should retry: {decision.should_retry}")
        
        # Critical invariant check
        if decision.action.value == "HARD_STOP":
            if decision.should_retry == True:
                print("✗ CRITICAL: HARD_STOP has should_retry=True (invariant violation)")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Guardrail check failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("  MIVA Studio Health Check")
    print("="*60 + "\n")
    
    checks = {
        "PyTorch": check_pytorch,
        "Configuration": check_config,
        "Pipeline": check_pipeline,
        "Guardrails": check_guardrails,
    }
    
    results = {}
    for name, check_fn in checks.items():
        print(f"\n{'─'*60}")
        print(f"Checking: {name}")
        print('─'*60)
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Health Check Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed")
        print("="*60 + "\n")
        return 0
    else:
        print("✗ Some checks failed")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
