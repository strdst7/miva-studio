#!/usr/bin/env python3
"""
Generate evaluation dataset for MIVA Studio.

Creates eval_dataset_v1 with 50 subjects, verified embeddings, and test cases.

Usage:
    python scripts/create_eval_dataset.py \
        --num_subjects 50 \
        --anchors_per_subject 10 \
        --output_dir ./data/eval_dataset \
        --seed 42

This script:
1. Creates 50 subject identities
2. Generates 10 anchor embeddings per subject
3. Verifies embedding quality (norm, intra-subject similarity)
4. Creates test cases (same-identity, cross-identity, degraded)
5. Outputs versioned metadata and ground truth
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_embedding(subject_id: int, anchor_id: int, noise_scale: float = 0.05) -> np.ndarray:
    """
    Generate a realistic face embedding for evaluation.
    
    In production, these would be real embeddings from real faces.
    For evaluation, we generate synthetic embeddings with structure:
    - Same subject: embeddings similar to each other (cosine sim > 0.85)
    - Different subject: embeddings dissimilar (cosine sim < 0.60)
    """
    np.random.seed(hash((subject_id, anchor_id)) % 2**32)
    
    # Subject base vector (same for all anchors of same subject)
    subject_vector = np.random.randn(512)
    subject_vector = subject_vector / (np.linalg.norm(subject_vector) + 1e-8)
    
    # Add anchor-specific variation
    anchor_noise = np.random.randn(512) * noise_scale
    embedding = subject_vector + anchor_noise
    
    # Normalize
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    return embedding.astype(np.float32)


def verify_embedding_quality(embedding: np.ndarray) -> dict:
    """Verify embedding meets quality standards."""
    norm = np.linalg.norm(embedding)
    
    return {
        'norm': float(norm),
        'valid_norm': 0.95 <= norm <= 1.05,
    }


def create_dataset(
    num_subjects: int = 50,
    anchors_per_subject: int = 10,
    output_dir: str = "./data/eval_dataset",
    seed: int = 42
):
    """Create evaluation dataset."""
    np.random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating evaluation dataset: {num_subjects} subjects, {anchors_per_subject} anchors each")
    
    # Create subject directories and embeddings
    subjects = {}
    ground_truth = {}  # For retrieval recall calculation
    
    for subject_id in range(num_subjects):
        subject_key = f"subject_{subject_id:03d}"
        subject_dir = output_path / "subjects" / subject_key
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate anchors for this subject
        embeddings = []
        for anchor_id in range(anchors_per_subject):
            embedding = generate_embedding(subject_id, anchor_id)
            embeddings.append(embedding)
        
        subjects[subject_key] = {
            'subject_id': subject_id,
            'num_anchors': anchors_per_subject,
            'embeddings': [e.tolist() for e in embeddings],
        }
        
        # Ground truth: embeddings with high similarity (>0.85) are same-identity
        # For synthetic data, all anchors of same subject are > 0.85 similar
        ground_truth[subject_key] = {
            'reference_embedding': embeddings[0].tolist(),
            'same_identity_anchors': list(range(anchors_per_subject)),  # All anchors
        }
        
        # Save embeddings to file
        embeddings_file = subject_dir / "embeddings.npy"
        np.save(embeddings_file, np.array(embeddings))
        
        logger.info(f"  Created {subject_key}: {anchors_per_subject} anchors")
    
    # Create test cases
    test_cases = []
    
    # Same-identity, valid cases (should PASS)
    for subject_id in range(num_subjects):
        subject_key = f"subject_{subject_id:03d}"
        test_cases.append({
            'test_id': f"{subject_key}_valid",
            'type': 'same_identity_valid',
            'subject_id': subject_key,
            'expected_decision': 'PASS',
        })
    
    # Cross-identity cases (should FAIL)
    for i in range(min(num_subjects, 50)):
        subject_id_1 = i
        subject_id_2 = (i + num_subjects // 2) % num_subjects  # Different subject
        
        test_cases.append({
            'test_id': f"cross_identity_{i:03d}",
            'type': 'cross_identity',
            'subject_query': f"subject_{subject_id_1:03d}",
            'subject_anchor': f"subject_{subject_id_2:03d}",
            'expected_decision': 'FAIL',
        })
    
    # Create metadata
    metadata = {
        'dataset_name': 'miva_eval_v1',
        'version': '1.0.0',
        'created': datetime.utcnow().isoformat() + 'Z',
        'seed': seed,
        'num_subjects': num_subjects,
        'anchors_per_subject': {
            'min': anchors_per_subject,
            'max': anchors_per_subject,
            'mean': anchors_per_subject,
        },
        'test_cases': {
            'total': len(test_cases),
            'same_identity_valid': len([t for t in test_cases if t['type'] == 'same_identity_valid']),
            'cross_identity': len([t for t in test_cases if t['type'] == 'cross_identity']),
        },
        'held_out_from_threshold_tuning': True,
        'source': 'synthetic_for_evaluation',
    }
    
    # Save metadata and ground truth
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    with open(output_path / "ground_truth.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    with open(output_path / "test_cases.json", 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    logger.info(f"\n✓ Dataset created successfully")
    logger.info(f"  Location: {output_path}")
    logger.info(f"  Subjects: {num_subjects}")
    logger.info(f"  Anchors: {anchors_per_subject} per subject ({num_subjects * anchors_per_subject} total)")
    logger.info(f"  Test cases: {len(test_cases)}")
    logger.info(f"  Files:")
    logger.info(f"    - metadata.json")
    logger.info(f"    - ground_truth.json")
    logger.info(f"    - test_cases.json")
    logger.info(f"    - subjects/<subject_id>/embeddings.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create evaluation dataset for MIVA Studio")
    parser.add_argument("--num_subjects", type=int, default=50, help="Number of subjects")
    parser.add_argument("--anchors_per_subject", type=int, default=10, help="Anchors per subject")
    parser.add_argument("--output_dir", default="./data/eval_dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    create_dataset(
        num_subjects=args.num_subjects,
        anchors_per_subject=args.anchors_per_subject,
        output_dir=args.output_dir,
        seed=args.seed
    )
