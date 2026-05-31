"""
miva/evaluation/regression_detector.py
Regression detection for MIVA Studio.

Compares a current evaluation report against a baseline and flags regressions
exceeding thresholds defined in the system configuration.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from miva.config import get_config

logger = logging.getLogger(__name__)


def load_report(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def check_regression(baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """
    Check for performance regressions.
    Returns True if regression detected, False otherwise.
    """
    config = get_config()
    thresholds = config.evaluation.regression_thresholds
    
    regressions = []
    
    # 1. Compare identity score (mean)
    b_id = baseline.get("identity_score_mean", {}).get("mean", 0)
    c_id = current.get("identity_score_mean", {}).get("mean", 0)
    delta_id = c_id - b_id
    if delta_id < thresholds["identity_score_mean"]:
        regressions.append(f"Identity Score Mean: baseline={b_id:.4f}, current={c_id:.4f}, delta={delta_id:.4f} (limit {thresholds['identity_score_mean']})")

    # 2. Compare Retrieval Recall@3 (mean)
    b_rec = baseline.get("retrieval_recall_at_3", {}).get("mean", 0)
    c_rec = current.get("retrieval_recall_at_3", {}).get("mean", 0)
    delta_rec = c_rec - b_rec
    if delta_rec < thresholds["retrieval_recall_at_3"]:
        regressions.append(f"Recall@3: baseline={b_rec:.4f}, current={c_rec:.4f}, delta={delta_rec:.4f} (limit {thresholds['retrieval_recall_at_3']})")

    # 3. Compare Hard Stop Rate (lower is better, so delta > threshold is regression)
    b_hs = baseline.get("hard_stop_rate", 0)
    c_hs = current.get("hard_stop_rate", 0)
    delta_hs = c_hs - b_hs
    if delta_hs > thresholds["hard_stop_rate"]:
        regressions.append(f"Hard Stop Rate: baseline={b_hs:.4f}, current={c_hs:.4f}, delta={delta_hs:.4f} (limit {thresholds['hard_stop_rate']})")

    # 4. Compare False Positive Rate (lower is better)
    b_fp = baseline.get("false_positive_rate", {}).get("mean", 0)
    c_fp = current.get("false_positive_rate", {}).get("mean", 0)
    delta_fp = c_fp - b_fp
    if delta_fp > thresholds["false_positive_rate"]:
        regressions.append(f"False Positive Rate: baseline={b_fp:.4f}, current={c_fp:.4f}, delta={delta_fp:.4f} (limit {thresholds['false_positive_rate']})")

    if regressions:
        print("\n" + "!" * 70)
        print("  REGRESSION DETECTED")
        print("!" * 70)
        for r in regressions:
            print(f"  - {r}")
        print("!" * 70 + "\n")
        return True
    
    print("\n✓ No performance regressions detected against baseline.\n")
    return False


def main():
    parser = argparse.ArgumentParser(description="MIVA Studio Regression Detector")
    parser.add_argument("--baseline", required=True, help="Baseline report JSON")
    parser.add_argument("--current", required=True, help="Current report JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        baseline = load_report(args.baseline)
        current = load_report(args.current)
    except Exception as e:
        logger.error(f"Failed to load reports: {e}")
        sys.exit(1)

    has_regression = check_regression(baseline, current)
    
    if has_regression:
        sys.exit(1)


if __name__ == "__main__":
    main()
