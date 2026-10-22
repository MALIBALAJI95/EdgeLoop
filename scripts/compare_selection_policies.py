#!/usr/bin/env python3
"""Run selective upload policy comparison against research stream."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgeloop.selection.evaluator import SelectionPolicyEvaluator


def main():
    evaluator = SelectionPolicyEvaluator()
    print("Evaluating selection policies with upload budget = 203 records...\n")
    results = evaluator.compare_all_policies(budget_records=203)
    df = evaluator.to_dataframe(results)
    print(df.to_string(index=False))

    print("\nDetailed Breakdown by Condition:")
    for strat, rep in results.items():
        print(f"[{strat.upper()}] Condition counts: {rep.condition_distribution}")


if __name__ == "__main__":
    main()
