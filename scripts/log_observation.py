"""Academic Observation Logger for LLM Course Evaluation (24AM7PCLLM).

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning

This utility records manual qualitative and empirical observations across
demonstrated test cases into results/observations.csv.
"""

import os
import csv
import argparse
from typing import List, Dict, Any

OBSERVATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "observations.csv"
)

FIELDNAMES = [
    "observation_id",
    "test_case_id",
    "configuration",
    "expected_behavior",
    "actual_output",
    "status",
    "remarks"
]


def load_observations(filepath: str = OBSERVATIONS_PATH) -> List[Dict[str, str]]:
    """Load all recorded observations from CSV."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_observation(
    test_case_id: str,
    configuration: str,
    expected_behavior: str,
    actual_output: str,
    status: str = "PASS",
    remarks: str = "",
    filepath: str = OBSERVATIONS_PATH
) -> str:
    """Append a new observation row to results/observations.csv."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    existing = load_observations(filepath)
    next_num = len(existing) + 1
    obs_id = f"OBS-{next_num:02d}"

    new_row = {
        "observation_id": obs_id,
        "test_case_id": test_case_id.strip(),
        "configuration": configuration.strip(),
        "expected_behavior": expected_behavior.strip(),
        "actual_output": actual_output.strip(),
        "status": status.strip().upper(),
        "remarks": remarks.strip()
    }

    file_exists = os.path.exists(filepath)
    with open(filepath, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists or os.path.getsize(filepath) == 0:
            writer.writeheader()
        writer.writerow(new_row)

    return obs_id


def print_observations(filepath: str = OBSERVATIONS_PATH) -> None:
    """Print a clean CLI table of all recorded observations."""
    obs = load_observations(filepath)
    if not obs:
        print("No observations recorded yet.")
        return

    print(f"\n{'ID':<8} | {'TC':<6} | {'Status':<6} | {'Configuration':<30} | {'Remarks'}")
    print("-" * 90)
    for row in obs:
        print(f"{row['observation_id']:<8} | {row['test_case_id']:<6} | {row['status']:<6} | {row['configuration']:<30} | {row['remarks'][:40]}...")
    print(f"\nTotal Recorded Observations: {len(obs)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record academic observations for BMSCE 24AM7PCLLM project.")
    parser.add_argument("--list", action="store_true", help="List all currently recorded observations")
    parser.add_argument("--test_case", type=str, help="Test Case ID (e.g. TC1, TC2)")
    parser.add_argument("--config", type=str, help="Configuration (e.g. 'RAG + few-shot', 'Baseline')")
    parser.add_argument("--expected", type=str, help="Expected behavior / output")
    parser.add_argument("--actual", type=str, help="Actual model output / behavior")
    parser.add_argument("--status", type=str, default="PASS", choices=["PASS", "FAIL"], help="Verdict (PASS/FAIL)")
    parser.add_argument("--remarks", type=str, default="", help="Academic / pedagogical remarks")
    args = parser.parse_args()

    if args.list:
        print_observations()
        return

    if args.test_case and args.config and args.expected and args.actual:
        obs_id = save_observation(
            test_case_id=args.test_case,
            configuration=args.config,
            expected_behavior=args.expected,
            actual_output=args.actual,
            status=args.status,
            remarks=args.remarks
        )
        print(f"Successfully recorded observation {obs_id} to {OBSERVATIONS_PATH}")
    else:
        # Interactive mode
        print("\n--- Academic Observation Entry Form (24AM7PCLLM) ---")
        tc = input("Test Case ID (e.g. TC1, TC2, TC3): ").strip() or "TC1"
        cfg = input("Configuration (e.g. RAG + few-shot): ").strip() or "RAG + few-shot"
        exp = input("Expected Behavior: ").strip() or "Question adheres to Bloom taxonomy and retrieved context."
        act = input("Actual Output: ").strip() or "Generated compliant question grounded in syllabus concepts."
        st = input("Status [PASS/FAIL] (default PASS): ").strip() or "PASS"
        rem = input("Academic Remarks: ").strip() or "Validated grounding and taxonomy compliance."

        obs_id = save_observation(tc, cfg, exp, act, st, rem)
        print(f"\nObservation recorded successfully with ID: {obs_id}")
        print_observations()


if __name__ == "__main__":
    main()

