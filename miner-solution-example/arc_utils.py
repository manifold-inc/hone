"""
Shared utilities for ARC-AGI-2 Sandbox Runner

This file contains generic helper functions that the validator expects
to behave in a certain way:

- `load_input_data(input_dir)`:
    * MUST read the dataset from `miner_current_dataset.json`
    * MUST return a dict that has at least a `"tasks"` key
    * ❗❗ You SHOULD NOT change this ❗❗

- `save_output_data(results, output_dir)`:
    * MUST write a `results.json` file in `output_dir`
    * The validator expects a JSON object, not arbitrary text
    * ❗❗ You SHOULD NOT change this ❗❗

Only modify this file if you know exactly what the validator expects.
"""

import json
import socket
from pathlib import Path
from typing import Dict, Any, List


def save_output_data(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Save output data to mounted directory as `results.json`
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved output to: {output_path}")


def load_input_data(input_dir: Path) -> Dict[str, Any]:
    """
    Load input data from mounted directory

    Current convention:
    - File: `miner_current_dataset.json`
    - Top-level key: `"tasks"` (list of problems)
    """
    all_files = list(input_dir.glob("*"))
    print("All files in input dir: ", all_files)

    dataset_file = input_dir / "miner_current_dataset.json"

    if dataset_file.exists():
        print(f"Found dataset file: {dataset_file}")
        with open(dataset_file, "r") as f:
            data = json.load(f)
        return data

    raise FileNotFoundError(f"No input data found in {input_dir}")