"""
ARC-AGI-2 Sandbox Runner Entry Point

This preserves the old CLI:

    python arc_main.py --phase prep --input ... --output ...
    python arc_main.py --phase inference --input ... --output ...

Under the hood it just calls the dedicated phase scripts
"""

import argparse
import sys
from pathlib import Path

from arc_prep_phase import run_prep_phase
from arc_inference_phase import run_inference_phase


def main() -> int:
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Inference Script (wrapper)")
    parser.add_argument(
        "--phase",
        choices=["prep", "inference"],
        required=True,
        help="Execution phase",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory path",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print(f"\nPhase: {args.phase}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    if args.phase == "prep":
        run_prep_phase()
    else:
        run_inference_phase(input_dir, output_dir)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR (wrapper): {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
