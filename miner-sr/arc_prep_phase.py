"""
ARC-AGI-2 PREP PHASE SCRIPT

This runs in the **prep container**, where internet access is allowed

- Download EVERYTHING you will need later in the inference phase:
    * LLM weights (Hugging Face, etc.).
    * Other model weights (vision models, HRMs, GNNs, etc.).
    * Any auxiliary data, vocab files, tokenizers...

You ARE allowed to:
- Change which models are downloaded
- Add more downloads (multiple models, toolchains, etc.)

You MUST NOT:
- Write outside the provided `output_dir`
- Change the local cache paths


The validator calls `run_prep_phase(input_dir, output_dir)` or the CLI in this file
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from arc_solver_llm import model_name


def run_prep_phase(cache_dir = Path("/app/models")) -> None:
    """Prep phase: download model(s) and optionally validate input data."""
    print("\n" + "=" * 60)
    print("PREP PHASE - Downloading Models / Assets")
    print("=" * 60)

    
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Default example model to download: {model_name}")
    print(f"[2/4] Using cache directory: {cache_dir}")
    print("(This phase is allowed to use the internet.)")

    try:
        # -----------------------
        # Download the example model
        # -----------------------
        print("\n[4/4] Downloading model files from Hugging Face...")

        # Make sure to use the cache dir as root for your downloaded ressources (models, datasets, etc)
        local_dir = cache_dir / model_name.replace("/", "--")
        local_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )

        print(f"✓ Model files downloaded to cache: {downloaded_path}")
        print("✓ Model download verified")
        print(f"✓ Files in model directory: {len(list(Path(downloaded_path).glob('*')))}")

        prep_results = {
            "phase": "prep",
            "model": model_name,
            "status": "success",
            "message": f"Model downloaded to {downloaded_path}",
            "cache_dir": str(cache_dir),
        }

    except Exception as e:
        print(f"ERROR: Could not complete prep phase: {e}")
        import traceback

        traceback.print_exc()
        prep_results = {
            "phase": "prep",
            "model": model_name,
            "status": "failed",
            "message": str(e),
        }

    print("\n" + "=" * 60)
    print(f"PREP PHASE COMPLETED - Status: {prep_results['status']}")
    print("=" * 60)

    if prep_results["status"] == "failed":
        sys.exit(1)


def _cli() -> int:
    """CLI entry point for running only the prep phase."""
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Prep Phase Script")
    parser.add_argument("--input", type=str, required=True, help="Input directory path")
    parser.add_argument("--output", type=str, required=True, help="Output directory path")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print(f"\nPhase: prep")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    run_prep_phase()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_cli())
    except Exception as e:
        print(f"\nERROR (prep phase): {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
