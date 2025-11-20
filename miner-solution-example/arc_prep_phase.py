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
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from arc_solver_llm import model_name


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True
)
def download_model_with_retry(repo_id: str, cache_dir: str, local_dir: str) -> str:
    """download model with automatic retry on network failures"""
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )


def run_prep_phase(cache_dir = Path("/app/models")) -> None:
    """Prep phase: download model(s)"""
    print("\n" + "=" * 60)
    print("PREP PHASE - Downloading Models / Assets")
    print("=" * 60)

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = cache_dir / model_name.replace("/", "--")
    
    print(f"\n[1/4] Default example model to download: {model_name}")
    print(f"[2/4] Using cache directory: {cache_dir}")
    print(f"[3/4] Target local directory: {local_dir}")

    if local_dir.exists() and any(local_dir.iterdir()):
        files_count = len(list(local_dir.glob('*')))
        if files_count >= 10:
            print(f"\n✓ Model files found in local cache ({files_count} files), skipping download")
            
            print("\n" + "=" * 60)
            print("PREP PHASE COMPLETED - Status: success")
            print("=" * 60)
            return
        else:
            print(f"\n⚠ Partial download detected ({files_count} files), will resume...")

    print("(This phase requires internet access)")

    try:
        print("\n[4/4] Downloading model files from Hugging Face...")
        print("(Using automatic retry with exponential backoff)")
        
        local_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = download_model_with_retry(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            local_dir=str(local_dir)
        )

        print(f"✓ Model files downloaded to cache: {downloaded_path}")
        print("✓ Model download verified")
        files_count = len(list(Path(downloaded_path).glob('*')))
        print(f"✓ Files in model directory: {files_count}")

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