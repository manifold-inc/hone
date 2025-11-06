#!/usr/bin/env python3
"""
Simple LLM Miner using vLLM
Supports two-phase execution: prep and inference
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

print("Starting inference script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Arguments: {sys.argv}")


def load_input_data(input_dir: Path) -> Dict[str, Any]:
    """Load input data from mounted directory."""
    input_path = input_dir / "input.json"
    
    print(f"Looking for input at: {input_path}")
    print(f"Input dir exists: {input_dir.exists()}")
    if input_dir.exists():
        print(f"Input dir contents: {list(input_dir.iterdir())}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Show first few lines
    data_str = json.dumps(data, indent=2)
    lines = data_str.split('\n')
    preview = '\n'.join(lines[:10])
    
    print(f"\nInput data preview (first 10 lines):")
    print(preview)
    if len(lines) > 10:
        print(f"... ({len(lines) - 10} more lines)")
    
    return data


def save_output_data(results: Dict[str, Any], output_dir: Path):
    """Save output data to mounted directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved output to: {output_path}")


def run_prep_phase(input_dir: Path, output_dir: Path):
    """Prep phase: Download model and prepare environment."""
    print("\n" + "=" * 60)
    print("PREP PHASE - Downloading model")
    print("=" * 60)
    
    # Load input to see what model we need
    input_data = load_input_data(input_dir)
    model_name = input_data.get("model", "Qwen/Qwen3-0.6B")
    
    print(f"\n[1/2] Model to download: {model_name}")
    
    # Download model during prep phase
    from huggingface_hub import snapshot_download
    
    cache_dir = Path("/app/models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[2/2] Downloading model to {cache_dir}...")
    print("(This requires internet access)")
    
    try:
        snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            resume_download=True
        )
        print("✓ Model downloaded successfully!")
    except Exception as e:
        print(f"WARNING: Could not download model: {e}")
        print("Will try to use cached version in inference phase")
    
    # Save prep results
    prep_results = {
        "phase": "prep",
        "model": model_name,
        "status": "completed",
        "cache_dir": str(cache_dir)
    }
    save_output_data(prep_results, output_dir)
    
    print("\n" + "=" * 60)
    print("PREP PHASE COMPLETED")
    print("=" * 60)


def run_inference_phase(input_dir: Path, output_dir: Path):
    """Inference phase: Run actual inference (no internet)."""
    print("\n" + "=" * 60)
    print("INFERENCE PHASE - Running inference")
    print("=" * 60)
    
    # Load input data
    print("\n[1/4] Loading input data...")
    input_data = load_input_data(input_dir)
    
    # Get parameters
    prompts = input_data.get("prompts", ["Hello! How are you today?"])
    model_name = input_data.get("model", "facebook/opt-125m")
    max_tokens = input_data.get("max_tokens", 50)
    temperature = input_data.get("temperature", 0.7)
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Temperature: {temperature}")
    
    # Check GPU
    print(f"\n[2/4] Checking GPU availability...")
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU Available: {gpu_name} (count: {gpu_count})")
        else:
            print("⚠ No GPU detected, will use CPU")
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")
        gpu_available = False
    
    # Initialize model
    print(f"\n[3/4] Loading model: {model_name}")
    
    try:
        from vllm import LLM, SamplingParams
        
        llm = LLM(
            model=model_name,
            download_dir="/app/models",
            dtype="half" if gpu_available else "float32",
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            trust_remote_code=True,
            tensor_parallel_size=1
        )
        print("✓ Model loaded successfully!")
        
        # Set up sampling
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens
        )
        
        # Run inference
        print(f"\n[4/4] Running inference on {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Process results
        results = {
            "phase": "inference",
            "model": model_name,
            "gpu_used": gpu_available,
            "num_prompts": len(prompts),
            "outputs": []
        }
        
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            
            results["outputs"].append({
                "prompt": prompt,
                "generated_text": generated_text,
                "num_tokens": len(output.outputs[0].token_ids)
            })
            
            print(f"\n  Prompt {i+1}: {prompt[:60]}...")
            print(f"  Response: {generated_text[:80]}...")
        
    except Exception as e:
        print(f"\n⚠ Could not run vLLM inference: {e}")
        print("Generating simple response instead...")
        
        # Fallback: simple response
        results = {
            "phase": "inference",
            "model": model_name,
            "gpu_used": False,
            "num_prompts": len(prompts),
            "outputs": []
        }
        
        for prompt in prompts:
            results["outputs"].append({
                "prompt": prompt,
                "generated_text": f"Hello! This is a test response to: {prompt}",
                "num_tokens": 10
            })
    
    # Save results
    save_output_data(results, output_dir)
    
    print("\n" + "=" * 60)
    print("INFERENCE PHASE COMPLETED")
    print("=" * 60)


def main():
    """Main entry point with phase argument."""
    parser = argparse.ArgumentParser(description="LLM Inference Script")
    parser.add_argument(
        "--phase",
        choices=["prep", "inference"],
        required=True,
        help="Execution phase"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory path"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print(f"\nPhase: {args.phase}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    if args.phase == "prep":
        run_prep_phase(input_dir, output_dir)
    else:
        run_inference_phase(input_dir, output_dir)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)