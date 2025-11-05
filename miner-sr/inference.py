#!/usr/bin/env python3
"""
Simple LLM Miner using vLLM
Loads a model and performs inference on input data.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

from vllm import LLM, SamplingParams


def load_input_data() -> Dict[str, Any]:
    """Load input data from mounted directory."""
    input_path = Path("/workspace/input/input.json")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded input data: {len(data)} items")
    return data


def save_output_data(results: Dict[str, Any]):
    """Save output data to mounted directory."""
    output_dir = Path("/workspace/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved output to: {output_path}")


def main():
    """Main inference function."""
    print("=" * 60)
    print("Simple vLLM Miner - Starting")
    print("=" * 60)
    
    # Load input data
    print("\n[1/4] Loading input data...")
    input_data = load_input_data()
    
    # Get prompts from input
    prompts = input_data.get("prompts", ["Hello, how are you?"])
    model_name = input_data.get("model", "facebook/opt-125m")
    max_tokens = input_data.get("max_tokens", 100)
    temperature = input_data.get("temperature", 0.7)
    
    print(f"Model: {model_name}")
    print(f"Prompts: {len(prompts)}")
    
    # Initialize model
    print(f"\n[2/4] Loading model: {model_name}")
    print("(This may take a few minutes on first run...)")
    
    llm = LLM(
        model=model_name,
        download_dir="/workspace/models",  # Cache models here
        dtype="half",  # Use FP16 for faster inference
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        max_model_len=2048,  # Max sequence length
        trust_remote_code=True
    )
    
    print("Model loaded successfully!")
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stop=None
    )
    
    # Run inference
    print(f"\n[3/4] Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Process results
    results = {
        "model": model_name,
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
        
        print(f"\nPrompt {i+1}: {prompt[:50]}...")
        print(f"Response: {generated_text[:100]}...")
    
    # Save results
    print(f"\n[4/4] Saving results...")
    save_output_data(results)
    
    print("\n" + "=" * 60)
    print("Inference completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)