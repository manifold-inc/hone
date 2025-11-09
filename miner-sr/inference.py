#!/usr/bin/env python3
"""
LLM Miner using vLLM docker container
"""

import argparse
import json
import os
import sys
import requests
from pathlib import Path
from typing import Dict, Any

print("Starting inference script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Arguments: {sys.argv}")


def save_output_data(results: Dict[str, Any], output_dir: Path):
    """Save output data to mounted directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved output to: {output_path}")


def run_prep_phase(input_dir: Path, output_dir: Path):
    """Prep phase: Download model."""
    print("\n" + "=" * 60)
    print("PREP PHASE - Downloading model")
    print("=" * 60)
    
    from huggingface_hub import snapshot_download
    
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    cache_dir = Path("/app/models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/3] Model to download: {model_name}")
    print(f"[2/3] Downloading model to {cache_dir}...")
    print("(This requires internet access)")
    
    try:
        print("\n[3/3] Downloading model files...")
        
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
        
        print(f"✓ Model files downloaded to: {downloaded_path}")
        print("✓ Model download verified")
        print(f"✓ Files in model directory: {len(list(Path(downloaded_path).glob('*')))}")
        
        # Save model info for vLLM to use
        model_info = {
            "model_name": model_name,
            "model_path": str(local_dir),
            "downloaded_path": downloaded_path
        }
        
        # Save model info to output directory
        model_info_path = output_dir / "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        prep_results = {
            "phase": "prep",
            "model": model_name,
            "status": "success",
            "message": f"Model downloaded to {downloaded_path}",
            "cache_dir": str(cache_dir),
            "model_info": model_info
        }
        
    except Exception as e:
        print(f"ERROR: Could not download model: {e}")
        import traceback
        traceback.print_exc()
        prep_results = {
            "phase": "prep",
            "model": model_name,
            "status": "failed",
            "message": str(e)
        }
    
    save_output_data(prep_results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"PREP PHASE COMPLETED - Status: {prep_results['status']}")
    print("=" * 60)


def run_inference_phase(input_dir: Path, output_dir: Path):
    """Inference phase: Use external vLLM API for inference."""
    print("\n" + "=" * 60)
    print("INFERENCE PHASE - Using vLLM API")
    print("=" * 60)
    
    from openai import OpenAI
    
    # Configuration
    prompts = ["Hello! How are you today?"]
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    max_tokens = 50
    temperature = 0.7
    
    # Get vLLM API endpoint from environment
    vllm_api_base = os.getenv('VLLM_API_BASE', 'http://localhost:6919')
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  vLLM API: {vllm_api_base}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Temperature: {temperature}")
    
    try:
        # Create client pointing to vLLM server
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"{vllm_api_base}/v1"
        )
        
        # Test connection
        print(f"\nTesting connection to vLLM API...")
        response = requests.get(f"{vllm_api_base}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Connected to vLLM API")
        else:
            raise RuntimeError(f"vLLM API unhealthy: {response.status_code}")
        
        # Run inference
        outputs = []
        print(f"\nRunning inference on {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts, 1):
            try:
                print(f"  Processing prompt {i}/{len(prompts)}...")
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                generated_text = response.choices[0].message.content
                
                outputs.append({
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                })
                
                print(f"    ✓ Generated {response.usage.completion_tokens} tokens")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                outputs.append({
                    "prompt": prompt,
                    "error": str(e)
                })
        
        print(f"\nCompleted inference")
        
        results = {
            "phase": "inference",
            "model": model_name,
            "vllm_api": vllm_api_base,
            "num_prompts": len(prompts),
            "status": "success",
            "outputs": outputs
        }
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        
        results = {
            "phase": "inference",
            "model": model_name,
            "vllm_api": vllm_api_base,
            "status": "failed",
            "error": str(e),
            "outputs": []
        }
    
    save_output_data(results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"INFERENCE PHASE COMPLETED - Status: {results['status']}")
    print("=" * 60)


def main():
    """Main entry point."""
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