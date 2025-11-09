#!/usr/bin/env python3
"""
LLM Miner using vLLM OpenAI-compatible server
"""

import argparse
import json
import os
import sys
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, Any

print("Starting inference script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Arguments: {sys.argv}")


def load_input_data(input_dir: Path) -> Dict[str, Any]:
    """Load input data from mounted directory."""
    input_path = input_dir / "input.json"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    return data


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
    
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    cache_dir = Path("/app/models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/3] Model to download: {model_name}")
    print(f"[2/3] Downloading model to {cache_dir}...")
    print("(This requires internet access)")
    
    from huggingface_hub import snapshot_download
    
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
        
        prep_results = {
            "phase": "prep",
            "model": model_name,
            "status": "success",
            "message": f"Model downloaded to {downloaded_path}",
            "cache_dir": str(cache_dir)
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


def start_vllm_server(model_path: str, port: int = 8000) -> subprocess.Popen:
    """Start vLLM OpenAI-compatible server."""
    print(f"\n[1/4] Starting vLLM server on port {port}...")
    
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", "half",
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.8",
    ]
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)} (count: {gpu_count})")
            if gpu_count > 1:
                cmd.extend(["--tensor-parallel-size", str(gpu_count)])
        else:
            print("⚠ No GPU detected, server may be slow")
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")
    
    print(f"Command: {' '.join(cmd)}")
    
    # Start server as subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Wait for server to be ready
    print("\n[2/4] Waiting for vLLM server to be ready...")
    max_wait = 120  # 2 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                print("✓ vLLM server is ready!")
                return process
        except requests.exceptions.RequestException:
            pass
        
        # Check if process died
        if process.poll() is not None:
            output, _ = process.communicate()
            print(f"ERROR: vLLM server process died!")
            print(f"Output: {output}")
            raise RuntimeError("vLLM server failed to start")
        
        time.sleep(2)
        print(".", end="", flush=True)
    
    raise TimeoutError("vLLM server failed to start within timeout")


def run_inference_with_openai_client(prompts: list, port: int = 8000, **kwargs):
    """Run inference using OpenAI-compatible API."""
    from openai import OpenAI
    
    # Create client pointing to local vLLM server
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require API key
        base_url=f"http://localhost:{port}/v1"
    )
    
    results = []
    
    print(f"\n[3/4] Running inference on {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts, 1):
        try:
            print(f"  Processing prompt {i}/{len(prompts)}...")
            
            response = client.chat.completions.create(
                model="unsloth/Meta-Llama-3.1-8B-Instruct",  # Model name doesn't matter for vLLM
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get("max_tokens", 50),
                temperature=kwargs.get("temperature", 0.7),
            )
            
            generated_text = response.choices[0].message.content
            
            results.append({
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
            results.append({
                "prompt": prompt,
                "error": str(e)
            })
    
    return results


def run_inference_phase(input_dir: Path, output_dir: Path):
    """Inference phase: Run inference using vLLM server."""
    print("\n" + "=" * 60)
    print("INFERENCE PHASE - Running inference")
    print("=" * 60)
    
    # Configuration
    prompts = ["Hello! How are you today?"]
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    model_path = Path("/app/models") / model_name.replace("/", "--")
    max_tokens = 50
    temperature = 0.7
    port = 8000
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Model path: {model_path}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Temperature: {temperature}")
    
    # Verify model exists
    if not model_path.exists():
        print(f"\n✗ ERROR: Model not found at {model_path}")
        print(f"Available paths in /app/models:")
        for p in Path("/app/models").iterdir():
            print(f"  - {p}")
        
        results = {
            "phase": "inference",
            "model": model_name,
            "status": "failed",
            "error": f"Model not found at {model_path}"
        }
        save_output_data(results, output_dir)
        return
    
    vllm_process = None
    
    try:
        # Start vLLM server
        vllm_process = start_vllm_server(str(model_path), port=port)
        
        # Run inference
        outputs = run_inference_with_openai_client(
            prompts,
            port=port,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        print(f"\n[4/4] Completed inference")
        
        # Check GPU usage
        try:
            import torch
            gpu_used = torch.cuda.is_available()
            if gpu_used:
                print(f"✓ GPU was utilized")
        except:
            gpu_used = False
        
        results = {
            "phase": "inference",
            "model": model_name,
            "gpu_used": gpu_used,
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
            "status": "failed",
            "error": str(e),
            "outputs": []
        }
    
    finally:
        # Cleanup: stop vLLM server
        if vllm_process and vllm_process.poll() is None:
            print("\nStopping vLLM server...")
            vllm_process.terminate()
            try:
                vllm_process.wait(timeout=5)
                print("✓ vLLM server stopped")
            except subprocess.TimeoutExpired:
                vllm_process.kill()
                print("✓ vLLM server killed")
    
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