#!/usr/bin/env python3
"""
ARC-AGI-2 Solver for Sandbox Runner
Prep phase: Downloads LLM model for vLLM
Inference phase: Solves ARC problems using vLLM (with fallback) and saves predictions
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import socket
from huggingface_hub import snapshot_download

print("Starting ARC-AGI-2 inference script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Arguments: {sys.argv}")


class ARCSolver:
    """
    ARC solver with vLLM and rule-based strategies
    """
    
    def __init__(self, use_vllm: bool = True):
        self.use_vllm = use_vllm
        self.vllm_available = False
        self.vllm_client = None
        self.vllm_model_name = None
        
        print("🔧 ARCSolver initialized")
        
        if use_vllm:
            self._init_vllm_client()
        
        # Rule-based strategies (fallback)
        self.strategies = [
            self._identity_transform,
            self._analyze_color_mapping,
            self._analyze_size_transform,
            self._analyze_pattern_transform,
            self._analyze_symmetry
        ]
    
    def _init_vllm_client(self):
        """Initialize vLLM client and discover actual model name"""
        try:
            from openai import OpenAI
            
            # Get vLLM API base from environment variable
            vllm_api_base = os.environ.get('VLLM_API_BASE', 'http://vllm-container:8000')
            print(f"🌐 Attempting to connect to vLLM at: {vllm_api_base}")
            
            self.vllm_client = OpenAI(
                base_url=f"{vllm_api_base}/v1",
                api_key="dummy"  # vLLM doesn't require real API key
            )
            
            # Test connection and get actual model name
            try:
                models = self.vllm_client.models.list()
                self.vllm_available = True
                model_ids = [m.id for m in models.data]
                print(f"✓ vLLM connection successful! Available models: {model_ids}")
                
                # Store the actual model name to use for completions
                if model_ids:
                    self.vllm_model_name = model_ids[0]
                    print(f"✓ Using model name: {self.vllm_model_name}")
                else:
                    print("⚠ No models available from vLLM")
                    self.vllm_available = False
                    self.vllm_model_name = None
            except Exception as e:
                print(f"⚠ vLLM connection test failed: {e}")
                print("  Will fall back to rule-based methods")
                self.vllm_available = False
                self.vllm_model_name = None
                
        except ImportError:
            print("⚠ OpenAI client not available, using rule-based methods only")
            self.vllm_available = False
            self.vllm_model_name = None
        except Exception as e:
            print(f"⚠ Failed to initialize vLLM client: {e}")
            self.vllm_available = False
            self.vllm_model_name = None
    
    def solve(self, train_examples: List[Dict], test_input: List[List[int]]) -> List[List[int]]:
        """
        Learn from training examples and apply to test input
        
        Args:
            train_examples: List of dicts with 'input' and 'output' grids
            test_input: The test input grid to solve
        """
        # Try vLLM first if available
        if self.vllm_available and self.vllm_client and self.vllm_model_name:
            try:
                result = self._solve_with_vllm(train_examples, test_input)
                if result and self._is_valid_output(result):
                    print("    ✓ Solved using vLLM")
                    return result
                else:
                    print("    ⚠ vLLM returned invalid output, falling back to rules")
            except Exception as e:
                print(f"    ⚠ vLLM solve failed: {e}, falling back to rules")
        
        # Fallback to rule-based methods
        if not train_examples:
            return [row[:] for row in test_input]
        
        transformation = self._identify_transformation(train_examples)
        
        if transformation and transformation.get("type"):
            return self._apply_learned_transformation(test_input, transformation)
        
        return self._apply_strategy(test_input, train_examples)
    
    def _solve_with_vllm(self, train_examples: List[Dict], test_input: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Use vLLM to solve the ARC problem
        
        Args:
            train_examples: Training examples
            test_input: Test input to solve
            
        Returns:
            Predicted output grid or None if failed
        """
        # Create prompt for the LLM
        prompt = self._create_arc_prompt(train_examples, test_input)
        
        try:
            # Call vLLM API with the actual model name
            response = self.vllm_client.chat.completions.create(
                model=self.vllm_model_name,  # Use discovered model name
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles. Analyze the pattern in the training examples and apply it to the test input. Return ONLY a valid JSON array representing the output grid."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            # LLM might wrap it in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse the output grid
            output_grid = json.loads(content)
            
            # Validate it's a proper grid
            if isinstance(output_grid, list) and all(isinstance(row, list) for row in output_grid):
                return output_grid
            else:
                print(f"    ⚠ vLLM returned non-grid format: {type(output_grid)}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"    ⚠ Failed to parse vLLM response as JSON: {e}")
            return None
        except Exception as e:
            print(f"    ⚠ vLLM API call failed: {e}")
            return None
    
    def _create_arc_prompt(self, train_examples: List[Dict], test_input: List[List[int]]) -> str:
        """Create a prompt for the LLM to solve the ARC problem"""
        prompt_parts = [
            "Solve this ARC puzzle by finding the pattern in the training examples.\n\n"
        ]
        
        # Add training examples
        prompt_parts.append("Training Examples:\n")
        for i, example in enumerate(train_examples, 1):
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Input:\n{json.dumps(example['input'])}")
            prompt_parts.append(f"Output:\n{json.dumps(example['output'])}\n")
        
        # Add test input
        prompt_parts.append("\nNow apply the pattern to this test input:")
        prompt_parts.append(f"Test Input:\n{json.dumps(test_input)}\n")
        
        prompt_parts.append("\nReturn ONLY the output grid as a JSON array. Do not include any explanation.")
        
        return "\n".join(prompt_parts)
    
    def _identify_transformation(self, examples: List[Dict]) -> Dict:
        """Analyze training examples to identify the transformation rule"""
        if not examples:
            return {}
        
        output_sizes = [(len(ex["output"]), len(ex["output"][0])) for ex in examples]
        same_output_size = len(set(output_sizes)) == 1
        
        size_preserved = all(
            len(ex["input"]) == len(ex["output"]) and 
            len(ex["input"][0]) == len(ex["output"][0])
            for ex in examples
        )
        
        color_mappings = []
        for ex in examples:
            in_colors = self._get_colors(ex["input"])
            out_colors = self._get_colors(ex["output"])
            color_mappings.append((in_colors, out_colors))
        
        transformation = {
            "same_output_size": same_output_size,
            "size_preserved": size_preserved,
            "color_mappings": color_mappings,
            "num_examples": len(examples)
        }
        
        if size_preserved and len(examples) > 0:
            rotation_count = sum(1 for ex in examples if self._is_rotated(ex["input"], ex["output"]))
            flip_count = sum(1 for ex in examples if self._is_flipped(ex["input"], ex["output"]))
            
            if rotation_count == len(examples):
                transformation["type"] = "rotation"
            elif flip_count == len(examples):
                transformation["type"] = "flip"
        
        return transformation
    
    def _apply_learned_transformation(self, grid: List[List[int]], transformation: Dict) -> List[List[int]]:
        """Apply the learned transformation to new input"""
        if transformation.get("type") == "rotation":
            return self._rotate_90(grid)
        elif transformation.get("type") == "flip":
            return self._flip_horizontal(grid)
        
        return [row[:] for row in grid]
    
    def _apply_strategy(self, grid: List[List[int]], examples: List[Dict]) -> List[List[int]]:
        """Try different strategies based on examples"""
        if not examples:
            return [row[:] for row in grid]
        
        target_size = (len(examples[0]["output"]), len(examples[0]["output"][0]))
        if all(len(ex["output"]) == target_size[0] and len(ex["output"][0]) == target_size[1] for ex in examples):
            if target_size[0] < len(grid) or target_size[1] < len(grid[0]):
                return self._crop_to_size(grid, target_size)
            elif target_size[0] > len(grid) or target_size[1] > len(grid[0]):
                return self._expand_to_size(grid, target_size)
        
        for strategy in self.strategies:
            try:
                result = strategy(grid, examples)
                if self._is_valid_output(result):
                    return result
            except Exception:
                continue
        
        return [row[:] for row in grid]
    
    def _is_valid_output(self, grid: List[List[int]]) -> bool:
        """Check if output is valid"""
        if not grid or not grid[0]:
            return False
        
        if len(grid) > 30 or len(grid[0]) > 30:
            return False
        
        for row in grid:
            if len(row) != len(grid[0]):
                return False
            for val in row:
                if not isinstance(val, int) or val < 0 or val > 9:
                    return False
        
        return True
    
    def _get_colors(self, grid: List[List[int]]) -> set:
        """Get all colors in grid"""
        colors = set()
        for row in grid:
            colors.update(row)
        return colors
    
    def _is_rotated(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if grid2 is a rotation of grid1"""
        if len(grid1) == len(grid2[0]) and len(grid1[0]) == len(grid2):
            rotated = self._rotate_90(grid1)
            return rotated == grid2
        return False
    
    def _is_flipped(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if grid2 is a flip of grid1"""
        if len(grid1) == len(grid2) and len(grid1[0]) == len(grid2[0]):
            flipped = self._flip_horizontal(grid1)
            return flipped == grid2
        return False
    
    def _rotate_90(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 90 degrees clockwise"""
        h, w = len(grid), len(grid[0])
        rotated = [[0] * h for _ in range(w)]
        for i in range(h):
            for j in range(w):
                rotated[j][h - 1 - i] = grid[i][j]
        return rotated
    
    def _flip_horizontal(self, grid: List[List[int]]) -> List[List[int]]:
        """Flip grid horizontally"""
        return [row[::-1] for row in grid]
    
    def _crop_to_size(self, grid: List[List[int]], target_size: tuple) -> List[List[int]]:
        """Crop grid to target size"""
        h, w = target_size
        return [row[:w] for row in grid[:h]]
    
    def _expand_to_size(self, grid: List[List[int]], target_size: tuple) -> List[List[int]]:
        """Expand grid to target size by padding with zeros"""
        h, w = target_size
        result = [[0] * w for _ in range(h)]
        for i in range(min(len(grid), h)):
            for j in range(min(len(grid[0]), w)):
                result[i][j] = grid[i][j]
        return result
    
    def _identity_transform(self, grid: List[List[int]], examples: List[Dict] = None) -> List[List[int]]:
        """Return the grid as-is"""
        return [row[:] for row in grid]
    
    def _analyze_color_mapping(self, grid: List[List[int]], examples: List[Dict] = None) -> List[List[int]]:
        """Analyze color changes across all examples and apply"""
        if not examples:
            return grid
        
        color_map = {}
        for ex in examples:
            in_flat = [val for row in ex["input"] for val in row]
            out_flat = [val for row in ex["output"] for val in row]
            
            if len(in_flat) == len(out_flat):
                for i, o in zip(in_flat, out_flat):
                    if i != o:
                        if i in color_map and color_map[i] != o:
                            color_map = {}
                            break
                        color_map[i] = o
        
        if color_map:
            result = []
            for row in grid:
                new_row = [color_map.get(val, val) for val in row]
                result.append(new_row)
            return result
        
        return grid
    
    def _analyze_size_transform(self, grid: List[List[int]], examples: List[Dict] = None) -> List[List[int]]:
        """Analyze size changes across examples"""
        if not examples:
            return grid
        
        all_smaller = all(
            len(ex["output"]) <= len(ex["input"]) and 
            len(ex["output"][0]) <= len(ex["input"][0])
            for ex in examples
        )
        
        if all_smaller and len(grid) > 2:
            result = []
            for i in range(0, len(grid), 2):
                row = []
                for j in range(0, len(grid[0]), 2):
                    row.append(grid[i][j])
                if row:
                    result.append(row)
            if result and result[0]:
                return result
        
        return grid
    
    def _analyze_pattern_transform(self, grid: List[List[int]], examples: List[Dict] = None) -> List[List[int]]:
        """Look for pattern transformations"""
        return self._pattern_complete(grid)
    
    def _analyze_symmetry(self, grid: List[List[int]], examples: List[Dict] = None) -> List[List[int]]:
        """Check for symmetry transformations across examples"""
        if examples:
            flip_count = sum(1 for ex in examples if self._is_flipped(ex["input"], ex["output"]))
            if flip_count == len(examples):
                return self._flip_horizontal(grid)
        return grid
    
    def _pattern_complete(self, grid: List[List[int]]) -> List[List[int]]:
        """Try to complete patterns in the grid"""
        h, w = len(grid), len(grid[0]) if grid else 0
        
        if h < 3 or w < 3:
            return grid
        
        result = [row[:] for row in grid]
        for i in range(h):
            if result[i][0] == result[i][-1] and result[i][0] != 0:
                for j in range(1, w // 2):
                    if result[i][j] == 0:
                        result[i][j] = result[i][w - 1 - j]
                    elif result[i][w - 1 - j] == 0:
                        result[i][w - 1 - j] = result[i][j]
        
        return result


def test_internet_access():
    """Test if internet access is blocked (it should be in inference phase)"""
    print("\n" + "=" * 60)
    print("TESTING INTERNET ACCESS (should be blocked)")
    print("=" * 60)
    
    test_urls = [
        ("google.com", 80),
        ("1.1.1.1", 80),
        ("8.8.8.8", 53),
    ]
    
    all_blocked = True
    for host, port in test_urls:
        try:
            socket.setdefaulttimeout(3)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"  ⚠ WARNING: Successfully connected to {host}:{port} - Internet NOT blocked!")
                all_blocked = False
            else:
                print(f"  ✓ Connection to {host}:{port} blocked (error code: {result})")
        except socket.gaierror:
            print(f"  ✓ DNS resolution for {host} blocked")
        except socket.timeout:
            print(f"  ✓ Connection to {host}:{port} timed out (blocked)")
        except Exception as e:
            print(f"  ✓ Connection to {host}:{port} failed: {e}")
    
    if all_blocked:
        print("\n✓ Internet access is properly blocked")
    else:
        print("\n⚠ WARNING: Internet access is NOT properly blocked!")
    
    print("=" * 60 + "\n")
    
    return all_blocked


def save_output_data(results: Dict, output_dir: Path):
    """Save output data to mounted directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved output to: {output_path}")


def load_input_data(input_dir: Path) -> Dict:
    """Load input data from mounted directory."""
    
    all_files = list(input_dir.glob("*"))
    print("all files : ", all_files)
    # Primary file to look for
    dataset_file = input_dir / "miner_current_dataset.json"
    
    if dataset_file.exists():
        print(f"Found dataset file: {dataset_file}")
        with open(dataset_file, 'r') as f:
            data = json.load(f)

        return data    
        
    raise FileNotFoundError(f"No input data found in {input_dir}")


def run_prep_phase(input_dir: Path, output_dir: Path):
    """Prep phase: Download LLM model and validate input data."""
    print("\n" + "=" * 60)
    print("PREP PHASE - Downloading Model")
    print("=" * 60)
    
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    cache_dir = Path("/app/models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/4] Model to download: {model_name}")
    print(f"[2/4] Downloading model to {cache_dir}...")
    print("(This requires internet access)")
    
    try:
        print("\n[3/4] Downloading model files...")
        
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
        
        # Validate input data
        print("\n[4/4] Validating input data...")
        try:
            problems = load_input_data(input_dir)
            problems = problems['tasks']
            
            print(f"✓ Found {len(problems)} problems")
            
            # Validate first few problems
            for i, problem in enumerate(problems[:3]):
                if "train_examples" not in problem:
                    print(f"    Available keys: {list(problem.keys())}")
                    raise ValueError(f"Problem {i} missing 'train_examples'")
                if "test_input" not in problem:
                    print(f"    Available keys: {list(problem.keys())}")
                    raise ValueError(f"Problem {i} missing 'test_input'")
                if "test_output" in problem:
                    print(f"    Available keys: {list(problem.keys())}")
                    raise ValueError(f"'test_output' is accessible in prep phase - this is dangerous")
            
            print("✓ Input data validation passed")
            
        except Exception as e:
            print(f"WARNING: Input data validation failed: {e}")
            print("Continuing with model download...")
        
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
    
    if prep_results["status"] == "failed":
        sys.exit(1)


def run_inference_phase(input_dir: Path, output_dir: Path):
    """Inference phase: Solve ARC-AGI-2 problems and save predictions."""
    print("\n" + "=" * 60)
    print("INFERENCE PHASE - Solving ARC-AGI-2 Problems")
    print("=" * 60)
    
    # Test internet access (should be blocked)
    internet_blocked = test_internet_access()
    
    try:
        # Load input data
        print(f"\n[1/4] Loading input data from {input_dir}...")
        problems = load_input_data(input_dir)
        problems = problems['tasks']
        
        # Initialize solver with vLLM
        print("[2/4] Initializing ARC solver with vLLM support...")
        solver = ARCSolver(use_vllm=True)
        
        # Solve first 10 problems
        print(f"[3/4] Solving first 10 problems (out of {len(problems)} total)...")
        predictions = []
        
        num_to_solve = min(10, len(problems))
        
        for i in range(num_to_solve):
            problem = problems[i]
            if 'train_examples' not in problem:
                print(f"    ✗ Problem {i} missing 'train_examples' field")
                print(f"    Available keys: {list(problem.keys())}")

            print(f"\n  Problem {i+1}/{num_to_solve}:")
            print(f"    - Training examples: {len(problem['train_examples'])}")
            print(f"    - Test input shape: {len(problem['test_input'])}x{len(problem['test_input'][0])}")
            
            try:
                # Solve the problem
                predicted_output = solver.solve(
                    train_examples=problem['train_examples'],
                    test_input=problem['test_input']
                )
                
                print(f"    - Predicted output shape: {len(predicted_output)}x{len(predicted_output[0])}")
                print(f"    ✓ Solved successfully")
                
                # Store prediction with metadata
                prediction_entry = {
                    "problem_index": i,
                    "task_hash": problem.get("task_hash"),
                    "predicted_output": predicted_output,
                    "metadata": problem.get("metadata", {})
                }

                predictions.append(prediction_entry)
                
            except Exception as e:
                print(f"    ✗ Error solving problem {i}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n[4/4] Saving predictions to {output_dir}...")
        results = {
            "phase": "inference",
            "status": "success",
            "num_problems_attempted": num_to_solve,
            "num_problems_solved": sum(1 for p in predictions if p.get("predicted_output") is not None),
            "internet_blocked": internet_blocked,
            "vllm_available": solver.vllm_available,
            "predictions": predictions
        }
        
        save_output_data(results, output_dir)
        
        print("\n" + "=" * 60)
        print(f"INFERENCE PHASE COMPLETED - Solved {results['num_problems_solved']}/{num_to_solve}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Inference phase failed: {e}")
        import traceback
        traceback.print_exc()
        
        results = {
            "phase": "inference",
            "status": "failed",
            "error": str(e),
            "internet_blocked": internet_blocked,
            "predictions": []
        }
        save_output_data(results, output_dir)
        
        print("\n" + "=" * 60)
        print("INFERENCE PHASE COMPLETED - Status: failed")
        print("=" * 60)
        
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Inference Script")
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