#!/usr/bin/env python3
"""
ARC-AGI-2 Solver for Sandbox Runner
Loads problems, generates predictions, and saves results
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional

print("Starting ARC-AGI-2 inference script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Arguments: {sys.argv}")


class ARCSolver:
    """
    ARC solver with rule-based strategies
    (OpenAI integration removed for sandbox environment)
    """
    
    def __init__(self):
        print("🔧 ARCSolver initialized with rule-based strategies")
        
        # Rule-based strategies
        self.strategies = [
            self._identity_transform,
            self._analyze_color_mapping,
            self._analyze_size_transform,
            self._analyze_pattern_transform,
            self._analyze_symmetry
        ]
    
    def solve(self, train_examples: List[Dict], test_input: List[List[int]]) -> List[List[int]]:
        """
        Learn from training examples and apply to test input
        
        Args:
            train_examples: List of dicts with 'input' and 'output' grids
            test_input: The test input grid to solve
        """
        if not train_examples:
            # No examples, return input as-is
            return [row[:] for row in test_input]
        
        # Identify transformation from training examples
        transformation = self._identify_transformation(train_examples)
        
        if transformation and transformation.get("type"):
            return self._apply_learned_transformation(test_input, transformation)
        
        return self._apply_strategy(test_input, train_examples)
    
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
        
        # Try basic strategies
        for strategy in self.strategies:
            try:
                result = strategy(grid, examples)
                if self._is_valid_output(result):
                    return result
            except Exception:
                continue
        
        # Last resort: return input
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


def save_output_data(results: Dict, output_dir: Path):
    """Save output data to mounted directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved output to: {output_path}")


def load_input_data(input_dir: Path) -> Dict:
    """Load input data from mounted directory."""
    
    # Try to find the input file
    potential_files = [
        input_dir / "input_data",
        input_dir / "input.json",
        input_dir / "data.json",
        input_dir / "checkpoint.json",
    ]
    
    # Also check for any .json file in the directory
    if input_dir.exists():
        json_files = list(input_dir.glob("*.json"))
        potential_files.extend(json_files)
    
    for file_path in potential_files:
        if file_path.exists():
            print(f"Found input file: {file_path}")
            with open(file_path, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"No input data found in {input_dir}")


def run_prep_phase(input_dir: Path, output_dir: Path):
    """Prep phase: Validate input data and prepare for inference."""
    print("\n" + "=" * 60)
    print("PREP PHASE - Validating Input Data")
    print("=" * 60)
    
    try:
        # Load input data
        print(f"\n[1/3] Loading input data from {input_dir}...")
        data = load_input_data(input_dir)
        
        # Validate structure
        print("[2/3] Validating data structure...")
        if "problems" not in data:
            raise ValueError("Input data must contain 'problems' field")
        
        problems = data["problems"]
        print(f"✓ Found {len(problems)} problems")
        
        # Validate first few problems
        print("[3/3] Validating problem format...")
        for i, problem in enumerate(problems[:3]):
            if "train_examples" not in problem:
                raise ValueError(f"Problem {i} missing 'train_examples'")
            if "test_input" not in problem:
                raise ValueError(f"Problem {i} missing 'test_input'")
            if "test_output" not in problem:
                raise ValueError(f"Problem {i} missing 'test_output'")
        
        print("✓ All validations passed")
        
        # Save prep results
        prep_results = {
            "phase": "prep",
            "status": "success",
            "message": f"Validated {len(problems)} problems",
            "num_problems": len(problems)
        }
        
        save_output_data(prep_results, output_dir)
        
        print("\n" + "=" * 60)
        print("PREP PHASE COMPLETED - Status: success")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR: Prep phase failed: {e}")
        import traceback
        traceback.print_exc()
        
        prep_results = {
            "phase": "prep",
            "status": "failed",
            "message": str(e)
        }
        save_output_data(prep_results, output_dir)
        
        print("\n" + "=" * 60)
        print("PREP PHASE COMPLETED - Status: failed")
        print("=" * 60)
        
        sys.exit(1)


def run_inference_phase(input_dir: Path, output_dir: Path):
    """Inference phase: Solve ARC-AGI-2 problems and save predictions."""
    print("\n" + "=" * 60)
    print("INFERENCE PHASE - Solving ARC-AGI-2 Problems")
    print("=" * 60)
    
    try:
        # Load input data
        print(f"\n[1/4] Loading input data from {input_dir}...")
        data = load_input_data(input_dir)
        problems = data["problems"]
        
        # Initialize solver
        print("[2/4] Initializing ARC solver...")
        solver = ARCSolver()
        
        # Solve first 10 problems
        print(f"[3/4] Solving first 10 problems (out of {len(problems)} total)...")
        predictions = []
        
        num_to_solve = min(10, len(problems))
        
        for i in range(num_to_solve):
            problem = problems[i]
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
                    "predicted_output": predicted_output,
                    "test_output": problem.get("test_output"),  # Include expected output for metrics
                    "metadata": problem.get("metadata", {})
                }
                predictions.append(prediction_entry)
                
            except Exception as e:
                print(f"    ✗ Error solving problem {i}: {e}")
                # Store failed prediction
                predictions.append({
                    "problem_index": i,
                    "predicted_output": None,
                    "test_output": problem.get("test_output"),
                    "error": str(e),
                    "metadata": problem.get("metadata", {})
                })
        
        # Save results
        print(f"\n[4/4] Saving predictions to {output_dir}...")
        results = {
            "phase": "inference",
            "status": "success",
            "num_problems_attempted": num_to_solve,
            "num_problems_solved": sum(1 for p in predictions if p.get("predicted_output") is not None),
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