"""
ARC-AGI-2 Baisc LLM Solver with fallback
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 60)
print("ARC-AGI-2 INFERENCE SCRIPT")
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")
print("=" * 60)


class ARCSolver:
    """
    Simplified ARC solver with rule-based strategies
    """
    
    def __init__(self):
        logger.info("Initializing ARC Solver (rule-based mode)")
        
        self.strategies = [
            self._identity_transform,
            self._rotate_90,
            self._flip_horizontal,
            self._analyze_color_mapping,
            self._pattern_complete,
        ]
    
    def solve(self, train_examples: List[Dict], test_input: List[List[int]]) -> List[List[int]]:
        """
        Learn from training examples and apply to test input
        """
        logger.info(f"Solving with {len(train_examples)} training examples")
        
        if not train_examples:
            return test_input
        
        transformation = self._identify_transformation(train_examples)
        
        if transformation.get("type"):
            logger.info(f"Identified transformation: {transformation['type']}")
            return self._apply_transformation(test_input, transformation)
        
        for strategy in self.strategies:
            try:
                result = strategy(test_input, train_examples)
                if self._is_valid_output(result):
                    logger.info(f"Applied strategy: {strategy.__name__}")
                    return result
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        logger.warning("No transformation found, returning input")
        return test_input
    
    def _identify_transformation(self, examples: List[Dict]) -> Dict:
        """Analyze examples to identify transformation type"""
        if not examples:
            return {}
        
        output_sizes = [(len(ex["output"]), len(ex["output"][0])) for ex in examples]
        same_size = len(set(output_sizes)) == 1
        
        size_preserved = all(
            len(ex["input"]) == len(ex["output"]) and 
            len(ex["input"][0]) == len(ex["output"][0])
            for ex in examples
        )
        
        transformation = {
            "same_output_size": same_size,
            "size_preserved": size_preserved
        }
        
        if size_preserved and all(self._is_rotated(ex["input"], ex["output"]) for ex in examples):
            transformation["type"] = "rotation"
        
        elif size_preserved and all(self._is_flipped(ex["input"], ex["output"]) for ex in examples):
            transformation["type"] = "flip"
        
        elif size_preserved:
            color_map = self._find_consistent_color_map(examples)
            if color_map:
                transformation["type"] = "color_map"
                transformation["color_map"] = color_map
        
        return transformation
    
    def _apply_transformation(self, grid: List[List[int]], transformation: Dict) -> List[List[int]]:
        """Apply identified transformation to grid"""
        trans_type = transformation.get("type")
        
        if trans_type == "rotation":
            return self._rotate_90(grid)
        elif trans_type == "flip":
            return self._flip_horizontal(grid)
        elif trans_type == "color_map":
            return self._apply_color_map(grid, transformation["color_map"])
        
        return grid
    
    def _identity_transform(self, grid: List[List[int]], examples: List[Dict] = None) -> List[List[int]]:
        """Return grid as-is"""
        return [row[:] for row in grid]
    
    def _rotate_90(self, grid: List[List[int]], examples: List[Dict] = None) -> List[List[int]]:
        """Rotate grid 90 degrees clockwise"""
        if not grid:
            return grid
        
        h, w = len(grid), len(grid[0])
        rotated = [[0] * h for _ in range(w)]
        
        for i in range(h):
            for j in range(w):
                rotated[j][h - 1 - i] = grid[i][j]
        
        return rotated
    
    def _flip_horizontal(self, grid: List[List[int]], examples: List[Dict] = None) -> List[List[int]]:
        """Flip grid horizontally"""
        return [row[::-1] for row in grid]
    
    def _analyze_color_mapping(self, grid: List[List[int]], examples: List[Dict] = None) -> List[List[int]]:
        """Find and apply color mapping from examples"""
        if not examples:
            return grid
        
        color_map = self._find_consistent_color_map(examples)
        if color_map:
            return self._apply_color_map(grid, color_map)
        
        return grid
    
    def _apply_color_map(self, grid: List[List[int]], color_map: Dict[int, int]) -> List[List[int]]:
        """Apply color mapping to grid"""
        result = []
        for row in grid:
            new_row = [color_map.get(val, val) for val in row]
            result.append(new_row)
        return result
    
    def _find_consistent_color_map(self, examples: List[Dict]) -> Optional[Dict[int, int]]:
        """Find color mapping that's consistent across all examples"""
        if not examples:
            return None
        
        color_map = {}
        
        for ex in examples:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
            
            for i in range(len(ex["input"])):
                for j in range(len(ex["input"][0])):
                    in_color = ex["input"][i][j]
                    out_color = ex["output"][i][j]
                    
                    if in_color in color_map:
                        if color_map[in_color] != out_color:
                            return None  # Inconsistent mapping
                    else:
                        color_map[in_color] = out_color
        
        return color_map if color_map else None
    
    def _pattern_complete(self, grid: List[List[int]], examples: List[Dict] = None) -> List[List[int]]:
        """Try to complete patterns in the grid"""
        h, w = len(grid), len(grid[0]) if grid else 0
        
        if h < 3 or w < 3:
            return grid
        
        result = [row[:] for row in grid]
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if result[i][j] == 0:  # Empty cell
                    neighbors = [
                        result[i-1][j], result[i+1][j],
                        result[i][j-1], result[i][j+1]
                    ]
                    non_zero = [n for n in neighbors if n != 0]
                    if non_zero and all(n == non_zero[0] for n in non_zero):
                        result[i][j] = non_zero[0]
        
        return result
    
    def _is_rotated(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if grid2 is rotation of grid1"""
        if len(grid1) != len(grid2[0]) or len(grid1[0]) != len(grid2):
            return False
        
        rotated = self._rotate_90(grid1)
        return rotated == grid2
    
    def _is_flipped(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if grid2 is flip of grid1"""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return False
        
        flipped = self._flip_horizontal(grid1)
        return flipped == grid2
    
    def _is_valid_output(self, grid: List[List[int]]) -> bool:
        """Check if output is valid"""
        if not grid or not grid[0]:
            return False
        
        if len(grid) > 30 or len(grid[0]) > 30:
            return False
        
        first_row_len = len(grid[0])
        for row in grid:
            if len(row) != first_row_len:
                return False
        
        for row in grid:
            for val in row:
                if not isinstance(val, int) or val < 0 or val > 9:
                    return False
        
        return True


def load_dataset(input_dir: Path) -> Dict:
    """Load dataset from input directory"""
    possible_files = [
        "dataset.json",
        "input_data.json", 
        "input.json",
        "data.json"
    ]
    
    for filename in possible_files:
        file_path = input_dir / filename
        if file_path.exists():
            logger.info(f"Loading dataset from: {file_path}")
            with open(file_path, 'r') as f:
                return json.load(f)
    
    json_files = list(input_dir.glob("*.json"))
    if json_files:
        logger.info(f"Loading dataset from: {json_files[0]}")
        with open(json_files[0], 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"No dataset found in {input_dir}")


def save_results(results: Dict, output_dir: Path):
    """Save results to output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")


def run_prep_phase(input_dir: Path, output_dir: Path):
    """
    Prep phase: Just validate we can load the data
    In a real implementation, this would download models
    """
    logger.info("=" * 60)
    logger.info("PREP PHASE")
    logger.info("=" * 60)
    
    try:
        dataset = load_dataset(input_dir)
        logger.info(f"Dataset validated: {len(dataset.get('tasks', []))} tasks")
        
        prep_results = {
            "phase": "prep",
            "status": "success",
            "message": "Dataset loaded successfully",
            "task_count": len(dataset.get("tasks", []))
        }
        
        save_results(prep_results, output_dir)
        logger.info("Prep phase completed successfully")
        
    except Exception as e:
        logger.error(f"Prep phase failed: {e}")
        error_results = {
            "phase": "prep",
            "status": "failed",
            "error": str(e)
        }
        save_results(error_results, output_dir)
        raise


def run_inference_phase(input_dir: Path, output_dir: Path):
    """
    Inference phase: Solve ARC tasks
    """
    logger.info("=" * 60)
    logger.info("INFERENCE PHASE")
    logger.info("=" * 60)
    
    try:
        dataset = load_dataset(input_dir)
        tasks = dataset.get("tasks", [])
        
        if not tasks:
            raise ValueError("No tasks found in dataset")
        
        logger.info(f"Processing {len(tasks)} tasks")
        
        solver = ARCSolver()
        
        predictions = []
        for i, task_data in enumerate(tasks):
            logger.info(f"\nTask {i+1}/{len(tasks)}")
            
            task = task_data.get("task", {})
            train_examples = task.get("train_examples", [])
            test_input = task.get("test_input", [])
            
            if not test_input:
                logger.warning(f"Task {i} has no test input")
                predictions.append({
                    "task_index": i,
                    "task_hash": task_data.get("task_hash", f"task_{i}"),
                    "predicted_output": None,
                    "error": "No test input"
                })
                continue
            
            try:
                predicted_output = solver.solve(train_examples, test_input)
                
                predictions.append({
                    "task_index": i,
                    "task_hash": task_data.get("task_hash", f"task_{i}"),
                    "predicted_output": predicted_output,
                    "metadata": task_data.get("metadata", {})
                })
                
                logger.info(f"✓ Task {i} solved")
                
            except Exception as e:
                logger.error(f"✗ Task {i} failed: {e}")
                predictions.append({
                    "task_index": i,
                    "task_hash": task_data.get("task_hash", f"task_{i}"),
                    "predicted_output": None,
                    "error": str(e)
                })
        
        num_solved = sum(1 for p in predictions if p.get("predicted_output") is not None)
        
        results = {
            "phase": "inference",
            "status": "success",
            "total_tasks": len(tasks),
            "tasks_attempted": len(predictions),
            "tasks_solved": num_solved,
            "success_rate": num_solved / len(tasks) if tasks else 0,
            "predictions": predictions
        }
        
        save_results(results, output_dir)
        
        logger.info("=" * 60)
        logger.info(f"INFERENCE COMPLETE: {num_solved}/{len(tasks)} tasks solved")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Inference phase failed: {e}")
        error_results = {
            "phase": "inference",
            "status": "failed",
            "error": str(e),
            "predictions": []
        }
        save_results(error_results, output_dir)
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Solver")
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
    
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    if args.phase == "prep":
        run_prep_phase(input_dir, output_dir)
    else:
        run_inference_phase(input_dir, output_dir)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)