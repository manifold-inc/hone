"""
ARC-AGI-2 Metrics Calculator

Calculates performance metrics for ARC-AGI-2 task predictions
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_grid_similarity(grid1: List[List[int]], grid2: List[List[int]]) -> float:
    """Calculate pixel-wise similarity between two grids"""
    if not grid1 or not grid2:
        return 0.0
    
    # Size mismatch
    if len(grid1) != len(grid2):
        return 0.0
    
    if grid1 and grid2 and len(grid1[0]) != len(grid2[0]):
        return 0.0
    
    total_cells = len(grid1) * len(grid1[0])
    if total_cells == 0:
        return 0.0
    
    matching_cells = sum(
        1 for i in range(len(grid1))
        for j in range(len(grid1[0]))
        if grid1[i][j] == grid2[i][j]
    )
    
    return matching_cells / total_cells


def calculate_partial_correctness(predicted: List[List[int]], expected: List[List[int]]) -> float:
    """
    Calculate partial correctness score considering:
    - Shape matching (30%)
    - Grid similarity (50%)
    - Color distribution (20%)
    """
    if not predicted or not expected:
        return 0.0
    
    score = 0.0
    weights = {'shape': 0.3, 'grid': 0.5, 'colors': 0.2}
    
    # Shape score
    shape_match = (len(predicted) == len(expected) and 
                   len(predicted[0]) == len(expected[0]) if predicted and expected else False)
    score += weights['shape'] if shape_match else 0
    
    # Grid similarity
    if shape_match:
        score += weights['grid'] * calculate_grid_similarity(predicted, expected)
    
    # Color distribution similarity
    pred_colors = set()
    exp_colors = set()
    for row in predicted:
        pred_colors.update(row)
    for row in expected:
        exp_colors.update(row)
    
    if exp_colors:
        color_overlap = len(pred_colors & exp_colors) / len(exp_colors)
        score += weights['colors'] * color_overlap
    
    return min(1.0, score)


def calculate_metrics_for_prediction(
    predicted_output: Optional[List[List[int]]],
    expected_output: List[List[int]],
    metadata: Dict = None
) -> Dict:
    """
    Calculate all metrics for a single prediction
    
    Args:
        predicted_output: Predicted grid (None if prediction failed)
        expected_output: Ground truth grid
        metadata: Optional metadata about the problem
        
    Returns:
        Dictionary with all calculated metrics
    """
    if predicted_output is None:
        return {
            "exact_match": False,
            "partial_correctness": 0.0,
            "grid_similarity": 0.0,
            "shape_match": False,
            "predicted_shape": None,
            "expected_shape": (len(expected_output), len(expected_output[0]) if expected_output else 0)
        }
    
    exact_match = predicted_output == expected_output
    partial_correctness = calculate_partial_correctness(predicted_output, expected_output)
    grid_similarity = calculate_grid_similarity(predicted_output, expected_output)
    
    shape_match = (
        len(predicted_output) == len(expected_output) and
        len(predicted_output[0]) == len(expected_output[0]) if predicted_output and expected_output else False
    )
    
    return {
        "exact_match": exact_match,
        "partial_correctness": partial_correctness,
        "grid_similarity": grid_similarity,
        "shape_match": shape_match,
        "predicted_shape": (len(predicted_output), len(predicted_output[0]) if predicted_output else 0),
        "expected_shape": (len(expected_output), len(expected_output[0]) if expected_output else 0)
    }


def calculate_aggregate_metrics(predictions: List[Dict]) -> Dict:
    """
    Calculate aggregate metrics across all predictions
    
    Args:
        predictions: List of prediction dictionaries with metrics
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not predictions:
        return {
            "total_problems": 0,
            "num_solved": 0,
            "num_exact_matches": 0,
            "avg_partial_correctness": 0.0,
            "avg_grid_similarity": 0.0,
            "shape_match_rate": 0.0,
            "success_rate": 0.0,
            "exact_match_rate": 0.0
        }
    
    total_problems = len(predictions)
    num_solved = sum(1 for p in predictions if p.get("predicted_output") is not None)
    
    # Calculate metrics only for solved problems
    solved_predictions = [p for p in predictions if p.get("predicted_output") is not None]
    
    if not solved_predictions:
        return {
            "total_problems": total_problems,
            "num_solved": 0,
            "num_exact_matches": 0,
            "avg_partial_correctness": 0.0,
            "avg_grid_similarity": 0.0,
            "shape_match_rate": 0.0,
            "success_rate": 0.0,
            "exact_match_rate": 0.0
        }
    
    # Calculate individual metrics
    metrics_list = []
    for pred in solved_predictions:
        metrics = calculate_metrics_for_prediction(
            pred.get("predicted_output"),
            pred.get("test_output"),
            pred.get("metadata", {})
        )
        metrics_list.append(metrics)
    
    num_exact_matches = sum(1 for m in metrics_list if m["exact_match"])
    num_shape_matches = sum(1 for m in metrics_list if m["shape_match"])
    
    avg_partial_correctness = sum(m["partial_correctness"] for m in metrics_list) / len(metrics_list)
    avg_grid_similarity = sum(m["grid_similarity"] for m in metrics_list) / len(metrics_list)
    
    return {
        "total_problems": total_problems,
        "num_solved": num_solved,
        "num_exact_matches": num_exact_matches,
        "num_shape_matches": num_shape_matches,
        "avg_partial_correctness": round(avg_partial_correctness, 4),
        "avg_grid_similarity": round(avg_grid_similarity, 4),
        "shape_match_rate": round(num_shape_matches / len(metrics_list), 4) if metrics_list else 0.0,
        "success_rate": round(num_solved / total_problems, 4),
        "exact_match_rate": round(num_exact_matches / len(metrics_list), 4) if metrics_list else 0.0
    }


def calculate_detailed_metrics(results_data: Dict) -> Dict:
    """
    Calculate detailed metrics from inference results
    
    Args:
        results_data: Results dictionary from inference.py output
        
    Returns:
        Dictionary with detailed metrics breakdown
    """
    predictions = results_data.get("predictions", [])
    
    # Calculate aggregate metrics
    aggregate = calculate_aggregate_metrics(predictions)
    
    # Calculate per-problem metrics
    per_problem_metrics = []
    for pred in predictions:
        problem_metrics = calculate_metrics_for_prediction(
            pred.get("predicted_output"),
            pred.get("test_output"),
            pred.get("metadata", {})
        )
        problem_metrics["problem_index"] = pred.get("problem_index")
        problem_metrics["has_prediction"] = pred.get("predicted_output") is not None
        problem_metrics["metadata"] = pred.get("metadata", {})
        per_problem_metrics.append(problem_metrics)
    
    return {
        "aggregate": aggregate,
        "per_problem": per_problem_metrics
    }