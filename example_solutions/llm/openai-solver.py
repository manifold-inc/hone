from openai import OpenAI
import os
from typing import List, Optional
from loguru import logger
import json

class ARCSolver:
    """
    Pure LLM-based ARC solver using o3-style prompting from https://github.com/arcprize/arc-agi-benchmarking/blob/main/prompt_example_o3.md
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        
        self.model = os.getenv("OPENAI_MODEL", "o3")
    
    def solve(self, input_grid: List[List[int]], difficulty: str = "medium") -> List[List[int]]:
        """
        Solve ARC problem using pure LLM approach
        """
        try:
            result = self._llm_solve_direct(input_grid, difficulty)
            
            if result and self._is_valid_output(result):
                return result
            
            logger.warning("LLM failed to produce valid output, returning input")
            return input_grid
            
        except Exception as e:
            logger.error(f"Solver error: {e}")
            return input_grid
    
    def _llm_solve_direct(self, input_grid: List[List[int]], difficulty: str) -> Optional[List[List[int]]]:
        """Direct LLM solving with o3-inspired prompt"""
        
        grid_str = self._grid_to_text(input_grid)
        
        system_prompt = """You are an expert at solving ARC (Abstraction and Reasoning Corpus) problems.

ARC problems involve finding patterns and transformations in colored grids.
Colors are represented as integers: 0=Black/Background, 1=Blue, 2=Red, 3=Green, 4=Yellow, 5=Grey, 6=Magenta, 7=Orange, 8=Light Blue, 9=Brown

Your task: Analyze the input grid and determine the most likely transformation to produce the output grid.

Common ARC transformations include:
- Geometric transformations (rotation, reflection, scaling)
- Color manipulations (swapping, filling, filtering)
- Pattern completion and extension
- Object detection and manipulation
- Symmetry operations
- Rule-based transformations

You must respond with a valid JSON object containing only:
{"reasoning": "brief explanation of the transformation", "output": [[the transformed grid as a 2D array]]}

The output grid must be valid - all rows same length, values 0-9 only."""

        if difficulty == "easy":
            hint = """This is an EASY problem. Common easy transformations:
- Simple rotations (90, 180, 270 degrees)
- Horizontal or vertical flips
- Color swaps (exchange two colors)
- Direct pattern copying
- Simple filling operations"""
        elif difficulty == "medium":
            hint = """This is a MEDIUM problem. Common medium transformations:
- Pattern extension or completion
- Object movement or duplication
- Conditional color changes
- Symmetry completion
- Size changes with rules"""
        else:  # hard
            hint = """This is a HARD problem. Common hard transformations (it can also include chains of transformations):
- Complex multi-step operations
- Object interaction rules
- Abstract pattern recognition
- Counting or grouping operations
- Conditional transformations based on neighbors
- Composite transformations"""

        prompt = f"""{hint}

Input grid:
{grid_str}

Grid dimensions: {len(input_grid)} rows × {len(input_grid[0])} columns

Analyze this grid carefully. Consider:
1. What colors are present? What patterns do they form?
2. Are there any objects, shapes, or structures?
3. Is there symmetry or repetition?
4. What transformation would make sense for this pattern?

Apply the most likely transformation and return the output grid."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            reasoning = result.get("reasoning", "No reasoning provided")
            logger.info(f"LLM reasoning: {reasoning}")
            
            output = result.get("output", [])
            
            if output and isinstance(output, list):
                cleaned = self._clean_output(output)
                if self._is_valid_output(cleaned):
                    return cleaned
                else:
                    logger.error("LLM output failed validation")
            else:
                logger.error("LLM did not return a valid output grid")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            logger.error(f"LLM API error: {e}")
        
        return None
    
    def _grid_to_text(self, grid: List[List[int]]) -> str:
        """Convert grid to clear text representation"""
        lines = []
        for row in grid:
            lines.append(" ".join(str(cell) for cell in row))
        return "\n".join(lines)
    
    def _clean_output(self, output: List[List]) -> List[List[int]]:
        """Clean and validate output from LLM"""
        cleaned = []
        
        for row in output:
            if not isinstance(row, list):
                logger.error(f"Invalid row type: {type(row)}")
                return []
            
            cleaned_row = []
            for cell in row:
                try:
                    val = int(cell)
                    val = max(0, min(9, val))
                    cleaned_row.append(val)
                except (ValueError, TypeError):
                    logger.error(f"Invalid cell value: {cell}")
                    cleaned_row.append(0)
            
            cleaned.append(cleaned_row)
        
        # ensure all rows have same length
        if cleaned:
            row_lengths = [len(row) for row in cleaned]
            if len(set(row_lengths)) > 1:
                logger.warning(f"Inconsistent row lengths: {row_lengths}")
                # pad or truncate to most common length
                target_len = max(set(row_lengths), key=row_lengths.count)
                for i, row in enumerate(cleaned):
                    if len(row) < target_len:
                        cleaned[i] = row + [0] * (target_len - len(row))
                    elif len(row) > target_len:
                        cleaned[i] = row[:target_len]
        
        return cleaned
    
    def _is_valid_output(self, grid: List[List[int]]) -> bool:
        """Check if output grid is valid"""
        if not grid or not isinstance(grid, list):
            return False
        
        if len(grid) == 0 or not grid[0]:
            return False
        
        # size limits
        if len(grid) > 30 or len(grid[0]) > 30:
            logger.error(f"Grid too large: {len(grid)}×{len(grid[0])}")
            return False
        
        # check all rows have same length
        first_row_len = len(grid[0])
        for i, row in enumerate(grid):
            if not isinstance(row, list):
                logger.error(f"Row {i} is not a list")
                return False
            if len(row) != first_row_len:
                logger.error(f"Row {i} has wrong length: {len(row)} vs {first_row_len}")
                return False
            
            # check all values are valid
            for j, val in enumerate(row):
                if not isinstance(val, int):
                    logger.error(f"Cell [{i},{j}] is not an int: {type(val)}")
                    return False
                if val < 0 or val > 9:
                    logger.error(f"Cell [{i},{j}] out of range: {val}")
                    return False
        
        return True