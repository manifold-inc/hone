from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import random

import validator.synthetics.arcgen.arc_agi2_utils as utils
import validator.synthetics.arcgen.task_list as task_list


def _count_non_black(grid: List[List[int]]) -> int:
    return sum(1 for row in grid for v in row if v != 0)


class ARC2Generator:
    """
    Generate ARC-AGI-2 style problems by applying transformation chains to ARC-1 base tasks.
    
    Key approach:
      - Start from a base ARC-1 task (input -> output)
      - Apply a parameterized chain of transforms to the *output*
      - The same chain (with frozen parameters) is reused for all examples
    
    This creates problems requiring compositional reasoning while remaining
    solvable by humans who can infer the transformation pattern.
    """

    def __init__(
        self,
        max_chain_length: int = 4,
        max_grid_size: int = 30,
        seed: Optional[int] = None,
    ):
        self.max_chain_length = max_chain_length
        self.max_grid_size = max_grid_size
        self.rng = random.Random(seed)

        # Quality thresholds to keep outputs meaningful
        self.min_distinct_colors = 2
        self.min_non_black_cells = 6
        self.max_resample_attempts = 4

        # Cache which transforms preserve grid dimensions
        self._preserves_size = {
            name: meta.get("preserves_size", False)
            for name, (_, meta) in utils.TRANSFORMATIONS.items()
        }

    def generate_initial_problem(
        self,
        task_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a base ARC-1 style problem from task_list.
        
        Returns:
            Dict with "input", "output", and "task_num" keys
        """
        tmap = task_list.task_list()
        if task_num is None:
            task_num = self.rng.choice(list(tmap.keys()))

        _, gen_fn, _ = tmap[task_num]
        pair = gen_fn()
        
        if isinstance(pair, dict):
            inp = pair["input"]
            out = pair["output"]
        else:
            inp, out = pair

        if not utils.is_valid_grid(inp) or not utils.is_valid_grid(out):
            raise ValueError(f"Base task produced invalid grid(s) - task_num: {task_num}")

        h, w = utils.get_grid_size(out)
        if h > self.max_grid_size or w > self.max_grid_size:
            raise ValueError(f"Base output too large: {h}x{w} > {self.max_grid_size} - task_num: {task_num}")

        return {"input": inp, "output": out, "task_num": task_num}

    def _sample_params(self, name: str, grid: List[List[int]]) -> Optional[Dict[str, Any]]:
        """
        Sample parameters for a transformation based on current grid state.
        Returns None if the transform doesn't need parameters or can't be applied.
        """
        colors_present = list(utils.get_colors_in_grid(grid) - {0})
        palette = list(range(1, 10))  # Colors 1-9

        if name == "swap_colors":
            if len(colors_present) >= 2:
                c1, c2 = self.rng.sample(colors_present, 2)
            elif len(colors_present) == 1:
                c1 = colors_present[0]
                c2 = self.rng.choice([c for c in palette if c != c1])
            else:
                c1, c2 = 1, 2
            return {"color1": c1, "color2": c2}

        if name == "remove_color":
            if len(colors_present) <= 1:
                return None
            return {"color": self.rng.choice(colors_present)}

        if name == "highlight_color":
            if not colors_present:
                return None
            return {"color": self.rng.choice(colors_present)}

        if name == "shift":
            direction = self.rng.choice(["up", "down", "left", "right"])
            h, w = utils.get_grid_size(grid)
            span = h if direction in ("up", "down") else w
            max_amt = max(1, span - 1)
            amt = self.rng.randint(1, min(3, max_amt))
            return {"direction": direction, "amount": amt, "wrap": False}

        return None

    def select_transformation_chain(
        self,
        grid: List[List[int]],
        chain_length: Optional[int] = None,
        preserves_size_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Build a parameterized transformation chain.
        
        Each transform is applied to the current grid state to ensure compatibility,
        and parameters are frozen for reuse across examples.
        """
        if chain_length is None:
            chain_length = self.max_chain_length

        pool = list(utils.TRANSFORMATIONS.keys())
        result_chain: List[Dict[str, Any]] = []
        cur = utils.deep_copy_grid(grid)

        for _ in range(chain_length):
            compatible = utils.get_compatible_transformations(cur, max_size=self.max_grid_size)
            available = [t for t in pool if t in compatible]

            if preserves_size_only:
                available = [t for t in available if self._preserves_size.get(t, False)]

            if not available:
                break

            # Avoid immediate reversals for better chain quality
            if result_chain and len(available) > 1:
                last = result_chain[-1]["name"]
                avoid = {
                    "flip_horizontal": {"flip_horizontal"},
                    "flip_vertical": {"flip_vertical"},
                    "rotate_90": {"rotate_270"},
                    "rotate_270": {"rotate_90"},
                    "rotate_180": {"rotate_180"},
                    "gravity_down": {"gravity_up"},
                    "gravity_up": {"gravity_down"},
                    "gravity_left": {"gravity_right"},
                    "gravity_right": {"gravity_left"},
                }.get(last, set())
                filtered = [t for t in available if t not in avoid]
                if filtered:
                    available = filtered

            name = self.rng.choice(available)
            params = self._sample_params(name, cur)
            
            # Skip if params are invalid
            if name in ("remove_color", "highlight_color") and params is None:
                continue

            new_cur = utils.apply_transformation(cur, name, params)
            if not utils.is_valid_grid(new_cur):
                continue

            result_chain.append({"name": name, "params": params})
            cur = new_cur

        return result_chain

    def apply_transformation_chain(
        self,
        grid: List[List[int]],
        chain: List[Dict[str, Any]],
    ) -> List[List[int]]:
        """Apply a frozen transformation chain to a grid."""
        out = utils.deep_copy_grid(grid)
        for step in chain:
            name = step["name"]
            params = step.get("params")
            out = utils.apply_transformation(out, name, params)
            if not utils.is_valid_grid(out):
                raise ValueError(f"Invalid grid after transform: {name}")
        return out

    def _non_degenerate(self, grid: List[List[int]]) -> bool:
        """Check if grid has sufficient complexity."""
        distinct_colors = utils.get_colors_in_grid(grid) - {0}
        if len(distinct_colors) < self.min_distinct_colors:
            return False
        if _count_non_black(grid) < self.min_non_black_cells:
            return False
        return True

    def generate_problem(
        self,
        task_num: Optional[int] = None,
        chain_length: Optional[int] = None,
        return_metadata: bool = True,
        preserves_size_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a single ARC-AGI-2 style problem.

        Returns:
            {
              "input": <grid>,
              "output": <grid after applying chain>,
              "metadata": {
                  "base_task": <int>,
                  "transformation_chain": [{"name": str, "params": dict|None}, ...],
                  "chain_length": int,
                  "initial_output": <grid>  # before chain
              }
            }
        """
        base = self.generate_initial_problem(task_num)
        attempts = 0

        while True:
            chain = self.select_transformation_chain(
                base["output"],
                chain_length=chain_length,
                preserves_size_only=preserves_size_only,
            )
            transformed = self.apply_transformation_chain(base["output"], chain)

            if self._non_degenerate(transformed):
                break

            attempts += 1
            if attempts >= self.max_resample_attempts:
                break

        result = {
            "input": base["input"],
            "output": transformed,
        }
        
        if return_metadata:
            result["metadata"] = {
                "base_task": base["task_num"],
                "transformation_chain": chain,
                "chain_length": len(chain),
                "initial_output": base["output"],
            }
        
        return result

    def generate_problem_set(
        self,
        num_train: int = 3,
        num_test: int = 1,
        task_num: Optional[int] = None,
        chain_length: Optional[int] = None,
        preserves_size_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate train/test examples with the same transformation chain.
        
        This mirrors the ARC-AGI format where the same underlying rule
        applies across all examples.
        
        Returns:
            {
              "train_examples": [{"input": <grid>, "output": <grid>}, ...],
              "test_input": <grid>,
              "test_output": <grid>,  # For validation
              "metadata": {
                  "base_task": <int>,
                  "transformation_chain": [...],
                  "chain_length": int
              }
            }
        """
        # Generate initial task to get base task number
        base_initial = self.generate_initial_problem(task_num)
        task_num = base_initial["task_num"]
        
        # Generate transformation chain (empty if chain_length is 0)
        if chain_length == 0:
            chain = []
        else:
            chain = self.select_transformation_chain(
                base_initial["output"],
                chain_length=chain_length,
                preserves_size_only=preserves_size_only,
            )
        
        # Generate training examples
        train_examples = []
        attempts = 0
        max_attempts = num_train * 5
        
        while len(train_examples) < num_train and attempts < max_attempts:
            attempts += 1
            try:
                base = self.generate_initial_problem(task_num=task_num)
                
                if chain:
                    output = self.apply_transformation_chain(base["output"], chain)
                else:
                    output = base["output"]
                
                if self._non_degenerate(output):
                    train_examples.append({
                        "input": base["input"],
                        "output": output
                    })
            except:
                continue
        
        # Generate test example
        test_base = self.generate_initial_problem(task_num=task_num)
        if chain:
            test_output = self.apply_transformation_chain(test_base["output"], chain)
        else:
            test_output = test_base["output"]
        
        return {
            "train_examples": train_examples,
            "test_input": test_base["input"],
            "test_output": test_output,
            "metadata": {
                "base_task": task_num,
                "transformation_chain": chain,
                "chain_length": len(chain)
            }
        }