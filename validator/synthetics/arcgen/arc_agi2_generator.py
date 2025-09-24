from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import random

import validator.synthetics.arcgen.arc_agi2_utils as utils
import validator.synthetics.arcgen.task_list as task_list


def _count_non_black(grid: List[List[int]]) -> int:
    return sum(1 for row in grid for v in row if v != 0)


class ARC2Generator:
    """
    Generate problems of the form:
      - Start from a base ARC-1-like task (input -> output).
      - Apply a parameterized chain of post-transforms to the *output*.
      - The same chain (incl. parameters) is reused for all examples.

    Key improvements:
      - Every transform has frozen parameters recorded in metadata.
      - Optional `preserves_size_only` filter if you need equal HxW across examples.
      - Light degeneracy checks to avoid all-black or trivial outputs.
      - Dedicated RNG to avoid global random state surprises.
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

        self.easy_transforms = [
            "flip_horizontal", "flip_vertical",
            "rotate_90", "rotate_180", "rotate_270",
        ]
        self.medium_transforms = [
            "transpose", "shift", "gravity_down", "gravity_left", "gravity_right",
            "swap_colors", "add_frame", "crop", "recenter", "highlight_color",
        ]
        self.hard_transforms = [
            "zoom_2x", "invert_colors", "remove_color",
            "downsample_2x", "add_noise"
        ]

        # safety net to keep outputs meaningful
        self.min_distinct_colors = 2
        self.min_non_black_cells = 6
        self.max_resample_attempts = 4

        self._preserves_size = {
            name: meta.get("preserves_size", False)
            for name, (_, meta) in utils.TRANSFORMATIONS.items()
        }

    # --------------------
    # Base ARC-1 problem
    # --------------------
    def generate_initial_problem(
        self,
        task_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Pull one base ARC-1 style instance from task_list
        Assumes task_list maps idx -> [task_id_str, generate, validate]
        Expects generate() to yield an (input_grid, output_grid) pair or a dict
        with "input"/"output" keys
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

        # basic validation
        if not utils.is_valid_grid(inp) or not utils.is_valid_grid(out):
            raise ValueError("Base task produced invalid grid(s).")

        # keep sizes within bounds
        h, w = utils.get_grid_size(out)
        if h > self.max_grid_size or w > self.max_grid_size:
            raise ValueError(f"Base output too large: {h}x{w} > {self.max_grid_size}")

        return {"input": inp, "output": out, "task_num": task_num}

    # --------------------------------
    # Parameter sampling per transform
    # --------------------------------
    def _sample_params(self, name: str, grid: List[List[int]]) -> Optional[Dict[str, Any]]:
        """
        Sample deterministic parameters for a transform, based on current grid content.
        If a transform is deterministic or content-based by default, return None.
        """
        colors_present = list(utils.get_colors_in_grid(grid) - {0})
        palette = list(range(10))

        if name == "swap_colors":
            if len(colors_present) >= 2:
                c1, c2 = self.rng.sample(colors_present, 2)
            elif len(colors_present) == 1:
                c1 = colors_present[0]
                c2 = next(c for c in palette if c != c1)
            else:
                c1, c2 = 1, 2
            return {"color1": c1, "color2": c2}

        if name == "remove_color":
            if len(colors_present) <= 1:
                return None
            c = self.rng.choice(colors_present)
            return {"color": c}

        if name in ("add_frame", "add_border"):
            c = self.rng.choice(colors_present) if colors_present else self.rng.choice(palette)
            thickness = 1
            return {"color": c, "thickness": thickness}

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

        if name == "add_noise":
            return {"prob": 0.05, "seed": self.rng.randint(0, 10**9)}

        return None

    # ----------------------------
    # Chain selection + execution
    # ----------------------------
    def select_transformation_chain(
        self,
        grid: List[List[int]],
        chain_length: Optional[int] = None,
        difficulty: str = "medium",
        preserves_size_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Build a *parameterized* chain over the given grid.
        Applies each chosen step to the *current* grid to ensure compatibility,
        freezing sampled parameters along the way.
        """
        if chain_length is None:
            chain_length = (
                self.rng.randint(0, self.max_chain_length)
            )

        if difficulty == "easy":
            pool = list(self.easy_transforms)
        elif difficulty == "medium":
            pool = list(set(self.easy_transforms + self.medium_transforms))
        elif difficulty == "hard":
            pool = list(set(self.medium_transforms + self.hard_transforms))
        else:
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

            # avoid immediate reversals where possible
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
            # if params are invalid (example: would degenerate), try picking another transform
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
        out = utils.deep_copy_grid(grid)
        for step in chain:
            name = step["name"]
            params = step.get("params")
            out = utils.apply_transformation(out, name, params)
            if not utils.is_valid_grid(out):
                raise ValueError(f"Invalid grid after transform: {name}")
        return out

    # ----------------------------
    # Problem generation
    # ----------------------------
    def _non_degenerate(self, grid: List[List[int]]) -> bool:
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
        difficulty: str = "medium",
        return_metadata: bool = True,
        preserves_size_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a single ARC-2 style problem with frozen chain parameters.

        Returns dict:
            {
              "input": <grid>,
              "output": <grid>,             # output after applying chain to base output
              "metadata": {
                  "base_task": <int>,
                  "transformation_chain": [ { "name": str, "params": dict|None }, ... ],
                  "difficulty": str,
                  "chain_length": int,
                  "initial_output": <grid>   # base ARC-1 output (before chain)
              }
            }
        """
        base = self.generate_initial_problem(task_num)
        attempts = 0

        while True:
            chain = self.select_transformation_chain(
                base["output"],
                chain_length=chain_length,
                difficulty=difficulty,
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
                "difficulty": difficulty,
                "chain_length": len(chain),
                "initial_output": base["output"],
            }
        return result
