"""
Example ARC-AGI-2 solver using vLLM + simple rule-based heuristics

THIS FILE IS JUST AN EXAMPLE SOLVER:

- You can:
    * Replace this with your own solver (GNN, HRM, custom C++ binary, etc.)
    * Remove vLLM usage if you don't want LLMs
    * Add more heuristics / algorithms / model calls

- You MUST preserve:
    * The class name `ARCSolver` (or at least the public interface used by
      the inference script).
    * The method signature:
          solve(train_examples: List[Dict], test_input: List[List[int]]) -> List[List[int]]
    * The output format: a rectangular 2D grid of integers 0-9

- You MUST NOT:
    * Perform any network calls in `solve()` (no internet during inference)
    * Read or depend on ground-truth `test_output` (it will fail)
"""

import json
import os
from typing import List, Dict, Optional


# This is the model *downloaded in prep phase* by default
# The prep-phase script imports this name
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"


class ARCSolver:
    """
    ARC solver with optional vLLM backend and rule-based fallback strategies

    You can completely replace the internals of this class as long as:

    - The constructor still exists (signature can be extended)
    - `solve(train_examples, test_input)` still returns a 2D int grid
    """

    def __init__(self, use_vllm: bool = True):
        self.use_vllm = use_vllm
        self.vllm_available = False
        self.vllm_client = None
        self.vllm_model_name: Optional[str] = None

        print("🔧 ARCSolver initialized (use_vllm=%s)" % self.use_vllm)

        if use_vllm:
            self._init_vllm_client()

        # rule-based strategies (fallback)
        self.strategies = [
            self._identity_transform,
            self._analyze_color_mapping,
            self._analyze_size_transform,
            self._analyze_pattern_transform,
            self._analyze_symmetry,
        ]

    # ------------------------------------------------------------------
    # vLLM-related helpers
    # ------------------------------------------------------------------

    def _init_vllm_client(self) -> None:
        """
        Initialize vLLM client and discover actual model name.

        This assumes that a vLLM-compatible server is reachable at
        VLLM_API_BASE (e.g. http://vllm-container:8000).
        """
        try:
            from openai import OpenAI

            vllm_api_base = os.environ.get("VLLM_API_BASE", "http://vllm-container:8000")
            print(f"🌐 Attempting to connect to vLLM at: {vllm_api_base}")

            self.vllm_client = OpenAI(
                base_url=f"{vllm_api_base}/v1",
                api_key="dummy",  # vLLM does not require a real API key
            )

            try:
                models = self.vllm_client.models.list()
                self.vllm_available = True
                model_ids = [m.id for m in models.data]
                print(f"✓ vLLM connection successful! Available models: {model_ids}")

                if model_ids:
                    self.vllm_model_name = model_ids[0]
                    print(f"✓ Using vLLM model name: {self.vllm_model_name}")
                else:
                    print("⚠ No models reported by vLLM server")
                    self.vllm_available = False
                    self.vllm_model_name = None
            except Exception as e:
                print(f"⚠ vLLM connection test failed: {e}")
                print("  Falling back to rule-based methods")
                self.vllm_available = False
                self.vllm_model_name = None

        except ImportError:
            print("⚠ OpenAI client not installed, using rule-based methods only")
            self.vllm_available = False
            self.vllm_model_name = None
        except Exception as e:
            print(f"⚠ Failed to initialize vLLM client: {e}")
            self.vllm_available = False
            self.vllm_model_name = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        train_examples: List[Dict],
        test_input: List[List[int]],
    ) -> List[List[int]]:
        """
        Learn from training examples and apply to test input

        Args:
            train_examples: List of dicts with 'input' and 'output' grids.
            test_input: The test input grid to solve.

        Returns:
            A 2D grid (list of lists) of ints in [0, 9].
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

        # fallback: simple rule-based solver
        transformation = self._identify_transformation(train_examples)

        if transformation and transformation.get("type"):
            return self._apply_learned_transformation(test_input, transformation)

        return self._apply_strategy(test_input, train_examples)

    # ------------------------------------------------------------------
    # vLLM solving
    # ------------------------------------------------------------------

    def _solve_with_vllm(
        self,
        train_examples: List[Dict],
        test_input: List[List[int]],
    ) -> Optional[List[List[int]]]:
        """Use vLLM to solve the ARC problem"""
        prompt = self._create_arc_prompt(train_examples, test_input)

        try:
            response = self.vllm_client.chat.completions.create(
                model=self.vllm_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at solving ARC (Abstraction and Reasoning Corpus) "
                            "puzzles. Analyze the pattern in the training examples and apply it "
                            "to the test input. Return ONLY a valid JSON array representing "
                            "the output grid."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()

            # extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            output_grid = json.loads(content)

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

    def _create_arc_prompt(
        self,
        train_examples: List[Dict],
        test_input: List[List[int]],
    ) -> str:
        """Create a text prompt describing the ARC problem"""
        parts: List[str] = ["Solve this ARC puzzle by finding the pattern in the training examples.\n\n"]

        parts.append("Training Examples:\n")
        for i, example in enumerate(train_examples, 1):
            parts.append(f"\nExample {i}:")
            parts.append(f"Input:\n{json.dumps(example['input'])}")
            parts.append(f"Output:\n{json.dumps(example['output'])}\n")

        parts.append("\nNow apply the pattern to this test input:")
        parts.append(f"Test Input:\n{json.dumps(test_input)}\n")

        parts.append("\nReturn ONLY the output grid as a JSON array. Do not include any explanation.")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Heuristic / rule-based methods
    # ------------------------------------------------------------------

    def _identify_transformation(self, examples: List[Dict]) -> Dict:
        """Analyze training examples to identify a simple transformation rule"""
        if not examples:
            return {}

        output_sizes = [(len(ex["output"]), len(ex["output"][0])) for ex in examples]
        same_output_size = len(set(output_sizes)) == 1

        size_preserved = all(
            len(ex["input"]) == len(ex["output"])
            and len(ex["input"][0]) == len(ex["output"][0])
            for ex in examples
        )

        color_mappings = []
        for ex in examples:
            in_colors = self._get_colors(ex["input"])
            out_colors = self._get_colors(ex["output"])
            color_mappings.append((in_colors, out_colors))

        transformation: Dict = {
            "same_output_size": same_output_size,
            "size_preserved": size_preserved,
            "color_mappings": color_mappings,
            "num_examples": len(examples),
        }

        if size_preserved and len(examples) > 0:
            rotation_count = sum(
                1 for ex in examples if self._is_rotated(ex["input"], ex["output"])
            )
            flip_count = sum(
                1 for ex in examples if self._is_flipped(ex["input"], ex["output"])
            )

            if rotation_count == len(examples):
                transformation["type"] = "rotation"
            elif flip_count == len(examples):
                transformation["type"] = "flip"

        return transformation

    def _apply_learned_transformation(
        self,
        grid: List[List[int]],
        transformation: Dict,
    ) -> List[List[int]]:
        """Apply a learned global transformation (rotation / flip)"""
        if transformation.get("type") == "rotation":
            return self._rotate_90(grid)
        elif transformation.get("type") == "flip":
            return self._flip_horizontal(grid)
        return [row[:] for row in grid]

    def _apply_strategy(
        self,
        grid: List[List[int]],
        examples: List[Dict],
    ) -> List[List[int]]:
        """Try different strategies based on examples"""
        if not examples:
            return [row[:] for row in grid]

        target_size = (len(examples[0]["output"]), len(examples[0]["output"][0]))
        if all(
            len(ex["output"]) == target_size[0]
            and len(ex["output"][0]) == target_size[1]
            for ex in examples
        ):
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

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _is_valid_output(self, grid: List[List[int]]) -> bool:
        """Check if output grid is rectangular and within allowed size / values"""
        if not grid or not grid[0]:
            return False

        if len(grid) > 30 or len(grid[0]) > 30:
            return False

        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                return False
            for val in row:
                if not isinstance(val, int) or not (0 <= val <= 9):
                    return False

        return True

    def _get_colors(self, grid: List[List[int]]) -> set:
        """Get set of colors used in a grid"""
        colors = set()
        for row in grid:
            colors.update(row)
        return colors

    def _is_rotated(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if grid2 is a 90° rotation of grid1"""
        if len(grid1) == len(grid2[0]) and len(grid1[0]) == len(grid2):
            rotated = self._rotate_90(grid1)
            return rotated == grid2
        return False

    def _is_flipped(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if grid2 is a horizontal flip of grid1"""
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

    def _crop_to_size(self, grid: List[List[int]], target_size: tuple[int, int]) -> List[List[int]]:
        """Crop grid to target size"""
        h, w = target_size
        return [row[:w] for row in grid[:h]]

    def _expand_to_size(self, grid: List[List[int]], target_size: tuple[int, int]) -> List[List[int]]:
        """Expand grid to target size by padding with zeros"""
        h, w = target_size
        result = [[0] * w for _ in range(h)]
        for i in range(min(len(grid), h)):
            for j in range(min(len(grid[0]), w)):
                result[i][j] = grid[i][j]
        return result

    def _identity_transform(
        self,
        grid: List[List[int]],
        examples: Optional[List[Dict]] = None,
    ) -> List[List[int]]:
        """Return the grid as-is"""
        return [row[:] for row in grid]

    def _analyze_color_mapping(
        self,
        grid: List[List[int]],
        examples: Optional[List[Dict]] = None,
    ) -> List[List[int]]:
        """Analyze global color remapping across examples and apply"""
        if not examples:
            return grid

        color_map: Dict[int, int] = {}
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
                result.append([color_map.get(val, val) for val in row])
            return result

        return grid

    def _analyze_size_transform(
        self,
        grid: List[List[int]],
        examples: Optional[List[Dict]] = None,
    ) -> List[List[int]]:
        """Analyze coarse downscaling patterns across examples"""
        if not examples:
            return grid

        all_smaller = all(
            len(ex["output"]) <= len(ex["input"])
            and len(ex["output"][0]) <= len(ex["input"][0])
            for ex in examples
        )

        if all_smaller and len(grid) > 2:
            result: List[List[int]] = []
            for i in range(0, len(grid), 2):
                row: List[int] = []
                for j in range(0, len(grid[0]), 2):
                    row.append(grid[i][j])
                if row:
                    result.append(row)
            if result and result[0]:
                return result

        return grid

    def _analyze_pattern_transform(
        self,
        grid: List[List[int]],
        examples: Optional[List[Dict]] = None,
    ) -> List[List[int]]:
        """Example pattern-completion heuristic"""
        return self._pattern_complete(grid)

    def _analyze_symmetry(
        self,
        grid: List[List[int]],
        examples: Optional[List[Dict]] = None,
    ) -> List[List[int]]:
        """Check for full horizontal symmetry across examples"""
        if examples:
            flip_count = sum(
                1 for ex in examples if self._is_flipped(ex["input"], ex["output"])
            )
            if flip_count == len(examples):
                return self._flip_horizontal(grid)
        return grid

    def _pattern_complete(self, grid: List[List[int]]) -> List[List[int]]:
        """Toy pattern completion example"""
        h = len(grid)
        w = len(grid[0]) if grid else 0

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
