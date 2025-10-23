import os
import re
import json
from typing import List, Dict, Optional
from openai import OpenAI


class ARCSolver:
    """
    ARC solver using LLM (OpenAI/OpenRouter) with rule-based fallback
    """

    def __init__(self):
        self.api_key = (
            os.getenv('OPENAI_API_KEY') or
            os.getenv('OPENROUTER_API_KEY')
        )
        self.model_id = (
            os.getenv('OPENAI_MODEL') or
            os.getenv('OPENROUTER_MODEL', 'gpt-4o')
        )
        self.base_url = os.getenv(
            'OPENROUTER_BASE_URL',
            'https://openrouter.ai/api/v1'
        )
        self.provider = self._detect_provider()
        self.use_llm = bool(self.api_key)

        if self.use_llm:
            client_config = {'api_key': self.api_key}
            if self.provider == 'openrouter':
                client_config['base_url'] = self.base_url

            self.client = OpenAI(**client_config)
            print(
                f"ARCSolver initialized with {self.provider} "
                f"model: {self.model_id}"
            )
        else:
            print(
                "ARCSolver initialized with rule-based strategies "
                "(no API key)"
            )

        self.strategies = [
            self._identity_transform,
            self._analyze_color_mapping,
            self._analyze_size_transform,
            self._analyze_pattern_transform,
            self._analyze_symmetry
        ]

    def _detect_provider(self) -> str:
        if os.getenv('OPENROUTER_API_KEY'):
            return 'openrouter'
        return 'openai'

    def solve(
        self,
        train_examples: List[Dict],
        test_input: List[List[int]]
    ) -> List[List[int]]:
        """
        Learn from training examples and apply to test input

        Args:
            train_examples: List of dicts with 'input' and 'output'
            test_input: The test input grid to solve
        """
        if not train_examples:
            return [row[:] for row in test_input]

        if self.use_llm:
            try:
                result = self._solve_with_llm(
                    train_examples,
                    test_input
                )
                if result is not None:
                    return result
            except Exception as e:
                print(f"LLM solve failed: {e}, falling back to rules")
        # Fallback to rule-based approach
        transformation = self._identify_transformation(train_examples)

        if transformation and transformation.get("type"):
            return self._apply_learned_transformation(
                test_input,
                transformation
            )

        return self._apply_strategy(test_input, train_examples)

    def _solve_with_llm(
        self,
        train_examples: List[Dict],
        test_input: List[List[int]]
    ) -> Optional[List[List[int]]]:
        """Solve using LLM"""
        prompt = self._format_prompt(train_examples, test_input)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
            )

            content = response.choices[0].message.content
            output_grid = self._parse_grid(content)

            return output_grid

        except Exception as e:
            print(f"LLM API error: {e}")
            return None

    def _format_prompt(
        self,
        train_examples: List[Dict],
        test_input: List[List[int]]
    ) -> str:
        """Format the prompt using the benchmark template"""
        training_text = ""
        for i, ex in enumerate(train_examples, 1):
            training_text += f"Example {i}:\n"
            training_text += (
                f"Input:\n{self._grid_to_string(ex['input'])}\n"
            )
            training_text += (
                f"Output:\n{self._grid_to_string(ex['output'])}\n\n"
            )

        test_text = self._grid_to_string(test_input)

        prompt = (
            "You are participating in a puzzle solving competition. "
            "You are an expert at solving puzzles.\n"
            "Below is a list of input and output pairs with a pattern. "
            "Your goal is to identify the pattern or transformation in "
            "the training examples that maps the input to the output, "
            "then apply that pattern to the test input to give a final "
            "output.\n"
            "Respond in the format of the training output examples\n"
            f"--Training Examples--\n"
            f"{training_text.strip()}\n"
            f"--End of Training Examples--\n"
            f"--Test Input--\n"
            f"{test_text}\n"
            f"--End of Test Input--\n"
            f"Your response:"
        )

        return prompt

    def _grid_to_string(self, grid: List[List[int]]) -> str:
        """Convert grid to string representation"""
        return '\n'.join([' '.join(map(str, row)) for row in grid])

    def _parse_grid(self, content: str) -> Optional[List[List[int]]]:
        """Parse grid from model response with robust parsing"""
        if not content:
            return None

        try:
            # remove any markdown formatting
            content = re.sub(r'```[a-z]*\n?', '', content)
            content = re.sub(r'```', '', content)

            try:
                result = json.loads(content.strip())
                if (isinstance(result, list) and
                        all(isinstance(row, list) for row in result)):
                    return result
            except json.JSONDecodeError:
                pass

            lines = content.strip().split('\n')
            grid = []

            for line in lines:
                line = line.strip()
                if not line or any(
                    c.isalpha() for c in line if c not in '[](),'
                ):
                    continue

                numbers = re.findall(r'\d+', line)
                if numbers:
                    row = [int(n) for n in numbers]
                    grid.append(row)

            if grid and len(grid) > 0:
                row_lengths = [len(row) for row in grid]
                if len(set(row_lengths)) == 1:
                    if all(
                        0 <= val <= 9 for row in grid for val in row
                    ):
                        return grid

            return None

        except Exception:
            return None

    def _identify_transformation(
        self,
        examples: List[Dict]
    ) -> Dict:
        """Analyze training examples to identify the transformation rule"""
        if not examples:
            return {}

        output_sizes = [
            (len(ex["output"]), len(ex["output"][0]))
            for ex in examples
        ]
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
            rotation_count = sum(
                1 for ex in examples
                if self._is_rotated(ex["input"], ex["output"])
            )
            flip_count = sum(
                1 for ex in examples
                if self._is_flipped(ex["input"], ex["output"])
            )

            if rotation_count == len(examples):
                transformation["type"] = "rotation"
            elif flip_count == len(examples):
                transformation["type"] = "flip"

        return transformation

    def _apply_learned_transformation(
        self,
        grid: List[List[int]],
        transformation: Dict
    ) -> List[List[int]]:
        """Apply the learned transformation to new input"""
        if transformation.get("type") == "rotation":
            return self._rotate_90(grid)
        elif transformation.get("type") == "flip":
            return self._flip_horizontal(grid)

        return [row[:] for row in grid]

    def _apply_strategy(
        self,
        grid: List[List[int]],
        examples: List[Dict]
    ) -> List[List[int]]:
        """Try different strategies based on examples"""
        if not examples:
            return [row[:] for row in grid]

        target_size = (
            len(examples[0]["output"]),
            len(examples[0]["output"][0])
        )
        if all(
            len(ex["output"]) == target_size[0] and
            len(ex["output"][0]) == target_size[1]
            for ex in examples
        ):
            if (target_size[0] < len(grid) or
                    target_size[1] < len(grid[0])):
                return self._crop_to_size(grid, target_size)
            elif (target_size[0] > len(grid) or
                    target_size[1] > len(grid[0])):
                return self._expand_to_size(grid, target_size)

        # try basic strategies
        for strategy in self.strategies:
            try:
                result = strategy(grid, examples)
                if self._is_valid_output(result):
                    return result
            except Exception:
                continue

        # last resort: return input
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

    def _is_rotated(
        self,
        grid1: List[List[int]],
        grid2: List[List[int]]
    ) -> bool:
        """Check if grid2 is a rotation of grid1"""
        if (len(grid1) == len(grid2[0]) and
                len(grid1[0]) == len(grid2)):
            rotated = self._rotate_90(grid1)
            return rotated == grid2
        return False

    def _is_flipped(
        self,
        grid1: List[List[int]],
        grid2: List[List[int]]
    ) -> bool:
        """Check if grid2 is a flip of grid1"""
        if (len(grid1) == len(grid2) and
                len(grid1[0]) == len(grid2[0])):
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

    def _flip_horizontal(
        self,
        grid: List[List[int]]
    ) -> List[List[int]]:
        """Flip grid horizontally"""
        return [row[::-1] for row in grid]

    def _crop_to_size(
        self,
        grid: List[List[int]],
        target_size: tuple
    ) -> List[List[int]]:
        """Crop grid to target size"""
        h, w = target_size
        return [row[:w] for row in grid[:h]]

    def _expand_to_size(
        self,
        grid: List[List[int]],
        target_size: tuple
    ) -> List[List[int]]:
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
        examples: List[Dict] = None
    ) -> List[List[int]]:
        """Return the grid as-is"""

        return [row[:] for row in grid]

    def _analyze_color_mapping(
        self,
        grid: List[List[int]],
        examples: List[Dict] = None
    ) -> List[List[int]]:
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

    def _analyze_size_transform(
        self,
        grid: List[List[int]],
        examples: List[Dict] = None
    ) -> List[List[int]]:
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

    def _analyze_pattern_transform(
        self,
        grid: List[List[int]],
        examples: List[Dict] = None
    ) -> List[List[int]]:
        """Look for pattern transformations"""
        return self._pattern_complete(grid)

    def _analyze_symmetry(
        self,
        grid: List[List[int]],
        examples: List[Dict] = None
    ) -> List[List[int]]:
        """Check for symmetry transformations across examples"""

        if examples:
            flip_count = sum(
                1 for ex in examples
                if self._is_flipped(ex["input"], ex["output"])
            )
            if flip_count == len(examples):
                return self._flip_horizontal(grid)
        return grid

    def _pattern_complete(
        self,
        grid: List[List[int]]
    ) -> List[List[int]]:
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
