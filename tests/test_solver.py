import os
from unittest.mock import patch, MagicMock
from miner.arc.solver import ARCSolver


class TestARCSolverInitialization:
    """Test solver initialization with different providers"""

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test-key',
        'OPENAI_MODEL': 'gpt-4o'
    }, clear=True)
    def test_init_with_openai(self):
        with patch('miner.arc.solver.OpenAI'):
            solver = ARCSolver()
            assert solver.provider == 'openai'
            assert solver.use_llm is True
            assert solver.api_key == 'sk-test-key'
            assert solver.model_id == 'gpt-4o'

    @patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'sk-or-test-key',
        'OPENROUTER_MODEL': 'openai/gpt-4o'
    }, clear=True)
    def test_init_with_openrouter(self):
        with patch('miner.arc.solver.OpenAI'):
            solver = ARCSolver()
            assert solver.provider == 'openrouter'
            assert solver.use_llm is True
            assert solver.api_key == 'sk-or-test-key'
            assert solver.model_id == 'openai/gpt-4o'

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key(self):
        solver = ARCSolver()
        assert solver.use_llm is False
        assert solver.api_key is None

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test',
        'OPENROUTER_API_KEY': 'sk-or-test'
    }, clear=True)
    def test_openrouter_takes_precedence(self):
        with patch('miner.arc.solver.OpenAI'):
            solver = ARCSolver()
            assert solver.provider == 'openrouter'
            assert solver.api_key == 'sk-test'


class TestARCSolverLLM:
    """Test LLM solving functionality"""

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test'
    }, clear=True)
    def test_solve_with_llm_success(self):
        with patch('miner.arc.solver.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content='[[1, 2], [3, 4]]'))
            ]
            mock_client.chat.completions.create.return_value = (
                mock_response
            )
            mock_openai.return_value = mock_client

            solver = ARCSolver()
            train_examples = [
                {'input': [[0, 1]], 'output': [[1, 0]]}
            ]
            test_input = [[0, 2]]

            result = solver.solve(train_examples, test_input)
            assert result == [[1, 2], [3, 4]]
            mock_client.chat.completions.create.assert_called_once()

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test'
    }, clear=True)
    def test_solve_with_llm_failure_fallback(self):
        with patch('miner.arc.solver.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = (
                Exception("API Error")
            )
            mock_openai.return_value = mock_client

            solver = ARCSolver()
            train_examples = [
                {'input': [[1, 2]], 'output': [[2, 1]]}
            ]
            test_input = [[3, 4]]

            result = solver.solve(train_examples, test_input)
            assert result == [[4, 3]]

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test'
    }, clear=True)
    def test_solve_with_llm_invalid_response(self):
        with patch('miner.arc.solver.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content='invalid response'))
            ]
            mock_client.chat.completions.create.return_value = (
                mock_response
            )
            mock_openai.return_value = mock_client

            solver = ARCSolver()
            train_examples = [
                {'input': [[1, 2]], 'output': [[2, 1]]}
            ]
            test_input = [[3, 4]]

            result = solver.solve(train_examples, test_input)
            assert result == [[4, 3]]


class TestARCSolverGridParsing:
    """Test grid parsing from different formats"""

    @patch.dict(os.environ, {}, clear=True)
    def test_parse_grid_json_format(self):
        solver = ARCSolver()
        content = '[[1, 2, 3], [4, 5, 6]]'
        result = solver._parse_grid(content)
        assert result == [[1, 2, 3], [4, 5, 6]]

    @patch.dict(os.environ, {}, clear=True)
    def test_parse_grid_text_format(self):
        solver = ARCSolver()
        content = '1 2 3\n4 5 6'
        result = solver._parse_grid(content)
        assert result == [[1, 2, 3], [4, 5, 6]]

    @patch.dict(os.environ, {}, clear=True)
    def test_parse_grid_with_markdown(self):
        solver = ARCSolver()
        content = '```\n1 2 3\n4 5 6\n```'
        result = solver._parse_grid(content)
        assert result == [[1, 2, 3], [4, 5, 6]]

    @patch.dict(os.environ, {}, clear=True)
    def test_parse_grid_empty_content(self):
        solver = ARCSolver()
        result = solver._parse_grid('')
        assert result is None

    @patch.dict(os.environ, {}, clear=True)
    def test_parse_grid_invalid_values(self):
        solver = ARCSolver()
        content = '10 11\n12 13'
        result = solver._parse_grid(content)
        assert result is None

    @patch.dict(os.environ, {}, clear=True)
    def test_parse_grid_irregular_rows(self):
        solver = ARCSolver()
        content = '1 2 3\n4 5'
        result = solver._parse_grid(content)
        assert result is None


class TestARCSolverTransformations:
    """Test transformation detection and application"""

    @patch.dict(os.environ, {}, clear=True)
    def test_rotation_detection(self):
        solver = ARCSolver()
        train_examples = [
            {
                'input': [[1, 2], [3, 4]],
                'output': [[3, 1], [4, 2]]
            }
        ]
        test_input = [[5, 6], [7, 8]]

        result = solver.solve(train_examples, test_input)
        assert result == [[7, 5], [8, 6]]

    @patch.dict(os.environ, {}, clear=True)
    def test_flip_detection(self):
        solver = ARCSolver()
        train_examples = [
            {
                'input': [[1, 2, 3]],
                'output': [[3, 2, 1]]
            }
        ]
        test_input = [[4, 5, 6]]

        result = solver.solve(train_examples, test_input)
        assert result == [[6, 5, 4]]

    @patch.dict(os.environ, {}, clear=True)
    def test_identity_transform(self):
        solver = ARCSolver()
        grid = [[1, 2], [3, 4]]
        result = solver._identity_transform(grid)
        assert result == [[1, 2], [3, 4]]
        assert result is not grid

    @patch.dict(os.environ, {}, clear=True)
    def test_rotate_90(self):
        solver = ARCSolver()
        grid = [[1, 2], [3, 4]]
        result = solver._rotate_90(grid)
        assert result == [[3, 1], [4, 2]]

    @patch.dict(os.environ, {}, clear=True)
    def test_flip_horizontal(self):
        solver = ARCSolver()
        grid = [[1, 2, 3], [4, 5, 6]]
        result = solver._flip_horizontal(grid)
        assert result == [[3, 2, 1], [6, 5, 4]]


class TestARCSolverStrategies:
    """Test rule-based strategies"""

    @patch.dict(os.environ, {}, clear=True)
    def test_color_mapping_strategy(self):
        solver = ARCSolver()
        examples = [
            {
                'input': [[1, 1, 2]],
                'output': [[3, 3, 4]]
            }
        ]
        grid = [[1, 2, 1]]

        result = solver._analyze_color_mapping(grid, examples)
        assert result == [[3, 4, 3]]

    @patch.dict(os.environ, {}, clear=True)
    def test_size_transform_crop(self):
        solver = ARCSolver()
        examples = [
            {
                'input': [[1, 2, 3, 4]],
                'output': [[1, 2]]
            }
        ]
        grid = [[5, 6, 7, 8]]

        result = solver._apply_strategy(grid, examples)
        assert result == [[5, 6]]

    @patch.dict(os.environ, {}, clear=True)
    def test_size_transform_expand(self):
        solver = ARCSolver()
        grid = [[1, 2]]
        target_size = (2, 4)

        result = solver._expand_to_size(grid, target_size)
        assert result == [[1, 2, 0, 0], [0, 0, 0, 0]]


class TestARCSolverValidation:
    """Test output validation"""

    @patch.dict(os.environ, {}, clear=True)
    def test_valid_output(self):
        solver = ARCSolver()
        grid = [[1, 2, 3], [4, 5, 6]]
        assert solver._is_valid_output(grid) is True

    @patch.dict(os.environ, {}, clear=True)
    def test_invalid_empty_grid(self):
        solver = ARCSolver()
        assert solver._is_valid_output([]) is False
        assert solver._is_valid_output([[]]) is False

    @patch.dict(os.environ, {}, clear=True)
    def test_invalid_too_large(self):
        solver = ARCSolver()
        grid = [[0] * 40 for _ in range(40)]
        assert solver._is_valid_output(grid) is False

    @patch.dict(os.environ, {}, clear=True)
    def test_invalid_irregular_rows(self):
        solver = ARCSolver()
        grid = [[1, 2], [3, 4, 5]]
        assert solver._is_valid_output(grid) is False

    @patch.dict(os.environ, {}, clear=True)
    def test_invalid_values(self):
        solver = ARCSolver()
        grid = [[1, 2], [3, 10]]
        assert solver._is_valid_output(grid) is False


class TestARCSolverEdgeCases:
    """Test edge cases and error handling"""

    @patch.dict(os.environ, {}, clear=True)
    def test_solve_empty_training_examples(self):
        solver = ARCSolver()
        test_input = [[1, 2], [3, 4]]

        result = solver.solve([], test_input)
        assert result == [[1, 2], [3, 4]]

    @patch.dict(os.environ, {}, clear=True)
    def test_solve_single_pixel(self):
        solver = ARCSolver()
        train_examples = [
            {'input': [[1]], 'output': [[2]]}
        ]
        test_input = [[3]]

        result = solver.solve(train_examples, test_input)
        assert len(result) == 1
        assert len(result[0]) == 1

    @patch.dict(os.environ, {}, clear=True)
    def test_get_colors(self):
        solver = ARCSolver()
        grid = [[1, 2], [2, 3], [1, 3]]

        colors = solver._get_colors(grid)
        assert colors == {1, 2, 3}

    @patch.dict(os.environ, {}, clear=True)
    def test_grid_to_string(self):
        solver = ARCSolver()
        grid = [[1, 2, 3], [4, 5, 6]]

        result = solver._grid_to_string(grid)
        assert result == '1 2 3\n4 5 6'
