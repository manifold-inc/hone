"""Utility functions for ARC-AGI-2 transformations."""

import random
from typing import List, Tuple, Optional, Dict, Any


def deep_copy_grid(grid: List[List[int]]) -> List[List[int]]:
    """Deep copy a grid to avoid mutations."""
    return [row[:] for row in grid]


def get_grid_size(grid: List[List[int]]) -> Tuple[int, int]:
    """Return height, width of grid."""
    if not grid:
        return 0, 0
    return len(grid), len(grid[0]) if grid else 0


def is_valid_grid(grid: List[List[int]]) -> bool:
    """Check if grid is valid (non-empty, rectangular, valid colors)."""
    if not grid or not grid[0]:
        return False
    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            return False
        for val in row:
            if not isinstance(val, int) or val < 0 or val > 9:
                return False
    return True


def get_colors_in_grid(grid: List[List[int]]) -> set:
    """Get all unique colors present in grid."""
    colors = set()
    for row in grid:
        colors.update(row)
    return colors


def count_color(grid: List[List[int]], color: int) -> int:
    """Count occurrences of a specific color."""
    count = 0
    for row in grid:
        count += row.count(color)
    return count


# ============= GEOMETRIC TRANSFORMATIONS =============

def rotate_90(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Rotate grid 90 degrees clockwise."""
    if not grid:
        return grid
    h, w = get_grid_size(grid)
    rotated = [[0 for _ in range(h)] for _ in range(w)]
    for i in range(h):
        for j in range(w):
            rotated[j][h - 1 - i] = grid[i][j]
    return rotated


def rotate_180(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Rotate grid 180 degrees."""
    if not grid:
        return grid
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Rotate grid 270 degrees clockwise (90 counter-clockwise)."""
    return rotate_90(rotate_180(grid))


def flip_horizontal(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Flip grid horizontally (mirror left-right)."""
    return [row[::-1] for row in grid]


def flip_vertical(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Flip grid vertically (mirror top-bottom)."""
    return grid[::-1]


def transpose(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Transpose grid (swap rows and columns)."""
    if not grid:
        return grid
    h, w = get_grid_size(grid)
    return [[grid[i][j] for i in range(h)] for j in range(w)]


def flip_diagonal(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Flip along main diagonal."""
    return transpose(grid)


def flip_antidiagonal(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Flip along anti-diagonal."""
    return rotate_90(flip_vertical(grid))


# ============= SPATIAL OPERATIONS =============

def crop(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Crop a rectangular region from grid."""
    h, w = get_grid_size(grid)
    if h <= 2 or w <= 2:
        return grid

    if params is None:
        # auto-detect interesting region (non-black area)
        min_r, max_r, min_c, max_c = h, -1, w, -1
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

        if max_r < 0:  # all black, random crop
            size = max(1, min(h // 2, w // 2, 5))
            row = random.randint(0, h - size)
            col = random.randint(0, w - size)
            return [grid[r][col:col + size] for r in range(row, row + size)]
        else:
            # crop around content with small margin
            margin = 1
            min_r = max(0, min_r - margin)
            max_r = min(h - 1, max_r + margin)
            min_c = max(0, min_c - margin)
            max_c = min(w - 1, max_c + margin)
            return [grid[r][min_c:max_c + 1] for r in range(min_r, max_r + 1)]
    else:
        row = int(params.get('row', 0))
        col = int(params.get('col', 0))
        height = int(params.get('height', min(5, max(0, h - row))))
        width = int(params.get('width', min(5, max(0, w - col))))
        return [grid[r][col:col + width] for r in range(row, min(row + height, h))]


def shift(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """
    Shift grid in a direction.

    Default: **no wrapping** (pad with 0). Set params['wrap']=True to wrap.
    Params:
      - direction: 'up'|'down'|'left'|'right' (default random/right)
      - amount: int >= 0 (default 1..3 if not provided)
      - wrap: bool (default False)
    """
    h, w = get_grid_size(grid)
    if params is None:
        direction = random.choice(['up', 'down', 'left', 'right'])
        amount = random.randint(1, 3)
        wrap = False
    else:
        direction = params.get('direction', 'right')
        amount = int(params.get('amount', 1))
        wrap = bool(params.get('wrap', False))

    amount = max(0, amount)

    if wrap:
        # toroidal shift (previous behavior)
        result = deep_copy_grid(grid)
        if direction == 'up':
            amount %= max(1, h)
            result = result[amount:] + result[:amount]
        elif direction == 'down':
            amount %= max(1, h)
            result = result[-amount:] + result[:-amount]
        elif direction == 'left':
            amount %= max(1, w)
            for i in range(h):
                result[i] = result[i][amount:] + result[i][:amount]
        elif direction == 'right':
            amount %= max(1, w)
            for i in range(h):
                result[i] = result[i][-amount:] + result[i][:-amount]
        return result

    # non-wrapping: pad with 0
    result = [[0 for _ in range(w)] for _ in range(h)]
    if direction == 'up':
        if amount < h:
            for r in range(amount, h):
                result[r - amount] = grid[r][:]
    elif direction == 'down':
        if amount < h:
            for r in range(h - 1 - amount, -1, -1):
                result[r + amount] = grid[r][:]
    elif direction == 'left':
        if amount < w:
            for r in range(h):
                result[r][:w - amount] = grid[r][amount:]
    elif direction == 'right':
        if amount < w:
            for r in range(h):
                result[r][amount:] = grid[r][:w - amount]
    return result


def recenter(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Center non-black content in the grid."""
    h, w = get_grid_size(grid)

    # find bounding box of non-black pixels
    min_r, max_r, min_c, max_c = h, -1, w, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    if max_r < 0:  # all black
        return grid

    content_h = max_r - min_r + 1
    content_w = max_c - min_c + 1

    # create new grid with centered content
    result = [[0 for _ in range(w)] for _ in range(h)]
    start_r = (h - content_h) // 2
    start_c = (w - content_w) // 2

    for r in range(content_h):
        for c in range(content_w):
            if start_r + r < h and start_c + c < w:
                result[start_r + r][start_c + c] = grid[min_r + r][min_c + c]

    return result


# ============= ZOOM/SCALE OPERATIONS =============

def zoom_2x(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Zoom in 2x (each pixel becomes 2x2)."""
    h, w = get_grid_size(grid)
    result = [[0 for _ in range(w * 2)] for _ in range(h * 2)]
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            result[r * 2][c * 2] = color
            result[r * 2][c * 2 + 1] = color
            result[r * 2 + 1][c * 2] = color
            result[r * 2 + 1][c * 2 + 1] = color
    return result


def zoom_3x(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Zoom in 3x (each pixel becomes 3x3)."""
    h, w = get_grid_size(grid)
    result = [[0 for _ in range(w * 3)] for _ in range(h * 3)]
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            for dr in range(3):
                for dc in range(3):
                    result[r * 3 + dr][c * 3 + dc] = color
    return result


def downsample_2x(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Downsample by 2x (take every other pixel)."""
    h, w = get_grid_size(grid)
    return [[grid[r * 2][c * 2] for c in range(w // 2)]
            for r in range(h // 2) if r * 2 < h]


# ============= COLOR OPERATIONS =============

def invert_colors(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Invert colors: keep 0, map 1..9 <-> 9..1."""
    return [[(10 - v) if 1 <= v <= 9 else 0 for v in row] for row in grid]


def swap_colors(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Swap two non-black colors (avoid swapping the background 0)."""
    palette = list(get_colors_in_grid(grid) - {0})
    if len(palette) < 2:
        return grid

    if params is None:
        c1, c2 = random.sample(palette, 2)
    else:
        c1 = params.get('color1')
        c2 = params.get('color2')
        if c1 not in palette:
            c1 = palette[0]
        if c2 not in palette or c2 == c1:
            # pick something different from c1
            c2 = next((c for c in palette if c != c1), c1)

    result = deep_copy_grid(grid)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if result[r][c] == c1:
                result[r][c] = c2
            elif result[r][c] == c2:
                result[r][c] = c1
    return result


def remove_color(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Remove a color (set to black)."""
    colors = list(get_colors_in_grid(grid) - {0})
    if not colors:
        return grid

    if params is None:
        color_to_remove = random.choice(colors)
    else:
        color_to_remove = params.get('color', colors[0])

    return [[0 if val == color_to_remove else val for val in row] for row in grid]


def highlight_color(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Highlight one color, dim others."""
    colors = list(get_colors_in_grid(grid) - {0})
    if not colors:
        return grid

    if params is None:
        highlight = random.choice(colors)
    else:
        highlight = params.get('color', colors[0])

    return [[val if val == highlight or val == 0 else 5 for val in row] for row in grid]


# ============= FRAME/BORDER OPERATIONS =============

def add_frame(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Add a frame/border around the grid."""
    h, w = get_grid_size(grid)

    if params is None:
        color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        thickness = 1
    else:
        color = params.get('color', 3)
        thickness = int(params.get('thickness', 1))

    new_h = h + 2 * thickness
    new_w = w + 2 * thickness
    result = [[color for _ in range(new_w)] for _ in range(new_h)]

    for r in range(h):
        for c in range(w):
            result[r + thickness][c + thickness] = grid[r][c]

    return result


def add_border(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Add a 1-pixel border (frame) around content."""
    return add_frame(grid, params)


# ============= PHYSICS/GRAVITY OPERATIONS =============

def gravity_down(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Apply gravity - non-black pixels fall down."""
    h, w = get_grid_size(grid)
    result = [[0 for _ in range(w)] for _ in range(h)]

    for c in range(w):
        write_pos = h - 1
        for r in range(h - 1, -1, -1):
            if grid[r][c] != 0:
                result[write_pos][c] = grid[r][c]
                write_pos -= 1

    return result


def gravity_up(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Apply gravity upward - non-black pixels float up."""
    h, w = get_grid_size(grid)
    result = [[0 for _ in range(w)] for _ in range(h)]

    for c in range(w):
        write_pos = 0
        for r in range(h):
            if grid[r][c] != 0:
                result[write_pos][c] = grid[r][c]
                write_pos += 1

    return result


def gravity_left(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Apply gravity leftward."""
    h, w = get_grid_size(grid)
    result = [[0 for _ in range(w)] for _ in range(h)]

    for r in range(h):
        write_pos = 0
        for c in range(w):
            if grid[r][c] != 0:
                result[r][write_pos] = grid[r][c]
                write_pos += 1

    return result


def gravity_right(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Apply gravity rightward."""
    h, w = get_grid_size(grid)
    result = [[0 for _ in range(w)] for _ in range(h)]

    for r in range(h):
        write_pos = w - 1
        for c in range(w - 1, -1, -1):
            if grid[r][c] != 0:
                result[r][write_pos] = grid[r][c]
                write_pos -= 1

    return result


# ============= NOISE OPERATIONS =============

def add_noise(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Add random noise pixels (only into black cells)."""
    h, w = get_grid_size(grid)
    result = deep_copy_grid(grid)

    if params is None:
        num_pixels = max(1, (h * w) // 20)
        noise_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        num_pixels = int(params.get('num_pixels', (h * w) // 20))
        noise_colors = list(params.get('colors', [1, 2, 3, 4, 5, 6, 7, 8, 9]))

    for _ in range(num_pixels):
        r = random.randint(0, h - 1)
        c = random.randint(0, w - 1)
        if result[r][c] == 0:  # only add noise to black pixels
            result[r][c] = random.choice(noise_colors)

    return result


def remove_noise(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Remove isolated pixels (noise)."""
    h, w = get_grid_size(grid)
    result = deep_copy_grid(grid)

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                # check if pixel is isolated
                neighbors = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                        neighbors += 1
                if neighbors == 0:
                    result[r][c] = 0

    return result


# ============= PATTERN OPERATIONS =============

def add_grid_lines(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Add grid lines at regular intervals (draw only over black)."""
    h, w = get_grid_size(grid)
    result = deep_copy_grid(grid)

    if params is None:
        spacing = random.choice([2, 3, 4])
        color = random.choice([1, 2, 3, 5])
    else:
        spacing = int(params.get('spacing', 3))
        color = int(params.get('color', 5))

    for r in range(0, h, spacing):
        for c in range(w):
            if result[r][c] == 0:
                result[r][c] = color

    for c in range(0, w, spacing):
        for r in range(h):
            if result[r][c] == 0:
                result[r][c] = color

    return result


def fill_background(grid: List[List[int]], params: Optional[Dict] = None) -> List[List[int]]:
    """Fill black background with a color."""
    if params is None:
        color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    else:
        color = int(params.get('color', 5))

    return [[color if val == 0 else val for val in row] for row in grid]


# ============= TRANSFORMATION REGISTRY =============

TRANSFORMATIONS = {
    # Geometric
    'rotate_90': (rotate_90, {'type': 'geometric', 'preserves_size': False}),
    'rotate_180': (rotate_180, {'type': 'geometric', 'preserves_size': True}),
    'rotate_270': (rotate_270, {'type': 'geometric', 'preserves_size': False}),
    'flip_horizontal': (flip_horizontal, {'type': 'geometric', 'preserves_size': True}),
    'flip_vertical': (flip_vertical, {'type': 'geometric', 'preserves_size': True}),
    'transpose': (transpose, {'type': 'geometric', 'preserves_size': False}),

    'flip_diagonal': (flip_diagonal, {'type': 'geometric', 'preserves_size': False}),
    'flip_antidiagonal': (flip_antidiagonal, {'type': 'geometric', 'preserves_size': False}),

    # Spatial
    'crop': (crop, {'type': 'spatial', 'preserves_size': False}),
    'shift': (shift, {'type': 'spatial', 'preserves_size': True}),
    'recenter': (recenter, {'type': 'spatial', 'preserves_size': True}),

    # Scale
    'zoom_2x': (zoom_2x, {'type': 'scale', 'preserves_size': False}),
    'zoom_3x': (zoom_3x, {'type': 'scale', 'preserves_size': False}),
    'downsample_2x': (downsample_2x, {'type': 'scale', 'preserves_size': False}),

    # Color
    'invert_colors': (invert_colors, {'type': 'color', 'preserves_size': True}),
    'swap_colors': (swap_colors, {'type': 'color', 'preserves_size': True}),
    'remove_color': (remove_color, {'type': 'color', 'preserves_size': True}),
    'highlight_color': (highlight_color, {'type': 'color', 'preserves_size': True}),

    # Frame
    'add_frame': (add_frame, {'type': 'frame', 'preserves_size': False}),
    'add_border': (add_border, {'type': 'frame', 'preserves_size': False}),

    # Physics
    'gravity_down': (gravity_down, {'type': 'physics', 'preserves_size': True}),
    'gravity_up': (gravity_up, {'type': 'physics', 'preserves_size': True}),
    'gravity_left': (gravity_left, {'type': 'physics', 'preserves_size': True}),
    'gravity_right': (gravity_right, {'type': 'physics', 'preserves_size': True}),

    # Pattern
    'add_grid_lines': (add_grid_lines, {'type': 'pattern', 'preserves_size': True}),
    'fill_background': (fill_background, {'type': 'pattern', 'preserves_size': True}),

    # Noise
    'add_noise': (add_noise, {'type': 'noise', 'preserves_size': True}),
    'remove_noise': (remove_noise, {'type': 'noise', 'preserves_size': True}),
}


def get_compatible_transformations(
    grid: List[List[int]],
    exclude_types: Optional[List[str]] = None,
    max_size: int = 30
) -> List[str]:
    """Get list of transformations compatible with current grid."""
    h, w = get_grid_size(grid)
    compatible = []

    for name, (_, meta) in TRANSFORMATIONS.items():
        if exclude_types and meta['type'] in exclude_types:
            continue

        # size constraints
        if name == 'zoom_2x' and (h * 2 > max_size or w * 2 > max_size):
            continue
        if name == 'zoom_3x' and (h * 3 > max_size or w * 3 > max_size):
            continue
        if name == 'downsample_2x' and (h < 4 or w < 4):
            continue
        if name == 'crop' and (h <= 3 or w <= 3):
            continue
        # conservative check for frames (default thickness=1)
        if name in ('add_frame', 'add_border') and (h + 2 > max_size or w + 2 > max_size):
            continue

        compatible.append(name)

    return compatible


def apply_transformation(
    grid: List[List[int]],
    transform_name: str,
    params: Optional[Dict] = None
) -> List[List[int]]:
    """Apply a named transformation to grid."""
    if transform_name not in TRANSFORMATIONS:
        raise ValueError(f"Unknown transformation: {transform_name}")

    func, _ = TRANSFORMATIONS[transform_name]
    return func(grid, params)
