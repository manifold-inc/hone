from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import validator.synthetics.arcgen.arc_agi2_utils as utils
from validator.synthetics.arcgen.arc_agi2_generator import ARC2Generator  # type: ignore


# ARC palette (0..9)
ARC_COLORS = [
    "#000000",  # 0
    "#0074D9",  # 1
    "#FF4136",  # 2
    "#2ECC40",  # 3
    "#FFDC00",  # 4
    "#AAAAAA",  # 5
    "#F012BE",  # 6
    "#FF851B",  # 7
    "#7FDBFF",  # 8
    "#870C25",  # 9
]


def _np(grid: List[List[int]]) -> np.ndarray:
    return np.array(grid, dtype=int)


def _step_name_and_params(step: Union[str, Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Backward-compatible helper:
      - If step is a string => (name, None)
      - If step is a dict like {"name": "...", "params": {...}} => (name, params)
    """
    if isinstance(step, str):
        return step, None
    if isinstance(step, dict):
        return step.get("name"), step.get("params")
    raise TypeError(f"Unexpected step type: {type(step)}")


class Visualizer:
    def __init__(self) -> None:
        self.cmap = colors.ListedColormap(ARC_COLORS)
        self.norm = colors.BoundaryNorm(range(11), ncolors=10)

    def _draw_grid(
        self,
        ax: plt.Axes,
        grid: List[List[int]],
        title: str = "",
        title_above: bool = False,
        grid_lines: bool = True,
        title_size: int = 10,
    ) -> None:
        arr = _np(grid)
        ax.imshow(arr, cmap=self.cmap, norm=self.norm, interpolation="nearest")
        if grid_lines:
            h, w = arr.shape
            for i in range(h + 1):
                ax.axhline(i - 0.5, color="white", linewidth=0.3, alpha=0.5)
            for j in range(w + 1):
                ax.axvline(j - 0.5, color="white", linewidth=0.3, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=title_size, pad=(14 if title_above else None))

    def _arrow_right(self, left_ax: plt.Axes, right_ax: plt.Axes, lw: float = 1.2, color: str = "black") -> None:
        lp, rp = left_ax.get_position(), right_ax.get_position()
        x0 = lp.x1 + 0.006
        y = lp.y1 + 0.02
        x1 = rp.x0 - 0.006
        plt.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            xycoords="figure fraction",
            textcoords="figure fraction",
            arrowprops=dict(arrowstyle="->", lw=lw, color=color),
        )

    # ---------------- chain reconstruction ----------------

    def _chain_tiles(
        self,
        start_grid: Optional[List[List[int]]],
        chain: List[Union[str, Dict[str, Any]]],
        final_output: Optional[List[List[int]]],
    ) -> List[Tuple[str, List[List[int]]]]:
        """
        Build [(label, grid_after_step)], starting at start_grid.

        Left→Right: Base Output → Step 1 → Step 2 → ... → Final Output (if different).
        """
        tiles: List[Tuple[str, List[List[int]]]] = []
        if start_grid is not None:
            current = utils.deep_copy_grid(start_grid)
            tiles.append(("Base Output", current))

            for i, raw_step in enumerate(chain, start=1):
                name, params = _step_name_and_params(raw_step)
                current = utils.apply_transformation(current, name, params)
                tiles.append((f"Step {i}: {name}", current))

            if final_output is not None and tiles[-1][1] != final_output:
                tiles.append(("Final Output", final_output))
        elif final_output is not None:
            tiles.append(("Final Output", final_output))
        return tiles

    # ---------------- main plot ----------------

    def plot(
        self,
        problem: Dict[str, Any],
        train_examples: Optional[List[Dict[str, Any]]] = None,
        test_examples: Optional[List[Dict[str, Any]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        meta = (problem or {}).get("metadata", {}) or {}
        chain: List[Union[str, Dict[str, Any]]] = meta.get("transformation_chain", []) or []

        base_task = meta.get("base_task")
        base_input = problem.get("input")
        base_output = meta.get("initial_output")
        final_output = problem.get("output")

        chain_tiles = self._chain_tiles(base_output, chain, final_output)

        n_base_cols = (1 if base_input is not None else 0) + (1 if base_output is not None else 0)
        n_chain_cols = max(1, len(chain_tiles))
        have_examples = bool(train_examples or test_examples)
        n_rows = 2 if have_examples else 1
        total_cols = n_base_cols + n_chain_cols

        if figsize is None:
            width = max(12.0, 1.8 * total_cols)
            height = 8.2 if n_rows == 2 else 4.6
            figsize = (width, height)

        fig = plt.figure(figsize=figsize)
        gs_top = fig.add_gridspec(n_rows, total_cols, wspace=0.28, hspace=0.34)

        fig.suptitle(
            f"ARC-AGI-2 | Base Task: {base_task if base_task is not None else 'None'}",
            fontsize=14,
        )

        col = 0

        # ---- BASE TASK (Input → Output) ----
        ax_in = None
        if base_input is not None:
            ax_in = fig.add_subplot(gs_top[0, col])
            title = f"Base Input\n(Task #{base_task})" if base_task is not None else "Base Input"
            self._draw_grid(ax_in, base_input, title=title)
            col += 1

        ax_out = None
        if base_output is not None:
            ax_out = fig.add_subplot(gs_top[0, col])
            title = f"Base Output\n(Task #{base_task})" if base_task is not None else "Base Output"
            self._draw_grid(ax_out, base_output, title=title)
            if ax_in is not None:
                self._arrow_right(ax_in, ax_out)
            col += 1

        # ---- CHAIN (Base Output → Step 1 → …) ----
        prev_ax: Optional[plt.Axes] = ax_out
        tiles_iter = chain_tiles[1:] if chain_tiles and chain_tiles[0][0] == "Base Output" else chain_tiles
        for label, grid in tiles_iter:
            ax = fig.add_subplot(gs_top[0, col])
            self._draw_grid(ax, grid, title=label, title_above=True)
            if prev_ax is not None:
                self._arrow_right(prev_ax, ax)
            prev_ax = ax
            col += 1

        # Chain text
        if chain:
            chain_names = [(_step_name_and_params(s)[0] or "?") for s in chain]
            fig.text(
                0.5,
                0.07 if n_rows == 2 else 0.08,
                "Transformation Chain: " + " → ".join(chain_names),
                ha="center",
                fontsize=10,
                style="italic",
            )

        # ---- EXAMPLES ROW: Input→Output PAIRS ----
        if n_rows == 2:
            trains = (train_examples or [])[:3]
            tests = (test_examples or [])[:1]

            num_train = len(trains)
            num_test = len(tests)

            extra_div = 1 if num_test > 0 else 0
            ex_cols = max(2 * num_train + extra_div + 2 * num_test, 2)

            gs_ex = fig.add_gridspec(
                1, ex_cols, left=0.06, right=0.94, bottom=0.05, top=0.33, wspace=0.22
            )

            col_e = 0

            def draw_pair(inp: List[List[int]], outp: List[List[int]], title_prefix: str, arrow_color: str = "green",
                          highlight: bool = False) -> None:
                nonlocal col_e
                axA = fig.add_subplot(gs_ex[0, col_e])
                self._draw_grid(axA, inp, title=f"{title_prefix} Input", title_size=10)
                col_e += 1
                axB = fig.add_subplot(gs_ex[0, col_e])
                self._draw_grid(axB, outp, title=f"{title_prefix} Output", title_size=10)
                if highlight:
                    for sp in list(axA.spines.values()) + list(axB.spines.values()):
                        sp.set_edgecolor("purple")
                        sp.set_linewidth(2)
                self._arrow_right(axA, axB, lw=1.5, color=arrow_color)
                col_e += 1

            for i, ex in enumerate(trains, start=1):
                draw_pair(ex["input"], ex["output"], f"Train {i}", arrow_color="green", highlight=False)

            if num_test > 0:
                ax_div = fig.add_subplot(gs_ex[0, col_e])
                ax_div.axis("off")
                x0, y0, x1, y1 = ax_div.get_position().x0, ax_div.get_position().y0, ax_div.get_position().x1, ax_div.get_position().y1
                plt.plot(
                    [(x0 + x1) / 2, (x0 + x1) / 2],
                    [y0, y1],
                    transform=fig.transFigure,
                    linestyle=(0, (3, 5)),
                    linewidth=1.4,
                )
                col_e += 1

            for i, ex in enumerate(tests, start=1):
                draw_pair(ex["input"], ex["output"], f"Test {i}", arrow_color="purple", highlight=True)

        return fig


# ---------- CLI demo ----------
def _demo(task_num:int|None=None) -> None:
    """
    Generates ONE problem, then resamples the same base task 3+1 times and
    applies the SAME chain (with params if present) to produce train/test examples.
    """
    try:

        gen = ARC2Generator()
        problem = gen.generate_problem(task_num=task_num, return_metadata=True)

        meta = problem.get("metadata", {}) or {}
        base_task = meta.get("base_task")
        chain = meta.get("transformation_chain", []) or []

        def make_example() -> Dict[str, Any]:
            initial = gen.generate_initial_problem(task_num=base_task)
            cur = utils.deep_copy_grid(initial["output"])
            for raw_step in chain:
                name, params = _step_name_and_params(raw_step)
                cur = utils.apply_transformation(cur, name, params)
            return {"input": initial["input"], "output": cur}

        train = [make_example() for _ in range(3)]
        test = [make_example() for _ in range(1)]

        fig = Visualizer().plot(problem, train_examples=train, test_examples=test)
        plt.savefig(f"samples/{task_num}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Visualization saved as arc_visualization.png")
    except Exception as e:
        print("viz demo failed:", e)


if __name__ == "__main__":
    for task_id in range(1, 400):
        _demo(task_id)
