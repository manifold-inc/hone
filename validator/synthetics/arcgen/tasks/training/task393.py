# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generator."""

from validator.synthetics.arcgen import common


def generate(rows=None, cols=None, idxs=None, brows=None, bcols=None,
             colors=None, size=12, num_boxes=3):
  """Returns input and output grids according to the given parameters.

  Args:
    rows: a list of vertical coordinates where pixels should be placed
    cols: a list of horizontal coordinates where pixels should be placed
    idxs: a list of indices into the sprites list
    brows: a list of vertical coordinates where the sprites should be placed
    bcols: a list of horizontal coordinates where the sprites should be placed
    colors: a list of colors to be used
    size: the width and height of the (square) grid
    num_boxes: the number of boxes to be placed
  """
  if rows is None:
    while True:
      counts = sorted(common.sample(range(3, 15), num_boxes), reverse=True)
      lengths = []
      for count in counts:
        length = 3
        if count > 6: length = 4
        if count > 12: length = 5
        lengths.append(length)
      brows = [common.randint(0, size - length) for length in lengths]
      bcols = [common.randint(0, size - length) for length in lengths]
      if not common.overlaps(brows, bcols, lengths, lengths, 1): break
    rows, cols, idxs = [], [], []
    for idx, length in enumerate(lengths):
      pixels = common.continuous_creature(counts[idx], length, length)
      rows.extend([p[0] for p in pixels])
      cols.extend([p[1] for p in pixels])
      idxs.extend([idx] * len(pixels))
    colors = common.random_colors(num_boxes)

  grid, output = common.grid(size, size), common.grid(1, len(colors))
  for r, c, i in zip(rows, cols, idxs):
    grid[r + brows[i]][c + bcols[i]] = colors[i]
  for i, color in enumerate(colors):
    output[i][0] = color
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(rows=[0, 0, 1, 1, 2, 2, 2, 3, 0, 0, 1, 1, 1, 1, 2, 0, 1, 1, 2],
               cols=[1, 2, 1, 2, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 1, 1, 0, 1, 2],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
               brows=[1, 6, 2], bcols=[1, 3, 8], colors=[3, 2, 8]),
      generate(rows=[0, 0, 0, 1, 1, 1, 2, 2, 3, 0, 1, 1, 1, 2, 2, 0, 1, 1, 2],
               cols=[0, 1, 3, 0, 1, 2, 2, 3, 1, 2, 0, 1, 2, 1, 2, 1, 0, 1, 1],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
               brows=[1, 8, 7], bcols=[7, 8, 2], colors=[1, 7, 2]),
      generate(rows=[0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 0, 1, 1, 1, 2, 2,
                     2, 0, 1, 1, 2, 2],
               cols=[2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2, 1, 2, 3, 0, 1,
                     2, 0, 0, 1, 0, 1],
               idxs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                     1, 2, 2, 2, 2, 2],
               brows=[7, 1, 2], bcols=[3, 1, 8], colors=[4, 2, 1]),
  ]
  test = [
      generate(rows=[0, 1, 1, 2, 2, 2, 3, 0, 0, 1, 1, 1, 0, 1, 1],
               cols=[2, 1, 2, 0, 2, 3, 2, 1, 2, 0, 1, 2, 1, 0, 1],
               idxs=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
               brows=[8, 5, 1], bcols=[5, 3, 1], colors=[6, 1, 3]),
  ]
  return {"train": train, "test": test}
