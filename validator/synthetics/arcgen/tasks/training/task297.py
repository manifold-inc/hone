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


def generate(colors=None):
  """Returns input and output grids according to the given parameters.

  Args:
    colors: the list of colors to use
  """
  if colors is None:
    colors = common.random_colors(common.randint(2, 6), exclude=[common.gray()])

  width, height = len(colors), 2 * len(colors) + 2
  grid, output = common.grids(width, height)
  for color_idx, color in enumerate(colors):
    output[0][color_idx] = grid[0][color_idx] = color
    output[1][color_idx] = grid[1][color_idx] = common.gray()
    for c in range(width):
      output[color_idx + 2 + len(colors)][c] = output[color_idx + 2][c] = color
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(colors=[2, 1, 4]),
      generate(colors=[3, 2, 1, 4]),
      generate(colors=[8, 3]),
  ]
  test = [
      generate(colors=[1, 2, 3, 4, 8]),
  ]
  return {"train": train, "test": test}
