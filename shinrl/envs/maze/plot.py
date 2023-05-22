from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from chex import Array
from matplotlib.axes import Axes

import itertools
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from .calc import TILE

cmap = cm.get_cmap("RdYlBu")

NOOP = np.array([[-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1], [0.1, 0.1]])
UP = np.array([[0, 0], [0.5, 0.5], [-0.5, 0.5]])
LEFT = np.array([[0, 0], [-0.5, 0.5], [-0.5, -0.5]])
RIGHT = np.array([[0, 0], [0.5, 0.5], [0.5, -0.5]])
DOWN = np.array([[0, 0], [0.5, -0.5], [-0.5, -0.5]])

TXT_OFFSET_VAL = 0.3
TXT_CENTERING = np.array([-0.08, -0.05])
TXT_NOOP = np.array([0.0, 0]) + TXT_CENTERING
TXT_UP = np.array([0, TXT_OFFSET_VAL]) + TXT_CENTERING
TXT_LEFT = np.array([-TXT_OFFSET_VAL, 0]) + TXT_CENTERING
TXT_RIGHT = np.array([TXT_OFFSET_VAL, 0]) + TXT_CENTERING
TXT_DOWN = np.array([0, -TXT_OFFSET_VAL]) + TXT_CENTERING

ACT_OFFSETS = [
    [NOOP, TXT_NOOP],
    [UP, TXT_UP],
    [DOWN, TXT_DOWN],
    [LEFT, TXT_LEFT],
    [RIGHT, TXT_RIGHT],
]

ROTATION = [0, 0, 0, 90, 270]


def plot_SA(
    tb: Array,
    maze: Array,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = 10,
    **kwargs: Any,
) -> None:
    assert len(tb.shape) == 2
    assert len(maze.shape) == 2
    w, h = maze.shape
    maze_SA = np.zeros((w, h, 5))

    for x, y, a in itertools.product(range(w), range(h), range(5)):
        s = x + y * w
        maze_SA[x, h - y - 1, a] = tb[s, a]
    
    # scale color
    color_SA = maze_SA - maze_SA.min()
    color_SA = color_SA / np.max(color_SA)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    for x, y in itertools.product(range(w), range(h)):
        # invert y
        y = h - y - 1
        xy = np.array([x, y])
        xy3 = np.expand_dims(xy, axis=0)

        for a in range(5 - 1, -1, -1):
            color_val = color_SA[x, y, a]
            val = maze_SA[x, y, a]
            patch_offset, txt_offset = ACT_OFFSETS[a]
            rotation = ROTATION[a]
            xy_text = xy + txt_offset
            ax.text(
                xy_text[0],
                xy_text[1],
                f"{val:.1f}",
                size="small",
                rotation=rotation,
            )
            color = cmap(color_val)
            ax.add_patch(Polygon(xy3 + patch_offset, True, color=color))

    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks(np.arange(-1, w + 1, 1))
    ax.set_yticks(np.arange(-1, h + 1, 1))
    ax.grid()


def plot_maze(
    maze: Array,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = 10,
    **kwargs: Any,
) -> None:
    TILE_COLORS = {
        "empty": "white",
        "rew": "yellow",
        "wall": "gray",
        "start": "green",
        "pitfall": "black",
    }
    assert len(maze.shape) == 2
    w, h = maze.shape

    if ax is None:
        plt.figure()
        ax = plt.gca()

    for x, y in itertools.product(range(w), range(h)):
        tile = TILE(maze[x, y]).name
        # invert y
        y = h - y - 1
        xy = np.array([x, y])
        xy3 = np.expand_dims(xy, axis=0)

        color = TILE_COLORS[tile]
        xy_text = xy + np.array([-0.08, -0.05])
        ax.text(
            xy_text[0],
            xy_text[1],
            tile,
            size="small",
        )
        patch_offset = np.array([[-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]])
        ax.add_patch(Polygon(xy3 + patch_offset, True, color=color))

    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks(np.arange(-1, w + 1, 1))
    ax.set_yticks(np.arange(-1, h + 1, 1))
    ax.grid()
