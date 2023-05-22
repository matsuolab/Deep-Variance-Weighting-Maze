import random
from typing import Any, Optional

import gym
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array
from matplotlib.axes import Axes

from shinrl import ShinEnv

from .calc import init_probs, onehot_observation, reward, str_to_maze_array, transition
from .config import MazeConfig
from .plot import plot_maze, plot_SA


class Maze(ShinEnv):
    DefaultConfig = MazeConfig

    def __init__(self, maze: Array, config: Optional[MazeConfig] = None):
        self.maze = maze
        dS = maze.shape[0] * maze.shape[1]
        shape = (dS, config.random_obs_dim)
        key = jax.random.PRNGKey(config.random_obs_seed)
        self.random_obs = jax.random.normal(key, shape)

        x, y = maze.shape
        xytype_obs = np.hstack((np.tile(np.arange(0, x), y)[:, None], np.repeat(np.arange(0, y), x)[:, None]))
        self.xytype_obs = np.hstack((xytype_obs, maze.reshape(-1, 1)))
        self.xytype_obs = jnp.array(self.xytype_obs / self.xytype_obs.max())
        super().__init__(config)

    @property
    def config(self) -> MazeConfig:
        return self._config

    @property
    def dS(self) -> int:
        return self.maze.shape[0] * self.maze.shape[1]

    @property
    def dA(self) -> int:
        return 5

    @property
    def observation_space(self) -> gym.spaces.Space:
        if self.config.obs_mode == MazeConfig.OBS_MODE.onehot:
            space = gym.spaces.Box(0, 1, (self.maze.shape[0] + self.maze.shape[1],))
        elif self.config.obs_mode == MazeConfig.OBS_MODE.random:
            space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.config.random_obs_dim,),
                dtype=float,
            )
        elif self.config.obs_mode == MazeConfig.OBS_MODE.xytype:
            space = gym.spaces.Box(0, 1, (3,))
        return space

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(5)

    def init_probs(self) -> Array:
        init_states, probs = init_probs(self.maze)
        res = np.zeros(self.dS)
        np.put(res, init_states, probs)
        return jnp.array(res)

    def transition(self, state, action):
        return transition(self.config, self.maze, state, action)

    def reward(self, state, action):
        return reward(self.maze, state, action)

    def observation(self, state):
        if self.config.obs_mode == MazeConfig.OBS_MODE.onehot:
            return onehot_observation(self.maze, state)
        elif self.config.obs_mode == MazeConfig.OBS_MODE.random:
            return self.random_obs[state]
        elif self.config.obs_mode == MazeConfig.OBS_MODE.xytype:
            return self.xytype_obs[state]
 
    @staticmethod
    def str_to_maze_array(string: str) -> Array:
        return str_to_maze_array(string)

    @staticmethod
    def create_random_maze_str(width: int, height: int) -> str:
        direction = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        maze = np.ones((height, width))
        maze[0, 0] = 0  # start position

        def fill_maze(updX, updY):
            rnd_array = list(range(4))
            random.shuffle(rnd_array)
            for index in rnd_array:
                # if it reaches out of the maze or visited cell
                if (
                    updY + direction[index][1] < 0
                    or updY + direction[index][1] > maze.shape[0]
                ):
                    continue
                elif (
                    updX + direction[index][0] < 0
                    or updX + direction[index][0] > maze.shape[1]
                ):
                    continue
                elif maze[updY + direction[index][1]][updX + direction[index][0]] == 0:
                    continue

                maze[updY + direction[index][1]][updX + direction[index][0]] = 0
                if index == 0:
                    maze[updY + direction[index][1] + 1][updX + direction[index][0]] = 0
                elif index == 1:
                    maze[updY + direction[index][1] - 1][updX + direction[index][0]] = 0
                elif index == 2:
                    maze[updY + direction[index][1]][updX + direction[index][0] + 1] = 0
                elif index == 3:
                    maze[updY + direction[index][1]][updX + direction[index][0] - 1] = 0
                fill_maze(updX + direction[index][0], updY + direction[index][1])

        fill_maze(0, 0)
        maze[0, 0] = -1  # start position
        maze[height - 1, width - 1] = 2  # goal position

        string = ""
        for dy_list in maze:
            for item in dy_list:
                if item == -1:
                    string += "S"
                elif item == 0:
                    string += "O"
                elif item == 1:
                    string += "#"
                elif item == 2:
                    string += "R"
            string += "\\"
        return string

    def plot_SA(
        self,
        tb: Array,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        fontsize: Optional[int] = 10,
        **kwargs: Any,
    ) -> None:
        plot_SA(tb, self.maze, title, ax, fontsize, **kwargs)

    def plot_maze(
        self,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        fontsize: Optional[int] = 10,
        **kwargs: Any,
    ) -> None:
        plot_maze(self.maze, title, ax, fontsize, **kwargs)
