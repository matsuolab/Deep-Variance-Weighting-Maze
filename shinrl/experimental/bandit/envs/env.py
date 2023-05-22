from abc import abstractmethod, abstractproperty

import chex
import gym
import numpy as np
from scipy.stats import bernoulli

Array = np.ndarray


class BaseBandit(gym.Env):
    """
    An abstractclass for bandit environments.
    This is not powered by JAX (numpy is faster) and ShinEnv (oracle analysis is not necessary).
    """

    @abstractproperty
    def rew_mean(self) -> np.ndarray:
        "Average reward of the arms"
        pass

    @abstractmethod
    def step(self, action: int):
        pass

    def seed(self, seed: int = 0) -> None:
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.np_random.seed(seed)
        self.action_space.np_random.seed(seed)
        self.observation_space.np_random.seed(seed)

    @property
    def dA(self) -> int:
        return len(self.rew_mean)

    @property
    def dS(self) -> int:
        return 1

    @property
    def observation_space(self) -> gym.spaces.Space:
        obs = np.array([0])
        return gym.spaces.Box(low=obs, high=obs, dtype=float)

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(self.dA)

    def observation(self, action: int):
        return np.array([0.0])

    def reset(self):
        return self.observation(0)


class GaussianBandit(BaseBandit):
    """A simple bandit environment with gaussian noise."""

    def __init__(self, rew_mean: Array, noise_scale: Array):
        super().__init__()

        chex.assert_equal_shape([rew_mean, noise_scale])
        chex.assert_rank([rew_mean, noise_scale], 1)
        self._rew_mean = rew_mean
        self._noise_scale = noise_scale
        self.seed(0)

    @property
    def rew_mean(self) -> np.ndarray:
        return self._rew_mean

    @rew_mean.setter
    def rew_mean(self, rew_mean):
        assert rew_mean.shape == self.rew_mean.shape
        self._rew_mean = rew_mean

    def step(self, action: int):
        obs = self.observation(action)
        mean = self.rew_mean[action]
        noise = self.np_random.randn() * self._noise_scale[action]
        reward = mean + noise
        return obs, reward, False, {}


class BernoulliBandit(BaseBandit):
    """A simple Bernoulli bandit environment."""

    def __init__(self, probs: Array):
        super().__init__()

        chex.assert_rank([probs], 1)
        assert np.all(probs >= 0) and np.all(probs <= 1)
        self._probs = probs
        self.seed(0)

    @property
    def rew_mean(self) -> np.ndarray:
        return self._probs

    @rew_mean.setter
    def rew_mean(self, rew_mean):
        assert rew_mean.shape == self.rew_mean.shape
        self._rew_mean = rew_mean

    def step(self, action: int):
        obs = self.observation(action)
        prob = self._probs[action]
        reward = np.random.binomial(1, prob)
        return obs, reward, False, {}
