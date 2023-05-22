"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import enum
from enum import auto
from typing import ClassVar, List, Optional, Type

import chex
import gym
import numpy as np
from scipy import special

import shinrl as srl

from ._bandit_eval_mixin import BanditEvalMixIn


class Exp3_TYPE(enum.IntEnum):
    vanilla = auto()


@chex.dataclass
class Exp3Config(srl.SolverConfig):
    Exp3_TYPE: ClassVar[Type[Exp3_TYPE]] = Exp3_TYPE
    exp3_type: Exp3_TYPE = Exp3_TYPE.vanilla
    lr: float = 0.1


class Exp3Solver(srl.BaseSolver):
    """A simple Exp3 algorithm.
    See Chapter 11 in https://tor-lattimore.com/downloads/book/book.pdf .
    """

    DefaultConfig = Exp3Config

    def make_mixins(env: gym.Env, config: Exp3Config) -> List[Type[object]]:
        mixin_list = [BanditEvalMixIn, Exp3Solver]
        return mixin_list

    def initialize(self, env: gym.Env, config: Optional[Exp3Config] = None) -> None:
        super().initialize(env, config)
        # Total estimated reward of each arm
        self.data["TotalRew"] = np.zeros(self.env.dA, dtype=float)
        # Exp3 policy distribution
        self.data["Policy"] = self._total_rew_to_policy()

    def _total_rew_to_policy(self):
        return special.softmax(self.data["TotalRew"] * self.config.lr, axis=-1)

    def step(self):
        self.data["Policy"] = self._total_rew_to_policy()

        # Play a bandit
        policy = self.data["Policy"]
        dA = len(policy)
        act = np.random.choice(dA, p=policy)
        prob = policy[act]
        _, rew, _, _ = self.env.step(act)
        assert 0.0 <= rew <= 1.0, "Reward must be in range [0, 1]."

        # Update total reward data
        s = self.data["TotalRew"]
        s += 1
        s[act] -= (1 - rew) / prob
        self.data["TotalRew"] = s

        # Update regret
        regret = self.update_regret(act)
        return {"Regret": regret}
