"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import List, Optional, Type

import chex
import gym
import numpy as np

import shinrl as srl

from ._bandit_eval_mixin import BanditEvalMixIn


@chex.dataclass
class ExploreThenCommitConfig(srl.SolverConfig):
    explore_round: int = 10


class ExploreThenCommitSolver(srl.BaseSolver):

    DefaultConfig = ExploreThenCommitConfig

    def make_mixins(env, config=None) -> List[Type[object]]:
        mixin_list = [BanditEvalMixIn, ExploreThenCommitSolver]
        return mixin_list

    def initialize(self, env, config=None) -> None:
        super().initialize(env, config)
        # Average reward received from arm i
        self.data["RewAvg"] = np.zeros(self.env.dA, dtype=float)
        # Number of times action i has been played
        self.data["ActNum"] = np.zeros(self.env.dA, dtype=int)

    def step(self):
        # Play a bandit
        act = self.act()
        _, rew, _, _ = self.env.step(act)

        # Update bandit data
        n = self.data["ActNum"][act]
        rew_avg = self.data["RewAvg"][act]
        rew_avg = rew_avg + (rew - rew_avg) / (n + 1)
        self.data["ActNum"][act] += 1
        self.data["RewAvg"][act] = rew_avg

        # Update regret
        regret = self.update_regret(act)
        return {"Regret": regret}

    def act(self):
        m = self.config.explore_round
        if self.n_step <= m * self.env.dA:
            act = self.n_step % self.env.dA
        else:
            act = np.argmax(self.data["RewAvg"])
        return act
