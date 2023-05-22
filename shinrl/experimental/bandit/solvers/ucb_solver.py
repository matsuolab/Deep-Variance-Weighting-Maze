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

import shinrl as srl

from ._bandit_eval_mixin import BanditEvalMixIn


class UCB_TYPE(enum.IntEnum):
    vanilla = auto()
    asymptotic_optimal = auto()
    moss = auto()


@chex.dataclass
class UcbConfig(srl.SolverConfig):
    UCB_TYPE: ClassVar[Type[UCB_TYPE]] = UCB_TYPE
    ucb_type: UCB_TYPE = UCB_TYPE.vanilla


class UcbSolver(srl.BaseSolver):

    DefaultConfig = UcbConfig

    def make_mixins(env: gym.Env, config: UcbConfig) -> List[Type[object]]:
        if config.ucb_type == UCB_TYPE.vanilla:
            ucb_mixin = VanillaUcbMixIn
        elif config.ucb_type == UCB_TYPE.asymptotic_optimal:
            ucb_mixin = AsymptoticOptimalUcbMixIn
        elif config.ucb_type == UCB_TYPE.moss:
            ucb_mixin = MossUcbMixIn
        else:
            raise NotImplementedError
        mixin_list = [ucb_mixin, BanditEvalMixIn, UcbSolver]
        return mixin_list

    def initialize(self, env: gym.Env, config: Optional[UcbConfig] = None) -> None:
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
        ucb = self.calc_ucb()
        act = np.argmax(ucb)
        return act


class VanillaUcbMixIn:
    """UCB algorithm with confidence_level == 1 / (horizon^2)
    See Chapter 7 in https://tor-lattimore.com/downloads/book/book.pdf .
    """

    def calc_ucb(self):
        n_acts = self.data["ActNum"]
        rew_avg = self.data["RewAvg"]
        n = self.config.steps_per_epoch  # horizon
        ucb = np.where(n_acts == 0, np.infty, rew_avg + np.sqrt(4 * np.log(n) / n_acts))
        return ucb


class AsymptoticOptimalUcbMixIn:
    """UCB algorithm with confidence_level == 1 / (1 + t log^2(t))
    See Chapter 8 in https://tor-lattimore.com/downloads/book/book.pdf .
    """

    def calc_ucb(self):
        n_acts = self.data["ActNum"]
        rew_avg = self.data["RewAvg"]
        t = self.n_step + 1
        f_t = 1 + t * (np.log(t) ** 2)
        ucb = np.where(
            n_acts == 0, np.infty, rew_avg + np.sqrt(2 * np.log(f_t) / n_acts)
        )
        return ucb


class MossUcbMixIn:
    """UCB algorithm which has the minimax optimal regret bound.
    See Chapter 9 in https://tor-lattimore.com/downloads/book/book.pdf .
    """

    def calc_ucb(self):
        n_acts = self.data["ActNum"]
        rew_avg = self.data["RewAvg"]
        n = self.config.steps_per_epoch  # horizon
        k = self.env.dA  # number of arms
        log_plus = np.log(np.maximum(1, n / (k * n_acts)))
        ucb = np.where(n_acts == 0, np.infty, rew_avg + np.sqrt(4 * log_plus / n_acts))
        return ucb
