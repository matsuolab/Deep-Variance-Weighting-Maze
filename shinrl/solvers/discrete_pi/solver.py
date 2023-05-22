"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Type

import gym

import shinrl as srl

from ._build_calc_params_mixin import BuildCalcParamsDpMixIn, BuildCalcParamsRlMixIn
from ._build_net_act_mixin import BuildNetActMixIn
from ._build_net_mixin import BuildNetMixIn
from ._build_table_mixin import BuildTableMixIn
from ._target_mixin import QTargetMixIn, SoftQTargetMixIn
from .config import PiConfig

# ----- MixIns to execute one-step update -----


class TabularDpStepMixIn:
    def step(self):
        # Update Policy & Q tables
        self.data["LogPolicy"] = self.target_log_pol(self.data["Q"], self.data)
        self.data["Q"] = self.target_q_tabular_dp(self.data)

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {}


class TabularRlStepMixIn:
    def step(self):
        # Collect samples
        samples = self.explore()

        # Update Policy & Q tables
        self.data["LogPolicy"] = self.target_log_pol(self.data["Q"], self.data)
        q_targ = self.target_q_tabular_rl(self.data, samples)
        state, act = samples.state, samples.act  # B
        q = self.data["Q"]
        self.data["Q"] = srl.calc_ma(self.config.q_lr, state, act, q, q_targ)

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {}


class DeepDpStepMixIn:
    def step(self) -> None:
        # Compute new parameters
        pol_res, q_res = self.calc_params(self.data)
        pol_loss, pol_prm, pol_state = pol_res
        q_loss, q_prm, q_state = q_res

        # Update parameters
        self.data.update(
            {
                "LogPolNetParams": pol_prm,
                "LogPolOptState": pol_state,
                "QNetParams": q_prm,
                "QOptState": q_state,
            }
        )
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.data["QNetTargParams"] = deepcopy(self.data["QNetParams"])

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {"PolLoss": pol_loss.item(), "QLoss": q_loss.item()}


class DeepRlStepMixIn:
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)
        self.buffer = srl.make_replay_buffer(self.env, self.config.buffer_size)

    def step(self) -> None:
        # Collect samples
        samples = self.explore(store_to_buffer=True)
        samples = srl.Sample(**self.buffer.sample(self.config.batch_size))

        # Compute new parameters
        pol_res, q_res = self.calc_params(self.data, samples)
        pol_loss, pol_prm, pol_state = pol_res
        q_loss, q_prm, q_state = q_res

        # Update parameters
        self.data.update(
            {
                "LogPolNetParams": pol_prm,
                "LogPolOptState": pol_state,
                "QNetParams": q_prm,
                "QOptState": q_state,
            }
        )
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.data["QNetTargParams"] = deepcopy(self.data["QNetParams"])

        if self.is_shin_env:
            # Update ExplorePolicy & EvaluatePolicy tables
            self.update_tb_data()
        return {"PolLoss": pol_loss.item(), "QLoss": q_loss.item()}


# --------------------------------------------------


def _is_shin_env(env: gym.Env) -> bool:
    """ Check if the env is ShinEnv or not """
    if isinstance(env, gym.Wrapper):
        is_shin_env = isinstance(env.unwrapped, srl.ShinEnv)
    else:
        is_shin_env = isinstance(env, srl.ShinEnv)
    return is_shin_env


class DiscretePiSolver(srl.BaseSolver):
    """Policy iteration (PI) solver.

    This solver implements some basic PI-based algorithms for a discrete action space.
    For example, DiscretePiSolver turns into Discrete-SAC when approx == "nn", explore != "oracle", and er_coef != 0.
    """

    DefaultConfig = PiConfig

    @staticmethod
    def make_mixins(env: gym.Env, config: PiConfig) -> List[Type[object]]:
        mixin_list: List[Type[object]] = []
        is_shin_env = _is_shin_env(env)
        approx, explore = config.approx, config.explore
        APPROX, EXPLORE = config.APPROX, config.EXPLORE

        # Add step mixins for tabular DP, deep DP, tabular RL, or deep RL
        if approx == APPROX.tabular and explore == EXPLORE.oracle:
            mixin_list.append(TabularDpStepMixIn)
        elif approx == APPROX.nn and explore == EXPLORE.oracle:
            mixin_list.append(DeepDpStepMixIn)
        elif config.approx == APPROX.tabular and explore != EXPLORE.oracle:
            mixin_list.append(TabularRlStepMixIn)
        elif config.approx == APPROX.nn and config.explore != EXPLORE.oracle:
            mixin_list.append(DeepRlStepMixIn)
        else:
            raise NotImplementedError

        # Add mixins to compute new network parameters
        if approx == APPROX.nn and explore == EXPLORE.oracle:
            mixin_list.append(BuildCalcParamsDpMixIn)
        elif config.approx == APPROX.nn and config.explore != EXPLORE.oracle:
            mixin_list.append(BuildCalcParamsRlMixIn)

        # Add algorithm mixins to compute Q-targets
        is_q_learning = config.er_coef == 0.0
        if is_q_learning:  # Vanilla Q target
            mixin_list += [QTargetMixIn]
        else:  # Soft Q target
            mixin_list += [SoftQTargetMixIn]

        # Add mixins to build tables.
        if is_shin_env:
            mixin_list.append(BuildTableMixIn)

        # Add mixins to build networks.
        if approx == APPROX.nn:
            mixin_list.append(BuildNetActMixIn)
            mixin_list.append(BuildNetMixIn)

        # Add mixins for evaluation and exploration.
        if is_shin_env:
            mixin_list.append(srl.BaseShinEvalMixIn)
            mixin_list.append(srl.BaseShinExploreMixIn)
        else:
            mixin_list.append(srl.BaseGymEvalMixIn)
            mixin_list.append(srl.BaseGymExploreMixIn)

        mixin_list.append(DiscretePiSolver)
        return mixin_list
