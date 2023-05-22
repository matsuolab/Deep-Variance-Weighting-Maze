"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

from typing import List, Optional, Type

import gym
import haiku as hk
import jax

import shinrl as srl

from ._build_calc_params_mixin import BuildCalcParamsDpMixIn, BuildCalcParamsRlMixIn
from ._build_net_act_mixin import BuildNetActMixIn
from ._build_net_mixin import BuildNetMixIn
from ._build_table_mixin import BuildTableMixIn
from ._target_mixin import QTargetMixIn
from .config import DdpgConfig

# ----- MixIns to execute one-step update -----


@jax.jit
def soft_target_update(
    params: hk.Params, targ_params: hk.Params, polyak: float = 0.005
) -> hk.Params:
    return jax.tree_multimap(
        lambda p, tp: (1 - polyak) * tp + polyak * p, params, targ_params
    )


class DeepDpStepMixIn:
    def step(self) -> None:
        # Compute new parameters
        pol_res, q_res = self.calc_params(self.data)
        pol_loss, pol_prm, pol_state = pol_res
        q_loss, q_prm, q_state = q_res

        # Update parameters
        self.data.update(
            {
                "PolNetParams": pol_prm,
                "PolOptState": pol_state,
                "QNetParams": q_prm,
                "QOptState": q_state,
            }
        )
        self.data["QNetTargParams"] = soft_target_update(
            self.data["QNetParams"],
            self.data["QNetTargParams"],
            self.config.polyak_rate,
        )
        self.data["PolNetTargParams"] = soft_target_update(
            self.data["PolNetParams"],
            self.data["PolNetTargParams"],
            self.config.polyak_rate,
        )

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {"PolLoss": pol_loss.item(), "QLoss": q_loss.item()}


class DeepRlStepMixIn:
    def initialize(self, env: gym.Env, config: Optional[DdpgConfig] = None) -> None:
        super().initialize(env, config)
        self.buffer = srl.make_replay_buffer(self.env, self.config.buffer_size)

    def step(self) -> None:
        # Collect samples
        samples = self.explore(store_to_buffer=True)
        samples = srl.Sample(**self.buffer.sample(self.config.batch_size))
        if self.buffer.get_stored_size() < self.config.replay_start_size:
            return {}

        # Compute new parameters
        pol_res, q_res = self.calc_params(self.data, samples)
        pol_loss, pol_prm, pol_state = pol_res
        q_loss, q_prm, q_state = q_res

        # Update parameters
        self.data.update(
            {
                "PolNetParams": pol_prm,
                "PolOptState": pol_state,
                "QNetParams": q_prm,
                "QOptState": q_state,
            }
        )
        self.data["QNetTargParams"] = soft_target_update(
            self.data["QNetParams"],
            self.data["QNetTargParams"],
            self.config.polyak_rate,
        )

        self.data["PolNetTargParams"] = soft_target_update(
            self.data["PolNetParams"],
            self.data["PolNetTargParams"],
            self.config.polyak_rate,
        )

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


class ContinuousDdpgSolver(srl.BaseSolver):
    """Deep Deterministic Policy Gradient (DDPG) solver.

    This solver implements variants of DDPG algorithm for a continuous action space.
    For example, ContinuousDdpgSolver with explore == "oracle" uses all the state and action pairs.
    """

    DefaultConfig = DdpgConfig

    @staticmethod
    def make_mixins(env: gym.Env, config: DdpgConfig) -> List[Type[object]]:
        mixin_list: List[Type[object]] = []
        is_shin_env = _is_shin_env(env)
        approx, explore = config.approx, config.explore
        APPROX, EXPLORE = config.APPROX, config.EXPLORE

        # Add step mixins for tabular DP, deep DP, tabular RL, or deep RL
        if approx == APPROX.nn and explore == EXPLORE.oracle:
            mixin_list.append(DeepDpStepMixIn)
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
        mixin_list += [QTargetMixIn]

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

        mixin_list.append(ContinuousDdpgSolver)
        return mixin_list
