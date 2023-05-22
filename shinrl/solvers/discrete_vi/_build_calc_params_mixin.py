"""MixIns to compute new params for nn-based algorithms.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Optional

import chex
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from chex import Array

import shinrl as srl

from .config import ViConfig


class BuildCalcParamsDpMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)

class BuildCalcParamsRlMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
