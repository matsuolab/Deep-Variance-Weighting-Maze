""" MixIn to prepare networks.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from copy import deepcopy
from typing import Optional

import gym
import jax
import jax.numpy as jnp
import optax

import shinrl as srl

from .config import ViConfig


class BuildNetMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)

        self.q_net = self._build_net()
        self.q_opt = getattr(optax, self.config.optimizer.name)(learning_rate=self.config.lr)
        net_param, opt_state = self._build_net_data(self.q_net, self.q_opt)
        self.data["QNetParams"] = net_param
        self.data["QNetTargParams"] = deepcopy(net_param)
        self.data["QNetPrevTargParams"] = deepcopy(net_param)
        self.data["QOptState"] = opt_state

        self.var_net = self._build_net()
        self.var_opt = getattr(optax, self.config.optimizer.name)(learning_rate=self.config.variance_lr)
        log_var_net_param, log_var_opt_state = self._build_net_data(self.var_net, self.var_opt)
        self.data["LogVarNetParams"] = log_var_net_param
        self.data["LogVarNetFrozenParams"] = deepcopy(log_var_net_param)
        self.data["LogVarOptState"] = log_var_opt_state

        optimizer = getattr(optax, self.config.optimizer.name)
        self.hypara_opt = optimizer(learning_rate=self.config.hypara_lr)
        hypara_params = {"log_eta": jnp.ones((1,)), "log_bottom": jnp.ones((1,))}
        hypara_opt_state = self.hypara_opt.init(hypara_params)
        self.data["HyparaParams"] = hypara_params
        self.data["HyparaOptState"] = hypara_opt_state

    def _build_net(self):
        n_out = self.env.action_space.n
        depth, hidden = self.config.depth, self.config.hidden
        act_layer = getattr(jax.nn, self.config.activation.name)
        if len(self.env.observation_space.shape) == 1:
            net = srl.build_obs_forward_fc(n_out, depth, hidden, act_layer)
        else:
            net = srl.build_obs_forward_conv(n_out, depth, hidden, act_layer)
        return net

    def _build_net_data(self, net, opt):
        self.key, key = jax.random.split(self.key)
        sample_init = jnp.ones([1, *self.env.observation_space.shape])
        net_param = net.init(key, sample_init)
        # init last layer as zero
        b = net_param[list(net_param.keys())[-1]]["b"]
        w = net_param[list(net_param.keys())[-1]]["w"]
        net_param[list(net_param.keys())[-1]]["b"] = jnp.zeros_like(b)
        net_param[list(net_param.keys())[-1]]["w"] = jnp.zeros_like(w)
        opt_state = opt.init(net_param)
        return net_param, opt_state
