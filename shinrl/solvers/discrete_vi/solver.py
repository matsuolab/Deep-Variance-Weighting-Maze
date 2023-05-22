"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from copy import deepcopy
from typing import List, Optional, Type
import functools

import gym
import jax
import jax.numpy as jnp
import chex
from chex import Array
import haiku as hk
import optax

import shinrl as srl

from ._build_calc_params_mixin import BuildCalcParamsDpMixIn, BuildCalcParamsRlMixIn
from ._build_net_act_mixin import BuildNetActMixIn
from ._build_net_mixin import BuildNetMixIn
from ._build_table_mixin import BuildTableMixIn
from ._target_mixin import DoubleQTargetMixIn, MunchausenTargetMixIn, QTargetMixIn
from .config import ViConfig

# ----- MixIns to execute one-step update -----


class TabularDpStepMixIn:
    def step(self):
        # Update Q table
        self.data["Q"] = self.target_tabular_dp(self.data)

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {}


class TabularRlStepMixIn:
    def step(self):
        # Collect samples
        samples = self.explore()

        # Update Q table
        q_targ = self.target_tabular_rl(self.data, samples)
        state, act = samples.state, samples.act  # B
        q = self.data["Q"]
        self.data["Q"] = srl.calc_ma(self.config.lr, state, act, q, q_targ)

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {}


class DeepDpStepMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        all_state, all_action = jnp.arange(self.env.dS), jnp.arange(self.env.dA)
        self.sa_state = jnp.repeat(all_state, self.env.dA, axis=0)  # dSdA
        self.sa_act = jnp.tile(all_action, self.env.dS)  # dSdA

        self.generate_nexts = self._build_generate_nexts(self.config.num_samples_target)
        self.generate_nexts_one = self._build_generate_nexts(1)
        if self.config.kl_coef == self.config.er_coef == 0:
            self.calc_q_target = self._build_q_target()
        else:
            self.calc_q_target = self._build_munchausen_target()

        self.uniform_weights, self.sigma_star_weights, self.calc_dvw_weights = self._build_calc_weight()
        if self.config.weight_mode == self.config.WEIGHTMODE.none:
            self.calc_weights = lambda *args: self.uniform_weights
        elif self.config.weight_mode == self.config.WEIGHTMODE.sigma_star:
            self.calc_weights = lambda *args: self.uniform_weights
        elif self.config.weight_mode == self.config.WEIGHTMODE.dvw:
            self.calc_weights = self.calc_dvw_weights

        # for update parameters
        if self.config.weight_mode == self.config.WEIGHTMODE.dvw:
            self.calc_var_target = self._build_var_target()
            self.calc_var_params = self._build_calc_var_params()
            self.calc_hypara_params = self._build_calc_hypara_params()
        self.calc_q_params = self._build_calc_q_params()
        self.multistep_calc_params = self._build_multistep_calc_params()

    # ===== Sample next states =====
    def _build_generate_nexts(self, num_nexts):
        def generate_nexts(key):
            """ generate next samples according to the transition matrix
            Returns:
                key
                next_states: self.config.num_samples_target x dSdA
                next_obs: self.config.num_samples_target x dSdA x dO
            """
            @jax.vmap
            def _generate_next(key, state, action):
                states, probs = self.env.transition(state, action)
                next_state = jax.random.choice(key, states, p=probs)
                next_obs = self.env.observation(next_state)
                return next_state, next_obs

            next_states, next_obss = [], []
            for _ in range(num_nexts):
                new_key, key = jax.random.split(key)
                keys = jax.random.split(new_key, self.dS * self.dA)
                next_state, next_obs = _generate_next(keys, self.sa_state, self.sa_act)
                next_states.append(next_state)
                next_obss.append(next_obs)
            return new_key, jnp.array(next_states), jnp.array(next_obss)
        return jax.jit(generate_nexts)

    # ===== Compute target values =====
    def _build_q_target(self):
        def target(next_obs_mats, q_targ_param):
            @jax.vmap
            def calc_next_v(next_obs_mat):
                next_q = self.q_net.apply(q_targ_param, next_obs_mat)
                return next_q.max(axis=-1, keepdims=True)

            next_v = calc_next_v(next_obs_mats).reshape(-1, self.dS, self.dA)
            rew_mat = self.env.mdp.rew_mat.reshape(1, self.dS, self.dA)
            q_targ = rew_mat + self.config.discount * next_v
            return q_targ
        return jax.jit(target)
 
    def _build_munchausen_target(self):
        def munchausen_target(next_obs_mats, q_targ_param):
            q = self.q_net.apply(q_targ_param, self.env.mdp.obs_mat)
            tau = self.config.kl_coef + self.config.er_coef
            alpha = self.config.kl_coef / tau
            log_pol = jax.nn.log_softmax(q / tau, axis=-1)  # (S, A)
            munchausen = alpha * jnp.clip(tau * log_pol, a_min=self.config.logp_clip)

            @jax.vmap
            def calc_next_v(next_obs_mat):
                next_q = self.q_net.apply(q_targ_param, next_obs_mat)
                next_pol = jax.nn.softmax(next_q / tau, axis=-1)
                next_log_pol = jax.nn.log_softmax(next_q / tau, axis=-1)
                return (next_pol * (next_q - tau * next_log_pol)).sum(axis=-1, keepdims=True)

            next_v = calc_next_v(next_obs_mats).reshape(-1, self.dS, self.dA)
            munchausen = munchausen.reshape(1, self.dS, self.dA)
            rew_mat = self.env.mdp.rew_mat.reshape(1, self.dS, self.dA)
            q_targ = munchausen + rew_mat + self.config.discount * next_v
            return q_targ
        return jax.jit(munchausen_target)

    def _build_var_target(self):
        def target(q_prev_targ: Array, q_targ_param):
            q_targ = self.q_net.apply(q_targ_param, self.env.mdp.obs_mat)
            q_targ = q_targ.reshape(1, self.dS, self.dA)
            # chex.assert_equal_rank((q_targ, q_prev_targ))
            var_targ = ((q_prev_targ - q_targ) ** 2).mean(axis=0)
            return var_targ
        return jax.jit(target)

    # ===== Compute Weight =====
    def _build_calc_weight(self):
        uniform_weight = jnp.ones((self.dS, self.dA))

        # sigma_star weight
        q = self.env.calc_optimal_q()
        horizon = 1 / (1 - self.config.discount)

        dS, dA = self.env.dS, self.env.dA
        tran_mat = self.env.mdp.tran_mat
        v = q.max(axis=-1, keepdims=True)  # S x 1
        Pv2 = srl.sp_mul(tran_mat, v ** 2, (dS * dA, dS)).reshape(dS, dA)
        Pv = srl.sp_mul(tran_mat, v, (dS * dA, dS)).reshape(dS, dA)
        sigma = Pv2 - (Pv) ** 2 + horizon
        sigma_star_weight = (horizon / sigma).reshape(self.dS, self.dA)

        def calc_weight(data):
            log_var = self.var_net.apply(data["LogVarNetFrozenParams"], self.env.mdp.obs_mat)
            var = jnp.exp(log_var)
            scaler = jnp.exp(data["HyparaParams"]["log_eta"])
            bottom = jnp.exp(data["HyparaParams"]["log_bottom"]) + self.config.weight_epsilon
            weights = scaler / (var + bottom)
            weights = jnp.maximum(weights, self.config.weight_min)
            return weights

        return uniform_weight, sigma_star_weight, jax.jit(calc_weight)

    # ===== For Variance update =====

    def _build_calc_var_params(self):
        def calc_var_loss(log_var_prm: hk.Params, var_targ: Array, obs: Array):
            log_pred = self.var_net.apply(log_var_prm, obs)
            var = jnp.exp(log_pred)
            # chex.assert_equal_shape((var, var_targ))
            loss = optax.huber_loss(var, var_targ)
            return loss.mean()

        def calc_params(data: srl.DataDict, var_targ: Array) -> Array:
            log_var_prm, opt_state = data["LogVarNetParams"], data["LogVarOptState"]
            mdp = self.env.mdp
            loss, grad = jax.value_and_grad(calc_var_loss)(log_var_prm, var_targ, mdp.obs_mat)
            updates, opt_state = self.var_opt.update(grad, opt_state, log_var_prm)
            log_var_prm = optax.apply_updates(log_var_prm, updates)
            return loss, log_var_prm, opt_state

        return jax.jit(calc_params)

    # ===== For weight hypara update =====

    def _build_calc_hypara_params(self):
        def calc_loss(hypara_prm: hk.Params, variance: Array):
            scaler = jnp.exp(hypara_prm["log_eta"])
            _bottom = jax.lax.stop_gradient(jnp.exp(hypara_prm["log_bottom"])) + self.config.weight_epsilon
            scaler_loss = ((scaler / (variance + _bottom)).mean() - 1.0) ** 2

            bottom = jnp.exp(hypara_prm["log_bottom"])
            bottom_loss = (jnp.sqrt(variance.max()) - bottom) ** 2
            return (scaler_loss + bottom_loss).mean()

        def calc_params(data: srl.DataDict, variance: Array) -> Array:
            hypara_prm, opt_state = data["HyparaParams"], data["HyparaOptState"]
            loss, grad = jax.value_and_grad(calc_loss)(hypara_prm, variance)
            updates, opt_state = self.hypara_opt.update(grad, opt_state, hypara_prm)
            hypara_prm = optax.apply_updates(hypara_prm, updates)
            return loss, hypara_prm, opt_state

        return jax.jit(calc_params)

    # ===== For Q update =====
 
    def _build_calc_q_params(self):
        def calc_q_loss(q_prm: hk.Params, q_targ: Array, obs: Array, weights: Array):
            pred = self.q_net.apply(q_prm, obs)
            # chex.assert_equal_shape((pred, q_targ, weights))
            loss = (pred - q_targ) ** 2
            return (loss * weights).mean()

        def calc_params(data: srl.DataDict, q_targ: Array, weights: Array) -> Array:
            q_prm, opt_state = data["QNetParams"], data["QOptState"]
            mdp = self.env.mdp
            loss, grad = jax.value_and_grad(calc_q_loss)(q_prm, q_targ, mdp.obs_mat, weights)
            updates, opt_state = self.q_opt.update(grad, opt_state, q_prm)
            q_prm = optax.apply_updates(q_prm, updates)
            return loss, q_prm, opt_state

        return jax.jit(calc_params)

    def _build_multistep_calc_params(self):
        # Do gradient descent by self.config.target_update_interval times
        def body_fun(_, val):
            loss, var_loss, hypara_loss, data, q_targ, var_targ, variance = val

            # train variance network
            if self.config.weight_mode == self.config.WEIGHTMODE.dvw:
                var_loss, var_prm, var_opt_state = self.calc_var_params(data, var_targ)
                hypara_loss, hypara_prm, hypara_opt_state = self.calc_hypara_params(data, variance)
                data.update(
                    {
                        "LogVarNetParams": var_prm,
                        "LogVarOptState": var_opt_state,
                        "HyparaParams": hypara_prm,
                        "HyparaOptState": hypara_opt_state,
                    }
                )
 
            # train q network
            weights = self.calc_weights(data)
            loss, q_prm, opt_state = self.calc_q_params(data, q_targ, weights)
            data.update(
                {
                    "QNetParams": q_prm,
                    "QOptState": opt_state,
                }
            )
            return (loss, var_loss, hypara_loss, data, q_targ, var_targ, variance)

        def multistep_calc_params(data, next_obs_mats):
            q_targ = self.calc_q_target(next_obs_mats, data["QNetTargParams"])  # num_samples S x A
            q_targ = q_targ.mean(axis=0)  # S x A
            var_targ, variance = 0, 0

            if self.config.weight_mode == self.config.WEIGHTMODE.dvw:
                prev_q_targ = self.calc_q_target(next_obs_mats, data["QNetPrevTargParams"])  # num_samples x S x A
                var_targ = self.calc_var_target(prev_q_targ, data["QNetTargParams"])
                log_var = self.var_net.apply(data["LogVarNetFrozenParams"], self.env.mdp.obs_mat)
                variance = jnp.exp(log_var)

            # chex.assert_equal_shape((q_targ, var_targ, variance))
            loss, var_loss, hypara_loss, data, _, _, _ = jax.lax.fori_loop(
                0, 
                self.config.target_update_interval, 
                body_fun, 
                (0, 0, 0, data, q_targ, var_targ, variance)
            )
            return loss, var_loss, hypara_loss, data
        return jax.jit(multistep_calc_params)


    def step(self):
        if self.config.weight_mode == self.config.WEIGHTMODE.dvw:
            self.data["LogVarNetFrozenParams"] = deepcopy(self.data["LogVarNetParams"])
            self.data["QNetPrevTargParams"] = deepcopy(self.data["QNetTargParams"])
        self.data["QNetTargParams"] = deepcopy(self.data["QNetParams"])

        self.key, _, next_obs_mats = self.generate_nexts(self.key)
        loss, var_loss, hypara_loss, self.data = self.multistep_calc_params(self.data, next_obs_mats)
        self.update_tb_data()

        res = {"Loss": loss.item(), "VarLoss": var_loss.item(), "HyparaLoss": hypara_loss.item()}

        # weight gap
        optimal_weight = self.sigma_star_weights / self.sigma_star_weights.mean()
        weight = self.calc_weights(self.data)
        weight = weight / weight.mean()

        weight_gap = jnp.mean(jnp.abs(optimal_weight - weight)).item()
        res.update({"WeightGap": weight_gap})
        return res


class DeepRlStepMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        self.buffer = srl.make_replay_buffer(self.env, self.config.buffer_size)
        self.calc_params = self._build_calc_params()
        self.uniform_weights, self.sigma_star_weights, self.calc_dvw_weights = self._build_calc_weight()
        if self.config.weight_mode == self.config.WEIGHTMODE.none:
            self.calc_weights = lambda data: self.uniform_weights
        elif self.config.weight_mode == self.config.WEIGHTMODE.sigma_star:
            self.calc_weights = lambda data: self.sigma_star_weights
        elif self.config.weight_mode == self.config.WEIGHTMODE.dvw:
            self.calc_weights = self.calc_dvw_weights
        else:
            raise ValueError


    # ===== Compute Weight =====
    def _build_calc_weight(self):
        uniform_weight = jnp.ones((self.dS, self.dA))

        # sigma_star weight
        q = self.env.calc_optimal_q()
        horizon = 1 / (1 - self.config.discount)

        dS, dA = self.env.dS, self.env.dA
        tran_mat = self.env.mdp.tran_mat
        v = q.max(axis=-1, keepdims=True)  # S x 1
        Pv2 = srl.sp_mul(tran_mat, v ** 2, (dS * dA, dS)).reshape(dS, dA)
        Pv = srl.sp_mul(tran_mat, v, (dS * dA, dS)).reshape(dS, dA)
        sigma = Pv2 - (Pv) ** 2 + horizon
        sigma_star_weight = (horizon / sigma).reshape(self.dS, self.dA)

        def calc_weight(data):
            log_var = self.var_net.apply(data["LogVarNetFrozenParams"], self.env.mdp.obs_mat)
            var = jnp.exp(log_var)
            scaler = jnp.exp(data["HyparaParams"]["log_eta"])
            bottom = jnp.exp(data["HyparaParams"]["log_bottom"]) + self.config.weight_epsilon
            weights = scaler / (var + bottom)
            weights = jnp.maximum(weights, self.config.weight_min)
            return weights

        return uniform_weight, sigma_star_weight, jax.jit(calc_weight)


    def _build_calc_params(self):
        def calc_var_loss(log_var_prm: hk.Params, var_targ: Array, obs: Array, act: Array):
            log_pred = self.var_net.apply(log_var_prm, obs)
            log_pred = jnp.take_along_axis(log_pred, act, axis=1)  # Bx1
            var = jnp.exp(log_pred)
            # chex.assert_equal_shape((var, var_targ))
            loss = optax.huber_loss(var, var_targ)
            return loss.mean()

        def calc_hypara_loss(hypara_prm: hk.Params, data: srl.DataDict, obs: Array):
            log_var = self.var_net.apply(data["LogVarNetFrozenParams"], obs)
            var = jnp.exp(log_var)
            scaler = jnp.exp(hypara_prm["log_eta"])
            _bottom = jax.lax.stop_gradient(jnp.exp(hypara_prm["log_bottom"])) + self.config.weight_epsilon
            scaler_loss = ((scaler / (var + _bottom)).mean() - 1.0) ** 2

            bottom = jnp.exp(hypara_prm["log_bottom"])
            bottom_loss = (jnp.sqrt(var.max()) - bottom) ** 2
            return (scaler_loss + bottom_loss).mean()

        def weighted_l2_loss(pred: Array, target: Array, weights: Array = None) -> float:
            chex.assert_equal_shape((pred, target))
            loss = optax.l2_loss(pred, target)
            return (loss * weights).mean()

        def calc_q_loss(q_prm: hk.Params, targ: Array, obs: Array, act: Array, weights: Array):
            pred = self.q_net.apply(q_prm, obs)
            pred = jnp.take_along_axis(pred, act, axis=1)  # Bx1
            chex.assert_equal_shape((pred, targ))
            return weighted_l2_loss(pred, targ, weights)

        def calc_params(data: srl.DataDict, samples: srl.Sample):
            act, obs = samples.act, samples.obs
            # update variance network
            var_prm, var_opt_state = data["LogVarNetParams"], data["LogVarOptState"]
            prev_data = {"QNetTargParams": data["QNetPrevTargParams"]}
            prev_q_targ = self.target_deep_rl(prev_data, samples)  # Bx1
            pred_q_targ = self.q_net.apply(data["QNetTargParams"], obs)
            pred_q_targ = jnp.take_along_axis(pred_q_targ, act, axis=1)  # Bx1
            var_targ = (pred_q_targ - prev_q_targ) ** 2
            var_loss, var_grad = jax.value_and_grad(calc_var_loss)(var_prm, var_targ, obs, act)
            var_updates, var_opt_state = self.var_opt.update(var_grad, var_opt_state, var_prm)
            var_prm = optax.apply_updates(var_prm, var_updates)

            # update hypara
            hypara_prm, hypara_opt_state = data["HyparaParams"], data["HyparaOptState"]
            hypara_loss, hypara_grad = jax.value_and_grad(calc_hypara_loss)(hypara_prm, data, obs)
            hypara_updates, hypara_opt_state = self.hypara_opt.update(hypara_grad, hypara_opt_state, hypara_prm)
            hypara_prm = optax.apply_updates(hypara_prm, hypara_updates)

            # update q network
            q_targ = self.target_deep_rl(data, samples)
            act, q_prm, opt_state = samples.act, data["QNetParams"], data["QOptState"]
            weights = self.calc_weights(data)
            weights = weights[samples.state, samples.act]
            q_loss, q_grad = jax.value_and_grad(calc_q_loss)(q_prm, q_targ, samples.obs, act, weights)
            updates, opt_state = self.q_opt.update(q_grad, opt_state, q_prm)
            q_prm = optax.apply_updates(q_prm, updates)
            return q_loss, q_prm, opt_state, var_loss, var_prm, var_opt_state, hypara_loss, hypara_prm, hypara_opt_state

        return jax.jit(calc_params)

    def step(self):
        # Collect samples
        self.explore(store_to_buffer=True)
        samples = srl.Sample(**self.buffer.sample(self.config.batch_size))

        # Compute new parameters
        (loss, q_prm, opt_state, 
         var_loss, var_prm, var_opt_state,
         hypara_loss, hypara_prm, hypara_opt_state) = self.calc_params(self.data, samples)

        # Update parameters
        self.data.update(
            {
                "QNetParams": q_prm,
                "QOptState": opt_state,
                "LogVarNetParams": var_prm,
                "LogVarOptState": var_opt_state,
                "HyparaParams": hypara_prm,
                "HyparaOptState": hypara_opt_state,
            }
        )
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.data["LogVarNetFrozenParams"] = deepcopy(self.data["LogVarNetParams"])
            self.data["QNetPrevTargParams"] = deepcopy(self.data["QNetTargParams"])
            self.data["QNetTargParams"] = deepcopy(self.data["QNetParams"])

        if self.is_shin_env:
            # Update ExplorePolicy & EvaluatePolicy tables
            self.update_tb_data()
        res = {"Loss": loss.item(), "VarLoss": var_loss.item(), "HyparaLoss": hypara_loss.item()}

        # weight gap
        optimal_weight = self.sigma_star_weights / self.sigma_star_weights.mean()
        weight = self.calc_weights(self.data)
        weight = weight / weight.mean()

        weight_gap = jnp.mean(jnp.abs(optimal_weight - weight)).item()
        res.update({"WeightGap": weight_gap})

        # variance gap
        dS, dA = self.dS, self.dA
        q = self.env.calc_optimal_q()
        tran_mat = self.env.mdp.tran_mat
        v = q.max(axis=-1, keepdims=True)  # S x 1
        Pv2 = srl.sp_mul(tran_mat, v ** 2, (dS * dA, dS)).reshape(dS, dA)
        Pv = srl.sp_mul(tran_mat, v, (dS * dA, dS)).reshape(dS, dA)
        oracle_variance = Pv2 - Pv ** 2
        variance_gap = jnp.mean(jnp.abs(oracle_variance - self.data["Var"])).item()
        res.update({"VarGap": variance_gap})
        return res


# --------------------------------------------------


def _is_shin_env(env: gym.Env) -> bool:
    """ Check if the env is ShinEnv or not """
    if isinstance(env, gym.Wrapper):
        is_shin_env = isinstance(env.unwrapped, srl.ShinEnv)
    else:
        is_shin_env = isinstance(env, srl.ShinEnv)
    return is_shin_env


class DiscreteViSolver(srl.BaseSolver):
    """Value iteration (VI) solver.

    This solver implements some basic VI-based algorithms.
    For example, DiscreteViSolver turns into DQN when approx == "nn" and explore != "oracle".
    """

    DefaultConfig = ViConfig

    @staticmethod
    def make_mixins(env: gym.Env, config: ViConfig) -> List[Type[object]]:
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
        is_q_learning = (config.er_coef == 0.0) * (config.kl_coef == 0.0)
        use_double_q = config.use_double_q
        if is_q_learning and not use_double_q:  # Vanilla Q target
            mixin_list.append(QTargetMixIn)
        elif is_q_learning and use_double_q:  # Double Q target
            mixin_list.append(DoubleQTargetMixIn)
        elif not is_q_learning and not use_double_q:  # Munchausen Q target
            mixin_list.append(MunchausenTargetMixIn)
        else:
            raise NotImplementedError

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

        mixin_list.append(DiscreteViSolver)
        return mixin_list
