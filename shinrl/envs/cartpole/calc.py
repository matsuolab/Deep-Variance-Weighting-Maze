"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from chex import Array

from .config import CartPoleConfig


@jax.jit
def to_discrete_act(config: CartPoleConfig, c_act: float) -> int:
    """Convert a continuous action to a discrete action.

    Args:
        config (CartPoleConfig)
        c_act (float): Continuous action in range [-1, 1].

    Returns:
        A discretized action id.
    """
    chex.assert_type(c_act, float)
    dA = config.dA
    c_act = jnp.clip(c_act, -1.0, 1.0)
    c_step = 2 / dA
    act = jnp.floor((c_act + 1.0) / c_step + 1e-5).astype(jnp.uint32)
    return jnp.clip(act, 0, dA - 1)


@jax.jit
def to_continuous_act(config: CartPoleConfig, act: int) -> float:
    """Convert a discrete action to a continuous action.

    Args:
        config (CartPoleConfig)
        act (int): Discrete action in [0, ..., dA-1].

    Returns:
        A continuous action in range [-1.0, 1.0]
    """
    chex.assert_type(act, int)
    dA = config.dA
    c_step = 2 / dA
    c_act = act * c_step - 1.0
    return jnp.clip(c_act, -1.0, 1.0)


@jax.jit
def state_to_x_th(config: CartPoleConfig, state: int) -> Tuple[float, float]:
    """Convert a state id to x, x_dot, th, th_dot

    Args:
        config (CartPoleConfig)
        state (int)

    Returns:
        x, x_dot, th, th_dot
    """
    x_res, x_dot_res = config.x_res, config.x_dot_res
    th_res, th_dot_res = config.th_res, config.th_dot_res
    x_max, x_dot_max = config.x_max, config.x_dot_max
    th_max, th_dot_max = config.th_max, config.th_dot_max

    x_step = 2 * x_max / (x_res - 1)
    x_dot_step = 2 * x_dot_max / (x_dot_res - 1)
    th_step = 2 * th_max / (th_res - 1)
    th_dot_step = 2 * th_dot_max / (th_dot_res - 1)

    x_idx = state % x_res
    state = (state - x_idx) / x_res
    x_dot_idx = state % x_dot_res
    state = (state - x_dot_idx) / x_dot_res
    th_idx = state % th_res
    th_dot_idx = (state - th_idx) / th_res

    x = -x_max + x_step * x_idx
    x = jnp.clip(x, -x_max, x_max)
    x_dot = -x_dot_max + x_dot_step * x_dot_idx
    x_dot = jnp.clip(x_dot, -x_dot_max, x_dot_max)
    th = -th_max + th_step * th_idx
    th = jnp.clip(th, -th_max, th_max)
    th_dot = -th_dot_max + th_dot_step * th_dot_idx
    th_dot = jnp.clip(th_dot, -th_dot_max, th_dot_max)
    return x, x_dot, th, th_dot


@jax.jit
def x_th_to_state(
    config: CartPoleConfig, x: float, x_dot: float, th: float, th_dot
) -> float:
    """Convert x, x_dot, th, th_dot to state id

    Args:
        config (CartPoleConfig)

    Returns:
        state id (int)
    """
    x_res, x_dot_res = config.x_res, config.x_dot_res
    th_res, th_dot_res = config.th_res, config.th_dot_res
    x_max, x_dot_max = config.x_max, config.x_dot_max
    th_max, th_dot_max = config.th_max, config.th_dot_max

    x_step = 2 * x_max / (x_res - 1)
    x_dot_step = 2 * x_dot_max / (x_dot_res - 1)
    th_step = 2 * th_max / (th_res - 1)
    th_dot_step = 2 * th_dot_max / (th_dot_res - 1)

    x_idx = jnp.floor((x + x_max) / x_step + 1e-5)
    x_dot_idx = jnp.floor((x_dot + x_dot_max) / x_dot_step + 1e-5)
    th_idx = jnp.floor((th + th_max) / th_step + 1e-5)
    th_dot_idx = jnp.floor((th_dot + th_dot_max) / th_dot_step + 1e-5)
    state = x_idx + x_res * (x_dot_idx + x_dot_res * (th_idx + th_res * th_dot_idx))
    state = jnp.clip(state, 0, x_res * x_dot_res * th_res * th_dot_res - 1)
    return state.astype(jnp.uint32)


@jax.jit
def transition(config: CartPoleConfig, state: int, action: int) -> Tuple[Array, Array]:
    chex.assert_type([state, action], int)
    polemass_length = config.masspole * config.length
    total_mass = config.masspole + config.masscart
    c_act = to_continuous_act(config, action)
    force = jnp.squeeze(c_act) * config.force_mag

    def body_fn(_, x_th):
        x, x_dot, th, th_dot = x_th
        costh, sinth = jnp.cos(th), jnp.sin(th)
        temp = (force + polemass_length * th_dot ** 2 * sinth) / total_mass
        thetaacc = (config.gravity * sinth - costh * temp) / (
            config.length * (4.0 / 3.0 - config.masspole * costh ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costh / total_mass
        x = x + config.tau * x_dot
        x_dot = x_dot + config.tau * xacc
        th = th + config.tau * th_dot
        th_dot = th_dot + config.tau * thetaacc

        x = jnp.clip(x, -config.x_max, config.x_max)
        x_dot = jnp.clip(x_dot, -config.x_dot_max, config.x_dot_max)
        th = jnp.clip(th, -config.th_max, config.th_max)
        th_dot = jnp.clip(th_dot, -config.th_dot_max, config.th_dot_max)
        return (x, x_dot, th, th_dot)

    x, x_dot, th, th_dot = state_to_x_th(config, state)
    out = (jnp.abs(x) >= config.x_max) + (jnp.abs(th) >= config.th_max)
    # one step is not enough when state is discretized
    x, x_dot, th, th_dot = jax.lax.fori_loop(0, 1, body_fn, (x, x_dot, th, th_dot))
    next_state = x_th_to_state(config, x, x_dot, th, th_dot)
    next_state = jax.lax.cond(
        out, lambda _: state.astype(jnp.uint32), lambda _: next_state, None
    )
    next_state = next_state.reshape(-1)
    prob = jnp.array((1.0,), dtype=float)
    return next_state, prob


@jax.jit
def reward(config: CartPoleConfig, state: int, action: int) -> float:
    x, _, th, _ = state_to_x_th(config, state)
    out = (jnp.abs(x) >= config.x_max) + (jnp.abs(th) >= config.th_max)
    return jax.lax.cond(out, lambda _: 0.0, lambda _: 1.0, None)


@jax.jit
def observation_tuple(config: CartPoleConfig, state: int) -> Array:
    """Make the tuple observation."""
    x, x_dot, th, th_dot = state_to_x_th(config, state)
    return jnp.array([x, x_dot, th, th_dot], dtype=float)
