"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import enum
from enum import auto
from inspect import getmembers
from typing import ClassVar, Type

import chex
import jax
import optax

import shinrl as srl
from shinrl import SolverConfig


class EXPLORE(enum.IntEnum):
    oracle = auto()
    eps_greedy = auto()
    softmax = auto()


class EVALUATE(enum.IntEnum):
    softmax = auto()
    greedy = auto()


class APPROX(enum.IntEnum):
    tabular = auto()
    nn = auto()


class WEIGHTMODE(enum.IntEnum):
    none = auto()
    sigma_star = auto()
    dvw = auto()


ACTIVATION = enum.IntEnum(
    "ACTIVATION",
    {fnc[0]: enum.auto() for fnc in getmembers(jax.nn) if callable(fnc[1])},
)


OPTIMIZER = enum.IntEnum(
    "OPTIMIZER",
    {fnc[0]: enum.auto() for fnc in getmembers(optax._src.alias) if callable(fnc[1])},
)


LOSS = enum.IntEnum(
    "LOSS",
    {fnc[0]: enum.auto() for fnc in getmembers(srl._calc.loss) if callable(fnc[1])},
)


@chex.dataclass
class ViConfig(SolverConfig):
    """Config for ViSolver.

    Args:
        explore (EXPLORE):
            Type of the policy for exploration.
            The solver uses all the state-action pairs if 'oracle'.
        evaluate (EVALUATE): Type of the policy for evaluation.
        approx (APPROX): Type of the function approximation.

        eps_end (float): Epsilon value at the end of the eps-greedy exploration.
        eps_warmup (int): Epsilon value is set 1.0 until 'eps_warmup'.
        eps_decay (int): Interval to decrease the epsilon value.
        softmax_tmp (float): temperature parameter for softmax.

        lr (float): Learning rate.
        num_samples (int): Number of samples to collect when explore != 'oracle'.
        batch_size (int): Size of the minibatch.
        buffer_size (int): Replay buffer capacity.
        kl_coef (float): Coefficient for kl regularization.
        er_coef (float): Coefficient for entropy regularization.
        logp_clip (float): Minimum value of logp term.
        use_double_q (bool): Use double q trick or not.

        hidden (int): Size of the linear layer.
        depth (int): Depth of the linear layer.
        target_update_interval (int): Interval to update the target network.
        activation (str): Activation function.
        optimizer (str): Optimizer for nn.
        loss_fn (str): Loss function for critic.
    """

    # class variables
    EXPLORE: ClassVar[Type[EXPLORE]] = EXPLORE
    EVALUATE: ClassVar[Type[EVALUATE]] = EVALUATE
    APPROX: ClassVar[Type[APPROX]] = APPROX
    ACTIVATION: ClassVar[Type[ACTIVATION]] = ACTIVATION
    LOSS: ClassVar[Type[LOSS]] = LOSS
    OPTIMIZER: ClassVar[Type[OPTIMIZER]] = OPTIMIZER
    WEIGHTMODE: ClassVar[Type[WEIGHTMODE]] = WEIGHTMODE

    explore: EXPLORE = EXPLORE.oracle
    evaluate: EVALUATE = EVALUATE.greedy
    approx: APPROX = APPROX.tabular
    weight_mode: WEIGHTMODE = WEIGHTMODE.none

    # algorithm configs
    lr: float = 1e-3
    num_samples: int = 4
    buffer_size: int = int(1e6)
    batch_size: int = 32
    er_coef: float = 0.0
    kl_coef: float = 0.0
    logp_clip: float = -1e8
    use_double_q: bool = False

    # policy configs
    eps_end: float = 0.1
    eps_warmup: int = 0
    eps_decay: int = 10 ** 5
    softmax_tmp: float = 1.0

    # network configs
    hidden: int = 128
    depth: int = 2
    activation: ACTIVATION = ACTIVATION.relu
    optimizer: OPTIMIZER = OPTIMIZER.adam
    loss_fn: LOSS = LOSS.l2_loss

    # generative model setting
    num_samples_target: int = 1
    target_update_interval: int = 100

    # dvw setting
    weight_epsilon: float = 0.1
    weight_min: float = 0.1
    hypara_lr: float = 5e-3
    variance_lr: float = 1e-4
