from __future__ import annotations

import abc
import math
from typing import Callable
from typing import Literal

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from allenact.base_abstractions.distributions import CategoricalDistr
from torch.distributions import Normal

import uvd.utils as U
from uvd.models import make_mlp_hidden, convert_activation

__all__ = [
    "DistributionBase",
    "DistributionHeadBase",
    "Deterministic",
    "DeterministicHead",
    "Gaussian",
    "GaussianHead",
    "DiagonalGaussian",
    "DiagonalGaussianHead",
    "TanhTransform",
    "SquashedGaussian",
    "SquashedGaussianHead",
    "Categorical",
    "CategoricalHead",
    "MultiCategorical",
    "MultiCategoricalHead",
]


class DistributionBase(torch.distributions.Distribution, abc.ABC):
    @abc.abstractmethod
    def log_prob(self, actions: torch.Tensor):
        raise NotImplementedError

    @abc.abstractproperty
    def mean(self) -> torch.Tensor:
        raise NotImplementedError

    def mode(self) -> torch.Tensor:
        return self.mean

    def imitation_loss(
        self,
        actions: torch.Tensor,
        loss_name: str = "mse",
        reduction: str | None = "mean",
        target_mask: torch.Tensor | None = None,  # B, T
        **kwargs,
    ) -> torch.Tensor:
        if target_mask is not None:
            if loss_name == "mse":
                loss = F.mse_loss(self.mean, actions, reduction="none")
            elif loss_name == "nll":
                loss = -self.log_prob(actions)
            else:
                raise NotImplementedError
            assert loss.ndim == 3, loss.shape
            assert loss.ndim == actions.ndim, (
                loss.shape,
                actions.shape,
            )  # B, T, act_dim
            if target_mask.ndim != loss.ndim:
                target_mask = target_mask[..., None]  # B, T, 1
            if reduction == "mean":
                return loss.sum() / target_mask.sum()
            elif reduction == "sum":
                return loss.sum()
            elif reduction is None:
                return loss
            else:
                raise NotImplementedError

        if loss_name == "mse":
            return F.mse_loss(self.mean, actions, reduction=reduction)
        elif loss_name == "nll":
            return -torch.mean(self.log_prob(actions))
        else:
            raise NotImplementedError(loss_name)


class DistributionHeadBase(nn.Module):
    def __init__(self, action_dim: int | list[int], **kwargs):
        self.action_dim = action_dim
        super().__init__()

    def extra_repr(self) -> str:
        return f"(action_dim): {self.action_dim}"


class Deterministic(DistributionBase):
    def __init__(self, value, **kwargs):
        super().__init__()
        self._value = value

    @property
    def mean(self):
        return self._value

    def mode(self):
        return self._value

    def sample(self, sample_shape=torch.Size()):
        return self._value

    def rsample(self, sample_shape=torch.Size()):
        return self._value

    def log_prob(self, actions: torch.Tensor):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


class DeterministicHead(DistributionHeadBase):
    def forward(self, x: torch.Tensor):
        return Deterministic(x)


class Gaussian(torch.distributions.Normal, DistributionBase):
    def mode(self) -> torch.Tensor:
        return super().mean


class GaussianHead(DistributionHeadBase):
    def __init__(
        self,
        action_dim: int,
        use_log_std: bool = False,
        min_action_std: float = 0.1,
        max_action_std: float = 0.8,
    ):
        super().__init__(action_dim)
        self.use_log_std = use_log_std
        std_init = min_action_std if self.use_log_std else 0.0
        self.std = nn.Parameter(
            torch.ones(action_dim, dtype=torch.float32) * std_init,
            requires_grad=True,
        )
        self.min_action_std = min_action_std
        self.max_action_std = max_action_std

    def forward(self, x: torch.Tensor) -> Gaussian:
        std = (
            self.std
            if self.use_log_std
            else (
                self.min_action_std
                + (self.max_action_std - self.min_action_std) * torch.sigmoid(self.std)
            )
        )
        if self.use_log_std:
            std = torch.exp(std)
        return Gaussian(loc=x, scale=std)


class DiagonalGaussian(torch.distributions.Normal, DistributionBase):
    def log_prob(self, actions):
        # assume independent action dims, so the probs are additive
        return super().log_prob(actions).sum(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return super().mean


class DiagonalGaussianHead(DistributionHeadBase):
    def __init__(self, action_dim: int, initial_log_std: float = 0.0):
        super().__init__(action_dim)
        self.log_std = torch.nn.Parameter(
            torch.ones((action_dim,)) * initial_log_std, requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(loc=x, scale=self.log_std.exp())


class TanhTransform(torch.distributions.transforms.Transform):
    bijective = True
    domain = torch.distributions.transforms.constraints.real
    codomain = torch.distributions.transforms.constraints.interval(-1.0, 1.0)

    def __init__(self, cache_size=1, eps: float = 0):
        super().__init__(cache_size=cache_size)
        self._eps = eps  # to avoid NaN at inverse (atanh)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    @property
    def sign(self):
        return +1

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        if self._eps:
            y = y.clamp(-1 + self._eps, 1 - self._eps)
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedGaussian(
    torch.distributions.transformed_distribution.TransformedDistribution,
    DistributionBase,
):
    def __init__(self, loc, scale, atanh_eps: float | None = None):
        """
        Args:
            atanh_eps: clip atanh(action between [-1+eps, 1-eps]). If the action is
                exactly -1 or exactly 1, its log_prob will be inf/NaN
        """
        self.loc = loc
        self.scale = scale

        self.base_dist = Normal(loc, scale)
        transforms = [TanhTransform(eps=atanh_eps)]
        super().__init__(self.base_dist, transforms)

    def log_prob(self, actions):
        # assume independent action dims, so the probs are additive
        return super().log_prob(actions).sum(-1)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        raise NotImplementedError(
            "no analytical form, entropy must be estimated from -log_prob.mean()"
        )


class SquashedGaussianHead(DistributionHeadBase):
    def __init__(
        self,
        action_dim: int,
        process_log_std: Literal["expln", "scale", "clip", "none", None] = "expln",
        log_std_bounds: tuple[float, float] = (-10, 2),
        atanh_eps: float = 1e-6,
    ):
        """Output dim should be action_dim*2, because it will be chunked into
        (mean, log_std)

        Args:
          process_log_std: different methods to process raw log_std value from NN output
              before sending to SquashedNormal
            - "expln": trick introduced by "State-Dependent Exploration for Policy
               Gradient Methods", Schmidhuber group: https://people.idsia.ch/~juergen//ecml2008rueckstiess.pdf
               also appears in stable-baselines-3:
               https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py
               This trick works out of box with PPO and doesn't need other hypers
            - "scale": first apply tanh() to log_std, and then scale to `log_std_bounds`
               used in some SAC implementations: https://github.com/denisyarats/drq
               WARNING: you may see NaN with this mode
            - "clip": simply clips log_std to `log_std_bounds`, used in the original SAC
               WARNING: you may see NaN with this mode
            - "none"/None: do nothing and directly pass log_std.exp() to SquashedNormal
               WARNING: you may see NaN with this mode
          atanh_eps: clip actions between [-1+eps, 1-eps]. If the action is
              exactly -1 or exactly 1, its log_prob will be inf/NaN
        """
        super().__init__(action_dim)

        assert process_log_std in ["expln", "scale", "clip", "none", None]
        self._process_log_std = process_log_std
        self._log_std_bounds = log_std_bounds
        self._atanh_eps = atanh_eps

    def forward(self, x: torch.Tensor) -> SquashedGaussian:
        mean, log_std = x.chunk(2, dim=-1)
        # rescale log_std inside [log_std_min, log_std_max]
        if self._process_log_std == "exp_ln":
            below_threshold = log_std.exp() * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + 1e-6
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        elif self._process_log_std == "scale":
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self._log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            std = log_std.exp()
        elif self._process_log_std == "clip":
            log_std = log_std.clip(*self._log_std_bounds)
            std = log_std.exp()
        else:
            std = log_std.exp()

        return SquashedGaussian(loc=mean, scale=std, atanh_eps=self._atanh_eps)


class Categorical(CategoricalDistr, DistributionBase):
    def imitation_loss(
        self,
        actions: torch.Tensor,
        loss_name: str = "mse",
        reduction: str | None = "mean",
        **kwargs,
    ) -> torch.Tensor:
        assert actions.dtype == torch.long
        assert loss_name.lower() in ["nll", "cross_entropy"], loss_name
        if self.logits.ndim == 3:
            assert actions.ndim == 2
            assert self.logits.shape[:2] == actions.shape
            return (
                F.cross_entropy(
                    self.logits.reshape(-1, self.logits.shape[-1]),
                    actions.reshape(-1),
                    reduction=reduction,
                )
                if loss_name == "cross_entropy"
                else -torch.mean(self.log_prob(actions.reshape(-1)))
            )
        return F.cross_entropy(self.logits, actions, reduction=reduction)


class CategoricalHead(DistributionHeadBase):
    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=x)


class MultiCategorical(DistributionBase):
    def __init__(self, logits: torch.Tensor, action_dim: list[int]):
        super().__init__()
        self._dists: list[CategoricalDistr] = [
            Categorical(logits=split)
            for split in torch.split(logits, action_dim, dim=-1)
        ]

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                dist.log_prob(action)
                for dist, action in zip(self._dists, torch.unbind(actions, dim=-1))
            ],
            dim=-1,
        ).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self._dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        assert sample_shape == torch.Size()
        return torch.stack([dist.sample() for dist in self._dists], dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        return torch.stack(
            [torch.argmax(dist.probs, dim=-1) for dist in self._dists], dim=-1
        )

    def imitation_loss(
        self,
        actions,
        loss_name: str = "nll",
        weights: list | None = None,
        reduction="mean",
        **kwargs,
    ):
        """
        Args:
            actions: groundtruth actions from expert
            loss_name: negative log-likelihood (NLL) or cross entropy_loss
            weights: weight the imitation loss from each component in MultiDiscrete
            reduction: "mean" or "none"

        Returns:
            one torch float
        """
        assert actions.dtype == torch.long, actions.dtype
        assert reduction in ["mean", "none"]
        if weights is None:
            weights = [1.0] * len(self._dists)
        else:
            assert len(weights) == len(self._dists)

        aggregate = sum if reduction == "mean" else list
        return aggregate(
            dist.imitation_loss(a, loss_name=loss_name, reduction=reduction) * w
            for dist, a, w in zip(self._dists, torch.unbind(actions, dim=1), weights)
        )


class MultiCategoricalHead(DistributionHeadBase):
    def __init__(self, action_dim: gym.spaces.MultiDiscrete | list[int]):
        if isinstance(action_dim, gym.spaces.MultiDiscrete):
            action_dim = action_dim.nvec.tolist()
        assert isinstance(action_dim, list), action_dim
        super().__init__(action_dim=action_dim)

    def forward(self, x: torch.Tensor) -> MultiCategorical:
        return MultiCategorical(x, action_dim=self.action_dim)


class GMMHead(DistributionHeadBase):
    def __init__(
        self,
        action_dim: int,
        num_gaussians: int,
        input_dim: int | None = None,
        hidden_dims: int | tuple | None = None,
        activation: str | Callable = nn.ReLU,
        dropouts: tuple | None = None,
    ):
        super().__init__(action_dim=action_dim)
        if input_dim is None:
            U.rank_zero_print(
                "input_dim set None for GMM, assume the same as action_dim"
            )
            input_dim = action_dim
        hidden_dims = hidden_dims or ()
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self._categorical_mlp = nn.Sequential(
            *make_mlp_hidden(
                convert_activation(activation),
                hidden_dims=(input_dim, *hidden_dims, num_gaussians),
                dropouts=dropouts,
            )
        )
        self._mu_mlp = nn.Sequential(
            *make_mlp_hidden(
                convert_activation(activation),
                hidden_dims=(input_dim, *hidden_dims, action_dim * num_gaussians),
                dropouts=dropouts,
            )
        )
        self._sigma_mlp = nn.Sequential(
            *make_mlp_hidden(
                convert_activation(activation),
                hidden_dims=(input_dim, *hidden_dims, action_dim * num_gaussians),
                dropouts=dropouts,
            )
        )
        self._mu_act = nn.Tanh()

        self._action_dim = action_dim
        self._num_gaussians = num_gaussians

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            x = x[:, None, :]  # broadcast the time dimension
        assert x.dim() == 3, x.dim()
        pi_logits = self._categorical_mlp(x)
        mean = self._mu_mlp(x)  # (B, T, num_gaussians * action_dim)
        mean = self._mu_act(mean)
        log_sigma = self._sigma_mlp(x)  # (B, T, num_gaussians * action_dim)
        mean = mean.reshape(*mean.shape[:-1], self._num_gaussians, self._action_dim)
        log_sigma = log_sigma.reshape(
            *log_sigma.shape[:-1], self._num_gaussians, self._action_dim
        )

        assert pi_logits.shape[-1] == self._num_gaussians
        assert mean.shape[-2:] == (self._num_gaussians, self._action_dim)
        assert log_sigma.shape[-2:] == (self._num_gaussians, self._action_dim)
        return MixtureOfGaussian(pi_logits, mean, log_sigma)


class MixtureOfGaussian(DistributionBase):
    def __init__(
        self,
        pi_logits: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ):
        super().__init__()
        """
        pi_logits: (B, T, num_gaussians)
        mu: (B, T, num_gaussians, dim)
        log_sigma: (B, T, num_gaussians, dim)
        """
        assert pi_logits.dim() == 3, pi_logits.dim()
        assert pi_logits.dim() + 1 == mu.dim() == log_sigma.dim()
        assert pi_logits.shape[-1] == mu.shape[-2] == log_sigma.shape[-2]
        assert mu.shape == log_sigma.shape

        mixture_distribution = torch.distributions.Categorical(logits=pi_logits)
        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, timesteps, num_modes) since MixtureSameFamily expects this shape
        component_distribution = torch.distributions.Normal(
            loc=mu, scale=F.softplus(log_sigma)
        )
        component_distribution = torch.distributions.Independent(
            component_distribution, 1
        )  # shift action dim to event shape
        self.dists = torch.distributions.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

    @property
    def mean(self):
        return self.dists.mean

    def mode(self):
        return self.mean

    def sample(self, sample_shape=torch.Size()):
        return self.dists.sample(sample_shape)

    def log_prob(self, actions: torch.Tensor):
        return self.dists.log_prob(actions)

    def imitation_loss(
        self,
        actions,
        loss_name: str = "nll",
        reduction="mean",
        target_mask=None,
        **kwargs,
    ):
        """-MLE/NLL loss."""
        # assert loss_name == "nll"
        if actions.shape != self.dists.mean.shape:
            actions = actions[:, None, ...]  # B, T, action_dim
        assert (
            actions.shape == self.dists.mean.shape
        ), f"{actions.shape} != {self.dists.mean.shape=}"
        if target_mask is not None:
            assert target_mask.shape == actions.shape[:-1]

        if loss_name == "mse":
            if target_mask is not None and reduction == "mean":
                return (
                    F.mse_loss(self.dists.mean, actions, reduction="sum")
                    * target_mask
                    / target_mask.sum()
                )
            assert target_mask is None
            return F.mse_loss(self.dists.mean, actions, reduction=reduction)

        assert loss_name == "nll", loss_name
        loss = -self.log_prob(actions)
        if target_mask is not None:
            loss *= target_mask
        if reduction == "mean":
            if target_mask is not None:
                return loss.sum() / target_mask.sum()
            else:
                return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
