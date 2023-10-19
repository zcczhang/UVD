from __future__ import annotations

import abc
import functools
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T

T.Resize = functools.partial(T.Resize, antialias=True)

from einops import rearrange
from torch.nn.modules.module import _addindent

import uvd.utils as U

__all__ = ["Preprocessor"]


class Preprocessor:
    def __init__(
        self,
        model_type: str | None = None,
        device: torch.device | None = None,
        remove_bn: bool = False,
        bn_to_gn: bool = False,
        remove_pool: bool = False,
        preprocess_with_fc: bool = False,
        save_fc: bool = False,
        use_language_goal: bool = False,
    ):
        super().__init__()
        device = "cpu" if device is None else device
        if isinstance(device, torch.device):
            if device.type == "cuda":
                device = f"{device.type}:{device.index}"
            else:
                device = "cpu"
        self.device = device
        assert not (remove_bn and bn_to_gn), "cannot be both true"
        self.remove_bn = remove_bn
        self.bn_to_gn = bn_to_gn
        self.model_type = model_type
        self.use_language_goal = use_language_goal
        self._output_dim = None

        self.remove_pool = remove_pool
        self.preprocess_with_fc = preprocess_with_fc and remove_pool
        self.save_fc = save_fc and not self.preprocess_with_fc
        self._fc: Optional[torch.nn.Linear] = None
        self._pool: Optional[torch.nn.Module] = None

        self._model: Optional[torch.nn.Module] = None
        self._transform: Optional[T] = None

    def to(self, device: torch.device, *args, **kwargs) -> "Preprocessor":
        self._model = self.model.to(device, *args, **kwargs)
        self.device = device
        return self

    @abc.abstractmethod
    def _get_model_and_transform(
        self, model_type: str | None
    ) -> tuple[torch.nn.Module, Optional[T]]:
        raise NotImplementedError

    @property
    def output_dim(self) -> tuple:
        if self._output_dim is None:
            dummy = self.process(np.random.randint(0, 255, size=(1, 1, 1, 3)))
            self._output_dim = (
                tuple(dummy.shape[1:]) if self.remove_pool else (dummy.shape[-1],)
            )
        return self._output_dim

    @property
    def model(self):
        if self._model is None:
            self._model, self._transform = self._get_model_and_transform(
                model_type=self.model_type
            )
            if self._transform is None:
                # by default
                self._transform = T.Compose(
                    [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
                )
            if self.remove_bn:
                for module in self._model.modules():
                    if "BatchNorm" in type(module).__name__:
                        module.momentum = 0.0
            elif self.bn_to_gn:
                self._model = U.bn_to_gn(self._model, device=self.device)
            self._model = U.freeze_module(self._model)
        return self._model

    @property
    def transform(self):
        if self._transform is None:
            if self._model is None:
                self._model = self.model
            else:
                self._transform = T.Compose(
                    [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
                )
        return self._transform

    def _encode_image(self, img_tensors: torch.Tensor) -> torch.FloatTensor:
        with torch.no_grad():
            return self.model(img_tensors)

    def transform_images(
        self, raw_img_batch: np.ndarray | torch.FloatTensor
    ) -> torch.FloatTensor:
        """Channel first, normalized, torch tensor."""
        assert raw_img_batch.ndim == 4, raw_img_batch.shape
        if torch.is_tensor(raw_img_batch) and torch.is_floating_point(raw_img_batch):
            return U.any_to_chw(raw_img_batch).to(self.device)
        # assert unnormalized
        assert not U.any_is_float(raw_img_batch), raw_img_batch.dtype

        raw_img_batch = U.any_to_chw(raw_img_batch)
        obs = self.transform(
            # optimize to transfer device first dtype next
            # as size uint8 is smaller than float32
            U.any_to_torch_tensor(
                raw_img_batch,
                dtype=torch.float32,
                device=self.device,
            )
            / 255.0
        )
        # L, 3, H, W
        assert obs.ndim == 4 and obs.shape[1] == 3, obs.shape
        return obs

    @property
    def preprocessor_fc(self) -> torch.nn.Linear | None:
        return self._fc if self.save_fc else None

    @property
    def fc_output_dim(self) -> tuple | None:
        if self.save_fc:
            assert self._fc is not None
            return (self._fc.out_features,)
        return None

    def process(
        self,
        raw_img_batch: np.ndarray | torch.FloatTensor,
        return_numpy: bool = False,
        reconstruct_linear: bool = False,
        return_linear_embed: bool = False,
    ) -> torch.Tensor | np.ndarray | tuple:
        """
        Args:
            raw_img_batch: raw L, H, W, 3 np array in range [0, 255]
            or pre-transformed when load data as L, 3, H, W in range [0, 1]
        """
        assert not (reconstruct_linear and return_linear_embed)
        if isinstance(raw_img_batch, np.ndarray):
            if np.issubdtype(raw_img_batch.dtype, np.integer):
                raw_img_batch = self.transform_images(raw_img_batch)
            else:
                raw_img_batch = U.any_to_chw(raw_img_batch)
                raw_img_batch = U.any_to_torch_tensor(raw_img_batch, device=self.device)
            obs = raw_img_batch
        else:
            assert torch.is_tensor(raw_img_batch), type(raw_img_batch)
            obs = self.transform_images(raw_img_batch)

        # L, 3, H, W
        assert obs.ndim == 4, f"{obs.shape}, {raw_img_batch.shape}"
        assert obs.shape[1] == 3, f"{obs.shape}, {raw_img_batch.shape}"
        embed = self._encode_image(obs)

        linear_embed = None
        if reconstruct_linear and self.remove_pool:
            assert embed.ndim != 2, embed.shape
            with torch.no_grad():
                embed = self._pool(embed)
                embed = torch.flatten(embed, 1)
                if self._fc is not None:
                    embed = self._fc(embed)
        elif self.preprocess_with_fc:
            if return_linear_embed:
                with torch.no_grad():
                    linear_embed = embed.detach().clone()
                    linear_embed = self._pool(linear_embed)
                    linear_embed = torch.flatten(linear_embed, 1)
                    linear_embed = self._fc(linear_embed)
            *_, h, w = embed.shape
            embed = rearrange(embed, "B D H W -> (B H W) D")
            with torch.no_grad():
                embed = self._fc(embed)
            embed = rearrange(embed, "(B H W) D -> B D H W", H=h, W=w)

        if reconstruct_linear:
            assert embed.ndim == 2, embed.shape

        if return_linear_embed:
            linear_embed = (
                embed.detach().clone() if linear_embed is None else linear_embed
            )
            assert linear_embed.ndim == 2, linear_embed.shape

        if return_numpy:
            embed = embed.cpu().numpy()
            return (embed, linear_embed.cpu().numpy()) if return_linear_embed else embed
        return (embed, linear_embed) if return_linear_embed else embed

    def encode_text(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> torch.Tensor | np.ndarray:
        return self.process(*args, **kwargs)

    def __repr__(self):
        return (
            f"(preprocessor): {self.__class__.__name__}(\n"
            f"  (model_type): {self.model_type}\n"
            f"  (transforms): {_addindent(repr(self.transform), 2)}\n"
            f"  (output_dim): {self.output_dim}\n"
            f")"
        )
