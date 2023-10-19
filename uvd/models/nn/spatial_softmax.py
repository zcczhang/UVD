from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


__all__ = ["SpatialSoftmax"]


class SpatialSoftmax(nn.Module):
    """Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn
    et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
        self,
        input_shape: list | tuple,
        num_kp: int | None = 32,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        output_variance: bool = False,
        noise_std: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            input_shape: shape of the input feature (C, H, W)
            num_kp: number of keypoints (None for not use spatialsoftmax)
            temperature: temperature term for the softmax.
            learnable_temperature: whether to learn the temperature
            output_variance: treat attention as a distribution, and compute second-order statistics to return
            noise_std: add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3, input_shape
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)
        self._in_c *= 2
        self.input_shape = (self._in_c, self._in_h, self._in_w)
        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=True
            )
            self.register_parameter("temperature", temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=False
            )
            self.register_buffer("temperature", temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        self.kps = None

    def __repr__(self):
        buffer_info = []
        for name, buffer in self._buffers.items():
            buffer_info.append(f"({name}): {buffer.shape}")
        buffer_str = "  " + "\n  ".join(buffer_info)
        return (
            f"{self.__class__.__name__}(\n"
            + buffer_str
            + f"\n  (nets): {self.nets}" * (self.nets is not None)
            + "\n)"
        )

    @property
    def output_dim(self):
        return [self._num_kp, 2]

    def forward(
        self,
        o: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        flatten: bool = True,
    ) -> torch.Tensor:
        """Forward pass through spatial softmax layer. For each keypoint, a 2D
        spatial probability distribution is created using a softmax, where the
        support is the pixel locations. This distribution is used to compute
        the expected value of the pixel location, which becomes a keypoint of
        dimension 2. K such keypoints are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        feature = o
        if c is not None:
            feature = torch.cat([feature, c], dim=1)
        assert (
            feature.shape[1:] == self.input_shape
        ), f"{feature.shape=} != {self.input_shape=}"
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(
                self.pos_x * self.pos_x * attention, dim=1, keepdim=True
            )
            expected_yy = torch.sum(
                self.pos_y * self.pos_y * attention, dim=1, keepdim=True
            )
            expected_xy = torch.sum(
                self.pos_x * self.pos_y * attention, dim=1, keepdim=True
            )
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(
                -1, self._num_kp, 2, 2
            )
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        if flatten:
            feature_keypoints = torch.flatten(feature_keypoints, 1)
        return feature_keypoints
