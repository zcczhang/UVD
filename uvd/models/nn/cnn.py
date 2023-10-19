from typing import Optional, Tuple, Sequence, Union, List

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import uvd.utils as U
from uvd.models.nn.net_base import NetBase

__all__ = ["CNNCombiner", "make_cnn", "conv_output_dim"]


class CNNCombiner(NetBase):
    def __init__(
        self,
        *,
        input_shape: Union[np.ndarray, tuple],
        preprocessor_fc: Optional[nn.Linear] = None,
        goal_conditioned: bool = True,
        layer_channels: Sequence[int] = (128, 32),
        kernel_sizes: Sequence[Union[Tuple[int, int], int]] = ((1, 1), (1, 1)),
        layers_stride: Sequence[Union[Tuple[int, int], int]] = ((1, 1), (1, 1)),
        paddings: Sequence[Union[Tuple[int, int], int]] = ((0, 0), (0, 0)),
        dilations: Sequence[Union[Tuple[int, int], int]] = ((1, 1), (1, 1)),
        output_relu: bool = True,
        input_bn: bool = False
    ):
        super().__init__()
        assert len(input_shape) == 3, input_shape  # CHW
        self._cnn_layers_channels = list(layer_channels)
        self._cnn_layers_kernel_size = list(kernel_sizes)
        self._cnn_layers_stride = list(layers_stride)
        self._cnn_layers_paddings = list(paddings)
        self._cnn_layers_dilations = list(dilations)
        self._output_dim = None
        # x2 for goal conditioned
        n = 2 if goal_conditioned else 1
        self.model = self.setup_model(
            input_dims=input_shape[1:],
            input_channels=input_shape[0] * n
            if preprocessor_fc is None
            else preprocessor_fc.out_features * n,
            output_relu=output_relu,
        )
        if preprocessor_fc is not None:
            preprocessor_fc = U.freeze_module(preprocessor_fc)
        self.preprocessor_fc = preprocessor_fc
        self.input_bn = None
        if input_bn:
            self.input_bn = nn.BatchNorm2d(
                input_shape[0]
                if self.preprocessor_fc is not None
                else input_shape[0] * 2
            )

    @property
    def output_dim(self) -> tuple:
        # unflatten output dim of cnn
        return self._output_dim

    @staticmethod
    def _maybe_int2tuple(*args) -> List[Tuple[int, int]]:
        return [(_, _) if isinstance(_, int) else _ for _ in args]

    def setup_model(
        self,
        input_dims: np.ndarray,
        input_channels: Optional[int] = None,
        output_relu: bool = True,
    ) -> nn.Module:
        output_dims = input_dims
        for kernel_size, stride, padding, dilation in zip(
            self._cnn_layers_kernel_size,
            self._cnn_layers_stride,
            self._cnn_layers_paddings,
            self._cnn_layers_dilations,
        ):
            kernel_size, stride, padding, dilation = self._maybe_int2tuple(
                kernel_size, stride, padding, dilation
            )
            output_dims = conv_output_dim(
                dimension=output_dims,
                padding=np.array(padding, dtype=np.float32),
                dilation=np.array(dilation, dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )
            self._output_dim = (self._cnn_layers_channels[-1], *output_dims)
            return make_cnn(
                input_channels=input_channels,
                layer_channels=self._cnn_layers_channels,
                kernel_sizes=self._cnn_layers_kernel_size,
                strides=self._cnn_layers_stride,
                paddings=self._cnn_layers_paddings,
                dilations=self._cnn_layers_dilations,
                output_height=output_dims[0],
                output_width=output_dims[1],
                output_relu=output_relu,
            )

    def forward(
        self,
        o: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        flatten: bool = True,
    ) -> torch.Tensor:
        def _spatial_linear(x: torch.Tensor) -> torch.Tensor:
            if self.preprocessor_fc is None:
                return x
            *_, h, w = x.shape
            x = rearrange(x, "B D H W -> (B H W) D")
            with torch.no_grad():
                x = self.preprocessor_fc(x)  # (BHW) n
            x = rearrange(x, "(B H W) D -> B D H W", H=h, W=w)
            return x

        # B, 2048, 7, 7 for resnet50 with 224x224 img
        # to B, d=fc_out or 2048, 7, 7
        x = _spatial_linear(o)
        if c is not None:
            c = _spatial_linear(c)
            # B, d=2*d, 7, 7
            x = torch.cat([x, c], dim=1)

        if self.input_bn is not None:
            x = self.input_bn(x)

        x = self.model(x)
        if flatten:
            x = torch.flatten(x, 1)
        return x


def make_cnn(
    input_channels: Optional[int],
    layer_channels: Sequence[int],
    kernel_sizes: Sequence[Union[int, Tuple[int, int]]],
    strides: Sequence[Union[int, Tuple[int, int]]],
    paddings: Sequence[Union[int, Tuple[int, int]]],
    dilations: Sequence[Union[int, Tuple[int, int]]],
    output_height: int,
    output_width: int,
    output_channels: Optional[int] = None,
    flatten: bool = False,
    output_relu: bool = False,
) -> nn.Module:
    assert (
        len(layer_channels)
        == len(kernel_sizes)
        == len(strides)
        == len(paddings)
        == len(dilations)
    ), "Mismatched sizes: layers {} kernels {} strides {} paddings {} dilations {}".format(
        layer_channels, kernel_sizes, strides, paddings, dilations
    )

    net = nn.Sequential()

    input_channels_list = (
        [input_channels] if input_channels is not None else []
    ) + list(layer_channels)

    for it, current_channels in enumerate(layer_channels):
        net.add_module(
            "conv_{}".format(it),
            nn.Conv2d(
                in_channels=input_channels_list[it],
                out_channels=current_channels,
                kernel_size=kernel_sizes[it],
                stride=strides[it],
                padding=paddings[it],
                dilation=dilations[it],
            ),
        )
        if it < len(layer_channels) - 1:
            net.add_module("relu_{}".format(it), nn.ReLU(inplace=True))

    if flatten:
        assert output_channels is not None
        net.add_module("flatten", nn.Flatten())
        net.add_module(
            "fc",
            nn.Linear(
                layer_channels[-1] * output_width * output_height, output_channels
            ),
        )
    if output_relu:
        net.add_module("out_relu", nn.ReLU(True))

    return net


def conv_output_dim(
    dimension: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    kernel_size: Sequence[int],
    stride: Sequence[int],
) -> Tuple[int, ...]:
    """Calculates the output height and width based on the input height and
    width to the convolution layer. For parameter definitions see.

    [here](https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d).
    # Parameters
    dimension : See above link.
    padding : See above link.
    dilation : See above link.
    kernel_size : See above link.
    stride : See above link.
    """
    assert len(dimension) == 2
    out_dimension = []
    for i in range(len(dimension)):
        out_dimension.append(
            int(
                np.floor(
                    (
                        (
                            dimension[i]
                            + 2 * padding[i]
                            - dilation[i] * (kernel_size[i] - 1)
                            - 1
                        )
                        / stride[i]
                    )
                    + 1
                )
            )
        )
    return tuple(out_dimension)
