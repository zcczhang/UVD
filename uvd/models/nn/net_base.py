from __future__ import annotations

import abc

from torch import nn


class NetBase(nn.Module):
    @abc.abstractproperty
    def output_dim(self) -> tuple | int:
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    def extra_repr(self) -> str:
        return f"(output_dim): {self.output_dim}"
