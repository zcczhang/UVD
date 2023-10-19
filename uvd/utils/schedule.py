import math
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineLRSchedule(LambdaLR):
    """Linear warmup and cosine (or constant) LR Schedule."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int = 0,
        warmup_start: float = 0.0,
        warmup_thresh: float = 1e-8,
        decay_method: Literal["cos", "linear"] = "cos",
        decay_steps: int,
        decay_start: float = 1.0,
        decay_end: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.warmup_start = warmup_start
        self.warmup_thresh = warmup_thresh
        self.decay_method = decay_method
        self.decay_steps = decay_steps
        self.decay_start = decay_start
        self.decay_end = decay_end
        super().__init__(optimizer, lr_lambda=self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        """lambda function for scheduling: lr_epoch (step) = lr_0 * lambda (step or epoch)"""
        if self.warmup_steps in [0, None] and self.decay_steps in [0, None]:
            return 1.0
        # warmup stage
        if (
            self.warmup_steps is not None
            and step <= self.warmup_steps
            and self.warmup_steps > 0
        ):
            return self.warmup_start + (1.0 - self.warmup_start) * max(
                self.warmup_thresh, float(step)
            ) / (float(max(1.0, self.warmup_steps)))
        # linear decay stage
        elif self.decay_steps is not None and self.decay_steps > 0:
            step = min(step, self.decay_steps)
            if self.decay_method == "cos":
                cos_decay = 0.5 * (
                    1
                    + math.cos(
                        math.pi
                        * (step - self.warmup_steps)
                        / (self.decay_steps - self.warmup_steps)
                    )
                )
                return self.decay_end + (self.decay_start - self.decay_end) * cos_decay
            elif self.decay_method == "linear":
                return max(
                    self.warmup_thresh,
                    self.decay_start
                    - (self.decay_end - self.decay_start)
                    * (
                        (self.warmup_steps - float(step))
                        / float(max(1.0, (self.decay_steps - self.warmup_steps)))
                    ),
                )
            else:
                raise NotImplementedError(self.decay_method)
        return 1.0
