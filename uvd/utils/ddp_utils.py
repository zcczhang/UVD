from __future__ import annotations

import copy
import inspect
import os
import sys
from collections import Counter
from typing import Sequence, Any

import numpy as np
import torch
from allenact.utils.system import get_logger
from omegaconf import ListConfig, DictConfig
from termcolor import colored
from torch import nn


def partition_inds(n: int, num_parts: int):
    return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(np.int32)


# @rank_zero_only
def rank_zero_print(*args, use_logger: bool = False, **kwargs):
    if not is_rank_zero():
        return
    # when using ddp, only print with rank 0 process
    if "color" in kwargs.keys():
        color = kwargs.pop("color")
        text = "".join([str(t) + " " for t in args])[:-1]
        print(colored(text, color, **kwargs))
    else:
        if use_logger:
            get_logger().debug("".join([str(t) + " " for t in args])[:-1], **kwargs)
        else:
            print(*args, **kwargs)


def get_local_rank() -> int:
    rank = int(os.environ.get("LOCAL_RANK", -1))
    return rank if rank > 0 else 0


def is_rank_zero() -> bool:
    return get_local_rank() == 0


def get_rank_zero_device() -> int:
    return int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])


# @rank_zero_only
def debug_model_info(
    model: nn.Module, trainable: bool = True, use_logger: bool = False, **kwargs
):
    if not is_rank_zero():
        return
    debug_msg = (
        f"frozen preprocessor: \n" * (not trainable)
        + f"{model}"
        + (
            f"\nTrainable Parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        * trainable
    )
    rank_zero_print(debug_msg, use_logger=use_logger, **kwargs)


def debug_batch_info(
    x: Any,
    indent: int = 2,
    title: str | None = "Batch:",
    also_print: bool = True,
    print_once: bool = False,
    **kwargs,
) -> str:
    def _debug_batch_info(x: Any, indent: int = 2, title: str | None = "Batch:"):
        output = []
        if title is not None:
            output.append(title)

        if not isinstance(x, dict):
            if isinstance(x, np.ndarray) or torch.is_tensor(x):
                output.append(f"{' ' * indent}shape={list(x.shape)} | dtype={x.dtype}")
            elif isinstance(x, Sequence):
                output.extend(
                    (
                        f"{' ' * indent}shape={list(v.shape)} | dtype={v.dtype}"
                        for v in x
                        if isinstance(v, np.ndarray) or torch.is_tensor(v)
                    )
                )
            else:
                output.append(f"{' ' * indent}: {x.__class__.__name__}")
        else:
            for k, v in x.items():
                if isinstance(v, np.ndarray) or torch.is_tensor(v):
                    output.append(
                        f"{' ' * indent}{k}: shape={list(v.shape)} | dtype={v.dtype}"
                    )
                elif isinstance(v, dict):
                    output.append(f"{' ' * indent}{k}:")
                    output.append(_debug_batch_info(v, indent=indent + 2, title=None))
                elif isinstance(v, Sequence):
                    output.append(f"{' ' * indent}{k}: {v.__class__.__name__}")
                    output.append(_debug_batch_info(v, indent=indent + 2, title=None))
                else:
                    output.append(f"{' ' * indent}{k}: {v.__class__.__name__}")

        return "\n".join(output)

    batch_info = _debug_batch_info(x, indent, title)
    if also_print:
        if print_once:
            rank_zero_print_once(batch_info, "\n", **kwargs)
        else:
            rank_zero_print(batch_info, "\n", **kwargs)
    return batch_info


def parse_gpu_devices(devices: list[int] | ListConfig[int] | str | int | None):
    if isinstance(devices, (list, ListConfig)):
        devices = list(devices)
        num_gpus = len(devices)
    elif devices in [-1, "-1", "auto"] or devices is None:
        num_gpus = torch.cuda.device_count()
        devices = list(range(num_gpus))
    elif isinstance(devices, str):
        devices = devices.strip()
        assert "[" in devices and "]" in devices, devices
        devices = devices[1:-1]
        devices = [int(i) for i in devices.split(",")]
        num_gpus = len(devices)
    else:
        devices = list(range(devices))
        num_gpus = len(devices)
    return devices, num_gpus


def caller_name(skip: int = 0):
    """Https://gist.github.com/techtonik/2151727#gistcomment-2333747."""

    def stack_(frame):
        framelist = []
        while frame:
            framelist.append(frame)
            frame = frame.f_back
        return framelist

    stack = stack_(sys._getframe(1))  # type: ignore
    start = 0 + skip
    if len(stack) < start + 1:
        return ""
    parentframe = stack[start]

    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    if module:
        name.append(module.__name__)
    # detect classname
    if "self" in parentframe.f_locals:
        name.append(parentframe.f_locals["self"].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != "<module>":  # top level usually
        name.append(codename)  # function or a method
    del parentframe
    return ".".join(name)


_GLOBAL_ONCE_SET = set()
_GLOBAL_NTIMES_COUNTER = Counter()


def global_once(name=None, skip=0):
    if name is None:
        name = caller_name(skip=skip + 1)
    if name in _GLOBAL_ONCE_SET:
        return False
    else:
        _GLOBAL_ONCE_SET.add(name)
        return True


def global_n_times(name, n: int):
    """Triggers N times."""
    assert n >= 1
    if _GLOBAL_NTIMES_COUNTER[name] < n:
        _GLOBAL_NTIMES_COUNTER[name] += 1
        return True
    else:
        return False


def rank_zero_print_once(*args, name=None, skip=0, **kwargs):
    if not is_rank_zero():
        return
    if global_once(name, skip=skip + 2):
        rank_zero_print(*args, **kwargs)


class Once:
    def __init__(self):
        self._triggered = False

    def __call__(self):
        if not self._triggered:
            self._triggered = True
            return True
        else:
            return False

    def __bool__(self):
        raise RuntimeError("`Once` objects should be used by calling ()")


def pl_debug(cfg: DictConfig):
    assert cfg.debug
    cfg = copy.deepcopy(cfg)
    _, num_gpus = parse_gpu_devices(cfg.gpus)
    cfg.exp_output_dir += "_debug"
    cfg.train.batch_size = 2 * num_gpus
    cfg.train.num_epochs = 1
    cfg.eval_every_n_epoch = 1
    cfg.evaluate.enable = True
    cfg.evaluate.num_processes = 2
    cfg.evaluate.evaluator.max_horizon = 10
    cfg.logging.wandb_kwargs.project = "test"
    cfg.logging.log_tb = False
    return cfg
