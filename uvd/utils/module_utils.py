from __future__ import annotations

from typing import Callable, Mapping, Any

import torch
import tree
from allenact.utils.system import get_logger
from torch import nn

from .file_utils import f_join


def freeze_module(module: nn.Module | torch.Tensor) -> nn.Module:
    if torch.is_tensor(module):
        module.requires_grad = False
        return module
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    return module


def unfreeze_module(module: nn.Module | torch.Tensor) -> nn.Module:
    if torch.is_tensor(module):
        module.requires_grad = True
        return module
    for param in module.parameters():
        param.requires_grad = True
    module.train()
    return module


def freeze_bn(module: nn.Module):
    for mod in module.modules():
        if "BatchNorm" in type(mod).__name__:
            mod.momentum = 0.0
            mod.eval()
    return module


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def bn_to_gn(
    module: nn.Module,
    group_ratio: int = 16,
    device: str | int | torch.device | None = None,
):
    return replace_submodules(
        root_module=module,
        predicate=lambda x: isinstance(x, (nn.BatchNorm2d, nn.BatchNorm1d)),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // group_ratio,
            num_channels=x.num_features,
            device=device,
        ),
    )


def torch_load(*fpath: str, map_location="cpu") -> dict:
    fpath = str(f_join(*fpath))
    return torch.load(fpath, map_location=map_location)


def tree_value_at_path(obj, paths: Tuple):
    try:
        for p in paths:
            obj = obj[p]
        return obj
    except Exception as e:
        raise ValueError(f"{e}\n\n-- Incorrect nested path {paths} for object: {obj}.")


def implements_method(object, method: str):
    """
    Returns:
        True if object implements a method
    """
    return hasattr(object, method) and callable(getattr(object, method))


def load_state_dict(
    objects,
    states,
    strip_prefix: str | None = None,
    strict: bool = True,
    filter_prefix: list[str] | str | None = None,
    verbose: bool = True,
):
    """
    Args:
        strict: objects and states must match exactly
        strip_prefix: only match the keys that have the prefix, and strip it
    """

    def _load(paths, obj):
        if not implements_method(obj, "load_state_dict"):
            raise ValueError(
                f"Object {type(obj)} does not support load_state_dict() method"
            )
        try:
            state = tree_value_at_path(states, paths)
        except ValueError:  # paths do not exist in `states` structure
            if strict:
                raise
            else:
                return
        if strip_prefix:
            assert isinstance(strip_prefix, str)
            state = {
                k[len(strip_prefix) :]: v
                for k, v in state.items()
                if k.startswith(strip_prefix)
            }
        if filter_prefix:
            state = {
                k: v
                for k, v in state.items()
                if all(
                    not k.startswith(p)
                    for p in (
                        (filter_prefix,)
                        if isinstance(filter_prefix, str)
                        else filter_prefix
                    )
                )
            }
        if isinstance(obj, nn.Module):
            return obj.load_state_dict(state, strict=strict)
        else:
            return obj.load_state_dict(state)

    keys = tree.map_structure_with_path(_load, objects)
    if not strict and verbose:
        if keys.missing_keys:
            get_logger().debug(
                f'Missing key(s) in state_dict: {", ".join(keys.missing_keys)}'
            )
        if keys.unexpected_keys:
            get_logger().debug(
                f'Unexpected key(s) in state dict: {", ".join(keys.unexpected_keys)}'
            )
    return keys


def load_pl_state_dict(
    model: nn.Module,
    states: Mapping[str:Any] | str,
    strip_prefix: str | None = None,
    filter_prefix: list[str] | str | None = None,
    strict: bool = True,
    verbose: bool = True,
    **kwargs,
):
    if isinstance(states, str):
        states = torch_load(states, **kwargs)
    if isinstance(strip_prefix, str) and strip_prefix[-1] != ".":
        strip_prefix += "."
    return load_state_dict(
        model,
        states.get("state_dict", states),
        strip_prefix=strip_prefix,
        strict=strict,
        filter_prefix=filter_prefix,
        verbose=verbose,
    )
