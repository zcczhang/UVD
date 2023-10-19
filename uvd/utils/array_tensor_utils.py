import functools
from collections import defaultdict
from typing import List, Optional, Dict, cast, Any, DefaultDict
from typing import Union

import numpy as np
import torch
import tree
from PIL import Image

from .file_utils import f_expand


def any_transpose_first_two_axes(*arg):
    """Util to convert between (L, B, ...) and (B, L, ...)"""

    def _transpose(x):
        if isinstance(x, np.ndarray):
            return np.swapaxes(x, 0, 1)
        elif torch.is_tensor(x):
            return torch.swapaxes(x, 0, 1)
        else:
            raise ValueError(
                f"Input ({type(x)}) must be either a numpy array or a tensor."
            )

    return (_transpose(x) for x in arg)


def any_stack(xs: List, *, dim: int = 0):
    """Works for both torch Tensor and numpy array."""

    def _any_stack_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.stack(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.stack(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_stack_helper, *xs)


# ==== convert utils ====

_TORCH_DTYPE_TABLE = {
    torch.bool: 1,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int16: 2,
    torch.short: 2,
    torch.int32: 4,
    torch.int: 4,
    torch.int64: 8,
    torch.long: 8,
    torch.float16: 2,
    torch.half: 2,
    torch.float32: 4,
    torch.float: 4,
    torch.float64: 8,
    torch.double: 8,
}


def torch_dtype(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f'"{dtype}" is not a valid torch dtype')
        assert isinstance(
            dtype, torch.dtype
        ), f"dtype {dtype} is not a valid torch tensor type"
        return dtype
    else:
        raise NotImplementedError(f"{dtype} not supported")


def torch_device(device: Union[str, int, None]) -> Optional[torch.device]:
    """
    Args:
        device:
            - "auto": use current torch context device, same as `.to('cuda')`
            - int: negative for CPU, otherwise GPU index
    """
    if device is None:
        return None
    elif device == "auto":
        return torch.device("cuda")
    elif isinstance(device, int) and device < 0:
        return torch.device("cpu")
    else:
        return torch.device(device)


def torch_dtype_size(dtype: Union[str, torch.dtype]) -> int:
    return _TORCH_DTYPE_TABLE[torch_dtype(dtype)]


def _convert_then_transfer(x, dtype, device, copy, non_blocking):
    x = x.to(dtype=dtype, copy=copy, non_blocking=non_blocking)
    return x.to(device=device, copy=False, non_blocking=non_blocking)


def _transfer_then_convert(x, dtype, device, copy, non_blocking):
    x = x.to(device=device, copy=copy, non_blocking=non_blocking)
    return x.to(dtype=dtype, copy=False, non_blocking=non_blocking)


def any_to_torch_tensor(
    x,
    dtype: Union[str, torch.dtype, None] = None,
    device: Union[str, int, torch.device, None] = None,
    copy=False,
    non_blocking=False,
    smart_optimize: bool = True,
):
    dtype = torch_dtype(dtype)
    device = torch_device(device)

    if not isinstance(x, (torch.Tensor, np.ndarray)):
        # x is a primitive python sequence
        x = torch.tensor(x, dtype=dtype)
        copy = False

    # This step does not create any copy.
    # If x is a numpy array, simply wraps it in Tensor. If it's already a Tensor, do nothing.
    x = torch.as_tensor(x)
    # avoid passing None to .to(), PyTorch 1.4 bug
    dtype = dtype or x.dtype
    device = device or x.device

    if not smart_optimize:
        # do a single stage type conversion and transfer
        return x.to(dtype=dtype, device=device, copy=copy, non_blocking=non_blocking)

    # we have two choices: (1) convert dtype and then transfer to GPU
    # (2) transfer to GPU and then convert dtype
    # because CPU-to-GPU memory transfer is the bottleneck, we will reduce it as
    # much as possible by sending the smaller dtype

    src_dtype_size = torch_dtype_size(x.dtype)

    # destination dtype size
    if dtype is None:
        dest_dtype_size = src_dtype_size
    else:
        dest_dtype_size = torch_dtype_size(dtype)

    if x.dtype != dtype or x.device != device:
        # a copy will always be performed, no need to force copy again
        copy = False

    if src_dtype_size > dest_dtype_size:
        # better to do conversion on one device (e.g. CPU) and then transfer to another
        return _convert_then_transfer(x, dtype, device, copy, non_blocking)
    elif src_dtype_size == dest_dtype_size:
        # when equal, we prefer to do the conversion on whichever device that's GPU
        if x.device.type == "cuda":
            return _convert_then_transfer(x, dtype, device, copy, non_blocking)
        else:
            return _transfer_then_convert(x, dtype, device, copy, non_blocking)
    else:
        # better to transfer data across device first, and then do conversion
        return _transfer_then_convert(x, dtype, device, copy, non_blocking)


def any_to_numpy(
    x,
    dtype: Union[str, np.dtype, None] = None,
    copy: bool = False,
    non_blocking: bool = False,
    smart_optimize: bool = True,
    exclude_none: bool = False,
):
    if exclude_none and x is None:
        return x
    if isinstance(x, torch.Tensor):
        x = any_to_torch_tensor(
            x,
            dtype=dtype,
            device="cpu",
            copy=copy,
            non_blocking=non_blocking,
            smart_optimize=smart_optimize,
        )
        return x.detach().numpy()
    else:
        # primitive python sequence or ndarray
        return np.array(x, dtype=dtype, copy=copy)


def img_to_tensor(file_path: str, dtype=None, device=None, add_batch_dim: bool = False):
    """
    Args:
        scale_255: if True, scale to [0, 255]
        add_batch_dim: if 3D, add a leading batch dim

    Returns:
        tensor between [0, 255]

    """
    # image path
    pic = Image.open(f_expand(file_path)).convert("RGB")
    # code referenced from torchvision.transforms.functional.to_tensor
    # handle PIL Image
    assert pic.mode == "RGB"
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()

    img = any_to_torch_tensor(img, dtype=dtype, device=device)
    if add_batch_dim:
        img.unsqueeze_(dim=0)
    return img


def any_to_float(x, strict: bool = False):
    """Convert a singleton torch tensor or ndarray to float.

    Args:
        strict: True to check if the input is a singleton and raise Exception if not.
            False to return the original value if not a singleton
    """

    if torch.is_tensor(x) and x.numel() == 1:
        return float(x)
    elif isinstance(x, np.ndarray) and x.size == 1:
        return float(x)
    else:
        if strict:
            raise ValueError(f"{x} cannot be converted to a single float.")
        else:
            return x


def any_to_primitive(x):
    if isinstance(x, (np.ndarray, np.number, torch.Tensor)):
        return x.tolist()
    else:
        return x


def get_batch_size(x, strict: bool = False) -> int:
    """
    Args:
        x: can be any arbitrary nested structure of np array and torch tensor
        strict: True to check all batch sizes are the same
    """

    def _get_batch_size(x):
        if isinstance(x, np.ndarray):
            return x.shape[0]
        elif torch.is_tensor(x):
            return x.size(0)
        else:
            return len(x)

    xs = tree.flatten(x)

    if strict:
        batch_sizes = [_get_batch_size(x) for x in xs]
        assert all(
            b == batch_sizes[0] for b in batch_sizes
        ), f"batch sizes must all be the same in nested structure: {batch_sizes}"
        return batch_sizes[0]
    else:
        return _get_batch_size(xs[0])


def any_concat(xs: List, *, dim: int = 0):
    def _any_concat_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.concatenate(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.cat(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_concat_helper, *xs)


def make_recursive_func(fn, *, with_path=False):
    """Decorator that turns a function that works on a single array/tensor to
    working on arbitrary nested structures."""

    @functools.wraps(fn)
    def _wrapper(tensor_struct, *args, **kwargs):
        if with_path:
            return tree.map_structure_with_path(
                lambda paths, x: fn(paths, x, *args, **kwargs), tensor_struct
            )
        else:
            return tree.map_structure(lambda x: fn(x, *args, **kwargs), tensor_struct)

    return _wrapper


@make_recursive_func
def any_slice(x, slice):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return x[slice]
    else:
        return x


@make_recursive_func
def any_zeros_like(x: Union[Dict, np.ndarray, torch.Tensor, int, float, np.number]):
    """Returns a zero-filled object of the same (d)type and shape as the input.

    The difference between this and `np.zeros_like()` is that this works well
    with `np.number`, `int`, `float`, and `jax.numpy.DeviceArray` objects without
    converting them to `np.ndarray`s.
    Args:
      x: The object to replace with 0s.
    Returns:
      A zero-filed object of the same (d)type and shape as the input.
    """
    if isinstance(x, (int, float, np.number)):
        return type(x)(0)
    elif torch.is_tensor(x):
        return torch.zeros_like(x)
    elif isinstance(x, np.ndarray):
        return np.zeros_like(x)
    else:
        raise ValueError(
            f"Input ({type(x)}) must be either a numpy array, a tensor, an int, or a float."
        )


def batch_observations(
    observations: List[Dict],
    to_tensor: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[Dict, torch.Tensor]]:
    """Transpose a batch of observation dicts to a dict of batched
    observations.

    # Arguments

    observations :  List of dicts of observations.
    device : The torch.device to put the resulting tensors on.
        Will not move the tensors if None.

    # Returns

    Transposed dict of lists of observations.
    """

    def maybe_to_tensor(sensor_obs: Any):
        return (
            any_to_torch_tensor(sensor_obs, device=device) if to_tensor else sensor_obs
        )

    def dict_from_observation(
        observation: Dict[str, Any]
    ) -> Dict[str, Union[Dict, List]]:
        batch_dict: DefaultDict = defaultdict(list)

        for sensor in observation:
            if isinstance(observation[sensor], Dict):
                batch_dict[sensor] = dict_from_observation(observation[sensor])
            else:
                batch_dict[sensor].append(maybe_to_tensor(observation[sensor]))

        return batch_dict

    def fill_dict_from_observations(
        input_batch: Any, observation: Dict[str, Any]
    ) -> None:
        for sensor in observation:
            if isinstance(observation[sensor], Dict):
                fill_dict_from_observations(input_batch[sensor], observation[sensor])
            else:
                input_batch[sensor].append(maybe_to_tensor(observation[sensor]))

    def dict_to_batch(input_batch: Any) -> None:
        for sensor in input_batch:
            if isinstance(input_batch[sensor], Dict):
                dict_to_batch(input_batch[sensor])
            else:
                input_batch[sensor] = (
                    torch.stack(
                        [batch.to(device=device) for batch in input_batch[sensor]],
                        dim=0,
                    )
                    if to_tensor
                    else np.stack([_ for _ in input_batch[sensor]], axis=0)
                )

    if len(observations) == 0:
        return cast(Dict[str, Union[Dict, torch.Tensor]], observations)

    batch = dict_from_observation(observations[0])

    for obs in observations[1:]:
        fill_dict_from_observations(batch, obs)

    dict_to_batch(batch)

    return cast(Dict[str, Union[Dict, torch.Tensor, np.ndarray]], batch)


def any_is_float(x):
    if torch.is_tensor(x):
        return torch.is_floating_point(x)
    return isinstance(x, (np.floating, float))


def any_permute(x: Union[torch.Tensor, np.ndarray], order: tuple):
    if torch.is_tensor(x):
        return x.permute(*order)
    elif isinstance(x, np.ndarray):
        return x.transpose(order)


def any_to_chw(x):
    assert torch.is_tensor(x) or isinstance(x, np.ndarray), type(x)
    if x.ndim == 4 and x.shape[1] != 3:
        assert x.shape[-1] == 3, x.shape
        return any_permute(x, (0, 3, 1, 2))
    elif x.ndim == 3 and x.shape[0] != 3:
        assert x.shape[-1] == 3, x.shape
        return any_permute(x, (1, 2, 0))
    else:
        assert x.ndim in [3, 4], x.shape
        return x
