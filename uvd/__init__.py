from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from .decomp import *
from .models import *


def get_uvd_subgoals(
    frames: np.ndarray | str,
    preprocessor_name: Literal["vip", "r3m", "liv", "clip", "vc1", "dinov2"] = "vip",
    device: torch.device | str | None = "cuda",
    return_indices: bool = False,
) -> list | np.ndarray:
    """Quick API for UVD decomposition."""
    if isinstance(frames, str):
        from decord import VideoReader

        vr = VideoReader(frames, height=224, width=224)
        frames = vr[:].asnumpy()
    preprocessor = get_preprocessor(preprocessor_name, device=device)
    rep = preprocessor.process(frames, return_numpy=True)
    _, decomp_meta = decomp_trajectories("embed", rep)
    indices = decomp_meta.milestone_indices
    if return_indices:
        return indices
    return frames[indices]
