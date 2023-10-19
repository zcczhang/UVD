from __future__ import annotations

import random
from typing import Literal, NamedTuple, Callable

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from scipy.signal import medfilt
from scipy.signal import savgol_filter, argrelextrema

import uvd.utils as U
from uvd.decomp.kernel_reg import KernelRegression


def linear_random_skip(
    cur_step: int, next_goal_step: int, ratio: float = 0.0, progress_lower: float = 0.3
) -> bool:
    """Progress toward next goal exceed `progress_lower`, linearly increasing
    temperature for P(skip) <= ratio."""
    if ratio == 0.0 or progress_lower == 1:
        return False
    assert cur_step <= next_goal_step, f"{cur_step} > {next_goal_step}"
    # linearly increase until ratio
    progress = cur_step / next_goal_step
    temp = (
        1.0
        if progress < progress_lower
        else ((progress - progress_lower) / (1 - progress_lower)) * ratio
    )
    if 0 < temp < random.random():
        return True
    return False


class DecompMeta(NamedTuple):
    milestone_indices: list
    milestone_starts: list | None = None
    iter_curves: list[np.ndarray] | None = None


def _debug_plt(
    xs: np.ndarray,
    embed_distances: np.ndarray,
    starts: np.ndarray | list,
    ends: np.ndarray | list,
    return_numpy: bool = False,
):
    fig = plt.figure()
    for s in starts:
        plt.axvline(x=s, linestyle="--")
    for e in ends:
        plt.axvline(x=e, linestyle="dotted")
    plt.plot(xs, np.gradient(embed_distances), linewidth=1.5, label="1st derivative")
    plt.plot(
        xs,
        np.gradient(np.gradient(embed_distances)),
        linewidth=1.5,
        label="2nd derivative",
    )
    plt.plot(xs, embed_distances, linewidth=1.5, label="embedding distance")
    plt.legend()
    if return_numpy:
        # fig.canvas.draw()
        return U.plt_to_numpy(fig)
    else:
        plt.show()


def embed_decomp_no_robot(
    embeddings: np.ndarray | torch.Tensor,
    no_robot_embeddings: np.ndarray | torch.Tensor,
    window_length: int = 10,
    derivative_order: int = 1,
    derivative_threshold: float = 1e-3,
    threshold_subgoal_passing: float | None = None,
    force_interleave: bool = False,
    debug_plt: bool = False,
    debug_plt_to_wandb: bool = False,
    task_name: str | None = None,
    fill_embeddings: bool = True,
):
    debug_plt = debug_plt and U.is_rank_zero()
    debug_plt_to_wandb = debug_plt and debug_plt_to_wandb
    if threshold_subgoal_passing is not None:
        assert 0 < threshold_subgoal_passing <= 1.0, threshold_subgoal_passing
    if isinstance(embeddings, torch.Tensor):
        device = embeddings.device
    else:
        device = None
    # clip based preprocessors would have bf16 by default
    embeddings = U.any_to_numpy(embeddings, dtype="float32")
    no_robot_embeddings = U.any_to_numpy(no_robot_embeddings, dtype="float32")
    # L, N (though can be rgb for debugging as well)
    traj_length = embeddings.shape[0]

    debug_plt_figs = []
    if fill_embeddings:
        milestone_embeddings = []
    else:
        milestone_embeddings = None
    # milestone indices, i.e. end of obj changing
    subgoal_indices = []
    # indices when obj changing starts, i.e. milestone indices for hand reaching
    subgoal_starts = []
    cur_subgoal_idx = traj_length - 1
    # back to start
    while cur_subgoal_idx > 15:
        unnormalized_embed_distance = np.linalg.norm(
            no_robot_embeddings[: cur_subgoal_idx + 1]
            - no_robot_embeddings[cur_subgoal_idx],
            axis=1,
        )
        cur_embed_distance = unnormalized_embed_distance / np.linalg.norm(
            no_robot_embeddings[0] - no_robot_embeddings[cur_subgoal_idx]
        )
        cur_embed_distance = medfilt(cur_embed_distance, kernel_size=None)  # smooth

        if derivative_order == 1:
            slope = np.gradient(cur_embed_distance)
        elif derivative_order == 2:
            slope = np.gradient(np.gradient(cur_embed_distance))
        else:
            raise NotImplementedError(derivative_order)
        valid_slope_indices = np.where(np.abs(slope) >= derivative_threshold)[0]
        # Find the differences between consecutive valid_slope_indices
        diffs = np.diff(valid_slope_indices)
        # Find indices where the difference is greater than window_length + 1
        break_indices = np.where(diffs > window_length + 1)[0]
        # Extract start and end positions
        start_positions = valid_slope_indices[np.concatenate([[0], break_indices + 1])]
        end_positions = valid_slope_indices[
            np.concatenate((break_indices, [len(valid_slope_indices) - 1]))
        ]

        # small tolerance
        start_positions = [max(0, p - 1) for p in start_positions]
        end_positions = [min(traj_length - 1, p + 1) for p in end_positions]

        if debug_plt:
            x = np.arange(0, len(cur_embed_distance))
            fig = _debug_plt(
                x,
                cur_embed_distance,
                start_positions,
                end_positions,
                return_numpy=debug_plt_to_wandb,
            )
            if debug_plt_to_wandb:
                debug_plt_figs.append(fig)

        subgoal_indices.append(cur_subgoal_idx)
        subgoal_starts.append(start_positions[-1])
        if len(end_positions) < 2 or end_positions[-2] < 15:
            if threshold_subgoal_passing is not None:
                cur_milestone_dist = None
                if len(subgoal_indices) > 1:
                    cur_milestone_dist = np.linalg.norm(
                        no_robot_embeddings[cur_subgoal_idx]  # cur
                        - no_robot_embeddings[subgoal_indices[-2]]  # prev
                    )  # use the order from start to end
                if fill_embeddings:
                    for step in reversed(range(cur_subgoal_idx + 1)):
                        if (
                            len(subgoal_indices) == 1
                            or unnormalized_embed_distance[step] / cur_milestone_dist
                            > threshold_subgoal_passing
                        ):
                            milestone_embeddings.append(embeddings[cur_subgoal_idx])
                        else:
                            milestone_embeddings.append(embeddings[subgoal_indices[-2]])
            break
        if threshold_subgoal_passing is not None:
            cur_milestone_dist = None
            if len(subgoal_indices) > 1:
                cur_milestone_dist = np.linalg.norm(
                    no_robot_embeddings[cur_subgoal_idx]  # cur
                    - no_robot_embeddings[subgoal_indices[-2]]  # prev
                )
            if fill_embeddings:
                for step in reversed(range(end_positions[-2] + 1, cur_subgoal_idx + 1)):
                    # not pass threshold or 1st iter with only last frame contained
                    if (
                        len(subgoal_indices) == 1
                        or unnormalized_embed_distance[step] / cur_milestone_dist
                        > threshold_subgoal_passing
                    ):
                        # use the real subgoal
                        milestone_embeddings.append(embeddings[cur_subgoal_idx])
                    else:
                        # skip to the next subgoal if passing threshold
                        milestone_embeddings.append(embeddings[subgoal_indices[-2]])
        cur_subgoal_idx = end_positions[-2]

    subgoal_starts = list(reversed(subgoal_starts))
    subgoal_indices = list(reversed(subgoal_indices))

    if len(debug_plt_figs) > 0 and wandb.run is not None:
        debug_plt_figs = np.concatenate(debug_plt_figs, axis=1)
        wandb.log(
            {
                f"decomp_curves/{task_name}": wandb.Image(
                    debug_plt_figs,
                    caption=f"starts: {subgoal_starts}, ends: {subgoal_indices}",
                )
            }
        )

    assert len(subgoal_starts) == len(
        subgoal_indices
    ), f"{subgoal_starts}, {subgoal_indices}"

    if force_interleave:
        _starts, _ends = [], []
        for i in range(len(subgoal_starts)):
            if subgoal_starts[i] < subgoal_indices[i]:
                _starts.append(subgoal_starts[i])
                _ends.append(subgoal_indices[i])
            else:
                U.get_logger().warning(
                    f"{subgoal_starts} & {subgoal_indices} not interleaved"
                )

    if fill_embeddings:
        if threshold_subgoal_passing is not None:
            milestone_embeddings = np.stack(list(reversed(milestone_embeddings)))
        else:
            # slightly faster to do once here without threshold checking
            milestone_embeddings = np.concatenate(
                [embeddings[subgoal_indices[0], ...][None]]
                + [
                    np.full((end - start, *embeddings.shape[1:]), embeddings[end, ...])
                    for start, end in zip([0] + subgoal_indices[:-1], subgoal_indices)
                ],
            )

        if device is not None:
            milestone_embeddings = U.any_to_torch_tensor(
                milestone_embeddings, device=device
            )
    return milestone_embeddings, DecompMeta(
        milestone_indices=subgoal_indices, milestone_starts=subgoal_starts
    )


def embed_decomp_no_robot_extended(
    embeddings: np.ndarray | torch.Tensor,
    no_robot_embeddings: np.ndarray | torch.Tensor,
    threshold_subgoal_passing: float | None = None,
    **kwargs,
):
    kwargs["fill_embeddings"] = False
    _, decomp_meta = embed_decomp_no_robot(
        embeddings,
        no_robot_embeddings,
        threshold_subgoal_passing=None,
        **kwargs,
    )
    milestone_indices = decomp_meta.milestone_indices
    milestone_starts = decomp_meta.milestone_starts
    norm = (
        np.linalg.norm if isinstance(embeddings[0], np.ndarray) else torch.linalg.norm
    )
    assert len(milestone_starts) == len(milestone_indices)

    milestone_embeddings = []
    hybrid_indices = list(sorted(milestone_starts + milestone_indices))
    prev_idx = -1
    s = -1
    init_dist = None
    for i, goal_idx in enumerate(hybrid_indices):
        once_passed = False
        for _ in range(goal_idx - prev_idx):
            s += 1
            if threshold_subgoal_passing is None:
                milestone_embeddings.append(embeddings[goal_idx])
            elif once_passed:
                milestone_embeddings.append(embeddings[hybrid_indices[i + 1]])
            else:
                raw_cur_dist = float(norm(embeddings[s] - embeddings[goal_idx]))
                if init_dist is None:
                    assert s == 0, s
                    cur_dist = 1.0
                    init_dist = max(raw_cur_dist, 1e-7)
                else:
                    cur_dist = raw_cur_dist / init_dist
                if (
                    cur_dist <= threshold_subgoal_passing
                    and i < len(hybrid_indices) - 1
                ):
                    init_dist = float(
                        norm(embeddings[s] - embeddings[hybrid_indices[i + 1]])
                    )
                    init_dist = max(init_dist, 1e-7)
                    once_passed = True
                    milestone_embeddings.append(embeddings[hybrid_indices[i + 1]])
                else:
                    milestone_embeddings.append(embeddings[goal_idx])
        prev_idx = goal_idx

    milestone_embeddings = U.any_stack(milestone_embeddings)
    U.assert_(milestone_embeddings.shape, embeddings.shape)
    return milestone_embeddings, DecompMeta(milestone_indices=hybrid_indices)


def get_hybrid_milestones(
    start_embeddings: np.ndarray,  # w. robot
    end_embeddings: np.ndarray,  # w.o robot
    milestone_starts: list,
    milestone_indices: list,
) -> np.ndarray:
    assert len(milestone_starts) == len(milestone_indices)
    assert len(start_embeddings) == len(end_embeddings)
    milestone_only = len(milestone_starts) == len(start_embeddings)
    hybrid_milestones = np.empty(
        (start_embeddings.shape[0] * 2, *start_embeddings.shape[1:]),
        dtype=start_embeddings.dtype,
    )
    hybrid_milestones[::2] = (
        start_embeddings if milestone_only else start_embeddings[milestone_starts]
    )
    hybrid_milestones[1::2] = (
        end_embeddings if milestone_only else end_embeddings[milestone_indices]
    )
    return hybrid_milestones


def embedding_decomp(
    embeddings: np.ndarray | torch.Tensor,
    normalize_curve: bool = True,
    min_interval: int = 18,
    window_length: int | None = None,
    smooth_method: Literal["kernel", "savgol"] = "kernel",
    extrema_comparator: Callable = np.greater,
    fill_embeddings: bool = True,
    return_intermediate_curves: bool = False,
    **kwargs,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    if torch.is_tensor(embeddings):
        device = embeddings.device
        embeddings = U.any_to_numpy(embeddings)
    else:
        device = None
    # L, N
    assert embeddings.ndim == 2, embeddings.shape
    traj_length = embeddings.shape[0]

    cur_goal_idx = traj_length - 1
    goal_indices = [cur_goal_idx]
    cur_embeddings = embeddings[
        max(0, cur_goal_idx - (window_length or cur_goal_idx)) : cur_goal_idx + 1
    ]
    iterate_num = 0
    iter_curves = [] if return_intermediate_curves else None
    while cur_goal_idx > (window_length or min_interval):
        iterate_num += 1
        # get goal embedding
        goal_embedding = cur_embeddings[-1]
        distances = np.linalg.norm(cur_embeddings - goal_embedding, axis=1)
        if normalize_curve:
            distances = distances / np.linalg.norm(cur_embeddings[0] - goal_embedding)

        x = np.arange(
            max(0, cur_goal_idx - (window_length or cur_goal_idx)), cur_goal_idx + 1
        )

        if smooth_method == "kernel":
            smooth_kwargs = dict(kernel="rbf", gamma=0.08)
            smooth_kwargs.update(kwargs or {})
            kr = KernelRegression(**smooth_kwargs)
            kr.fit(x.reshape(-1, 1), distances)
            distance_smoothed = kr.predict(x.reshape(-1, 1))
        elif smooth_method == "savgol":
            smooth_kwargs = dict(window_length=85, polyorder=2, mode="nearest")
            smooth_kwargs.update(kwargs or {})
            distance_smoothed = savgol_filter(distances, **smooth_kwargs)
        elif smooth_method is None:
            distance_smoothed = distances
        else:
            raise NotImplementedError(smooth_method)

        if iter_curves is not None:
            iter_curves.append(distance_smoothed)

        extrema_indices = argrelextrema(distance_smoothed, extrema_comparator)[0]
        x_extrema = x[extrema_indices]

        update_goal = False
        for i in range(len(x_extrema) - 1, -1, -1):
            if cur_goal_idx < min_interval:
                break
            if (
                cur_goal_idx - x_extrema[i] > min_interval
                and x_extrema[i] > min_interval
            ):
                cur_goal_idx = x_extrema[i]
                update_goal = True
                goal_indices.append(cur_goal_idx)
                break

        if not update_goal or cur_goal_idx < min_interval:
            break
        cur_embeddings = embeddings[
            max(0, cur_goal_idx - (window_length or cur_goal_idx)) : cur_goal_idx + 1
        ]

    goal_indices = goal_indices[::-1]
    if fill_embeddings:
        milestone_embeddings = np.concatenate(
            [embeddings[goal_indices[0], ...][None]]
            + [
                np.full((end - start, *embeddings.shape[1:]), embeddings[end, ...])
                for start, end in zip([0] + goal_indices[:-1], goal_indices)
            ],
        )
        if device is not None:
            milestone_embeddings = U.any_to_torch_tensor(
                milestone_embeddings, device=device
            )
    else:
        milestone_embeddings = None
    return milestone_embeddings, DecompMeta(
        milestone_indices=goal_indices, iter_curves=iter_curves
    )


def goal_idx_from_mask(goal_achieved_mask):
    diff = np.diff(goal_achieved_mask)
    goal_indices = np.where(diff != 0)[0] + 1
    traj_length = goal_achieved_mask.shape[0]
    goal_indices[-1] = traj_length - 1  # last
    goal_indices = goal_indices.tolist()
    return goal_indices


def oracle_decomp(
    embeddings: np.ndarray | torch.Tensor | None,
    goal_achieved_mask: np.ndarray,
    random_skip_ratio: float | None = None,
    linearly_random_skip_lower: float | None = None,
    fill_embeddings: bool = True,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    """Note: embeddings here only has the oracle subgoals, not full trajectory"""
    goal_indices = goal_idx_from_mask(goal_achieved_mask)
    if not fill_embeddings:
        return None, DecompMeta(milestone_indices=goal_indices)

    traj_length = goal_achieved_mask.shape[0]
    assert embeddings.shape[0] < traj_length, embeddings.shape
    milestone_embeddings = (
        torch.empty(
            (goal_achieved_mask.shape[0], *embeddings.shape[1:]),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        if not isinstance(embeddings, np.ndarray)
        else np.empty(
            (goal_achieved_mask.shape[0], *embeddings.shape[1:]),
            dtype=embeddings.dtype,
        )
    )

    assert len(goal_indices) == len(embeddings), goal_indices

    last_embedding = embeddings[-1]
    for i, idx in enumerate(goal_achieved_mask):
        if idx >= embeddings.shape[0]:
            # If the index in the mask is greater than the highest index in the embedding,
            # just use the last row of the embedding
            milestone_embeddings[i] = last_embedding
        else:
            skip = False
            if linearly_random_skip_lower is not None:
                skip = linear_random_skip(
                    cur_step=i,
                    next_goal_step=goal_indices[idx],
                    ratio=random_skip_ratio,
                    progress_lower=linearly_random_skip_lower,
                )
            elif (
                random_skip_ratio is not None
                and 0 < random_skip_ratio < random.random()
            ):
                skip = True
            if skip:
                milestone_embeddings[i] = embeddings[min(idx + 1, len(embeddings) - 1)]
            else:
                milestone_embeddings[i] = embeddings[idx]
    return milestone_embeddings, DecompMeta(milestone_indices=goal_indices)


def random_decomp(
    embeddings: np.ndarray | torch.Tensor,
    num_milestones: int | tuple[int, int],
    fill_embeddings: bool = True,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    if not isinstance(num_milestones, int):
        assert len(num_milestones) == 2, num_milestones
        # by randomly sample from lower and higher bound
        num_milestones = random.randint(*num_milestones)
    traj_length = embeddings.shape[0]
    goal_indices = random.sample(range(traj_length), k=num_milestones)
    goal_indices = list(sorted(goal_indices))
    if fill_embeddings:
        milestone_embeddings = (
            torch.empty_like(
                embeddings, dtype=embeddings.dtype, device=embeddings.device
            )
            if not isinstance(embeddings, np.ndarray)
            else np.empty_like(embeddings, dtype=embeddings.dtype)
        )
        for i, goal_idx in enumerate(goal_indices):
            milestone_embeddings[
                (goal_indices[i - 1] + 1) if i != 0 else 0 : goal_idx + 1
            ] = embeddings[goal_idx]
    else:
        milestone_embeddings = None
    return milestone_embeddings, DecompMeta(milestone_indices=goal_indices)


def equally_decomp(
    embeddings: np.ndarray | torch.Tensor,
    num_milestones: int | tuple[int, int],
    fill_embeddings: bool = True,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    if not isinstance(num_milestones, int):
        assert len(num_milestones) == 2, num_milestones
        # by randomly sample from lower and higher bound
        num_milestones = random.randint(*num_milestones)
    traj_length = embeddings.shape[0]
    indices = np.linspace(0, traj_length - 1, num_milestones + 1, dtype=int)
    if fill_embeddings:
        milestone_embeddings = (
            torch.empty_like(
                embeddings, dtype=embeddings.dtype, device=embeddings.device
            )
            if not isinstance(embeddings, np.ndarray)
            else np.empty_like(
                embeddings,
                dtype=embeddings.dtype,
            )
        )
        for i, goal_idx in enumerate(indices[1:], start=1):
            milestone_embeddings[
                (indices[i - 1] + 1) if i != 1 else 0 : goal_idx + 1
            ] = embeddings[goal_idx]
    else:
        milestone_embeddings = None
    return milestone_embeddings, DecompMeta(milestone_indices=indices[1:].tolist())


def near_future_decomp(
    embeddings: np.ndarray | torch.Tensor, advance_steps: int, **kwargs
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    return equally_decomp(
        embeddings, num_milestones=embeddings.shape[0] // advance_steps, **kwargs
    )


def no_decomp(
    embeddings: np.ndarray | torch.Tensor, fill_embeddings: bool = True
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    """Only conditioned on final goal."""
    if not fill_embeddings:
        return None, DecompMeta(milestone_indices=[-1])
    return embeddings[-1, ...].expand_as(embeddings).clone() if not isinstance(
        embeddings, np.ndarray
    ) else np.full(
        embeddings.shape,
        embeddings[-1, ...],
        dtype=embeddings.dtype,
    ), DecompMeta(
        milestone_indices=[-1]
    )


def decomp_trajectories(
    method_name: Literal[
        "embed", "embed_no_robot", "oracle", "random", "equally", "near_future"
    ]
    | None,
    embeddings: np.ndarray | torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor | np.ndarray, DecompMeta]:
    assert embeddings.ndim == 2 or embeddings.ndim == 4, (
        f"input embedding should be either 2 dimensional, "
        f"with (L, feature_dim), or raw rgb with shape (L, H, W, 3), "
        f"but get {embeddings.shape}"
    )
    if method_name is None:
        return no_decomp(embeddings)
    assert method_name in DEFAULT_DECOMP_KWARGS, method_name
    method_kwargs = DEFAULT_DECOMP_KWARGS[method_name]
    method_kwargs.update(kwargs)
    if method_name == "embed":
        return embedding_decomp(embeddings=embeddings, **method_kwargs)
    elif method_name == "embed_no_robot":
        return embed_decomp_no_robot(embeddings=embeddings, **method_kwargs)
    elif method_name == "embed_no_robot_extended":
        return embed_decomp_no_robot_extended(embeddings=embeddings, **method_kwargs)
    elif method_name == "oracle":
        return oracle_decomp(embeddings, **method_kwargs)
    elif method_name == "random":
        return random_decomp(embeddings, **method_kwargs)
    elif method_name == "equally":
        return equally_decomp(embeddings, **method_kwargs)
    elif method_name == "near_future":
        return near_future_decomp(embeddings, **method_kwargs)
    raise NotImplementedError(method_name)


DEFAULT_DECOMP_KWARGS = dict(
    embed=dict(
        normalize_curve=False,
        min_interval=18,
        smooth_method="kernel",
        gamma=0.08,
    ),
    embed_no_robot=dict(
        window_length=8,
        derivative_order=1,
        derivative_threshold=1e-3,
        threshold_subgoal_passing=None,
    ),
    embed_no_robot_extended=dict(
        window_length=3,
        derivative_order=1,
        derivative_threshold=1e-3,
        threshold_subgoal_passing=None,
    ),
    oracle=dict(),
    random=dict(num_milestones=(3, 6)),
    equally=dict(num_milestones=(3, 6)),
    near_future=dict(advance_steps=5),
)
