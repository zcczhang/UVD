from __future__ import annotations

import random
from typing import Literal

import gym
import numpy as np
import torch

import uvd.utils as U
from uvd.decomp import decomp_trajectories
from uvd.envs.franka_kitchen import KitchenBase
from .franka_kitchen_datasets import (
    FrankaKitchenDataset,
)

__all__ = ["DatasetWithAug", "MilestoneRandomSkipDataset"]


class DatasetWithAug(FrankaKitchenDataset):
    def __init__(
        self,
        dataset_path: str | list,
        *,
        specific_tasks: str | list | None,
        obs_keys: str | tuple = ("rgb", "proprio"),
        num_demos: int | None = None,
        shuffle: bool = False,
        replace_last_goal_as_last_frame: bool = True,
        # if `preprocess`, the observation for training is directly embeddings
        preprocess: bool = True,
        preprocess_name: str | None = None,
        preprocess_kwargs: dict | None = None,
        decomp_method: Literal["oracle", "embed", "random", "equally", "near_future"]
        | None = "embed",
        decomp_kwargs: dict | None = None,
        sequential: bool = True,
        include_no_robot: bool = False,
        del_preprocessor: bool = True,
        device: int | str | torch.device | None = None,
    ):
        decomp_kwargs = decomp_kwargs or {}
        decomp_kwargs.update(fill_embeddings=False)
        self.idx_to_all_milestones = {}
        self.max_num_milestones = -1
        super().__init__(**U.prepare_locals_for_super(locals()))

    def prepare_episode_data(
        self,
        episode_idx: int,
        episode: str,
        use_tensor: bool = True,
        sequential: bool = True,
    ):
        self.idx_to_episode_path[episode_idx] = episode
        data = U.load_pickle(episode)
        actions = data["actions"]
        episode_length = actions.shape[0]
        full_obs_data = data["obs_full"]
        rgb_traj = full_obs_data["rgb"][:-1, ...]  # rm goal
        raw_rgb_no_robot = data.get("no_robot_rgb", None)
        if self.include_no_robot:
            assert raw_rgb_no_robot is not None
            raw_rgb_no_robot = raw_rgb_no_robot[:-1]
            assert raw_rgb_no_robot.shape == rgb_traj.shape
        else:
            raw_rgb_no_robot = None
        assert rgb_traj.shape[0] == episode_length
        self._idx_to_episode_length[episode_idx] = episode_length

        prepared_data = dict()

        if use_tensor:
            actions = U.any_to_torch_tensor(actions, device="cpu", dtype=torch.float32)
        else:
            actions = U.any_to_numpy(actions, dtype="float32")
        prepared_data["actions"] = actions

        # embeddings are either fully preprocessed or transformed
        if self.preprocess or (
            self.decomp_method is not None and "embed" in self.decomp_method
        ):
            # preprocess with frozen visual backbones
            embeddings = self.preprocessor.process(
                raw_img_batch=rgb_traj, return_numpy=not use_tensor
            )
            # L, N
            assert embeddings.shape[0] == episode_length, embeddings.shape
        else:
            # L, 3, H, W
            embeddings = self.preprocessor.transform_images(rgb_traj)

        obs = {}
        for k, v in full_obs_data.items():
            if k in self.obs_keys and k == "rgb":
                # raw rgb, may use for rollout
                obs[k] = rgb_traj  # v[:-1]
            elif k in self.obs_keys:
                obs[k] = (
                    U.any_to_torch_tensor(
                        v[:-1], dtype=torch.float32, device="cpu"  # self.device
                    )
                    if use_tensor
                    else U.any_to_numpy(v[:-1], dtype="float32")
                )

        # obs = {k: v[:-1] for k, v in full_obs_data.items() if k in self.obs_keys}
        prepared_data["obs"] = obs
        rgb_no_robot = None
        if raw_rgb_no_robot is not None:
            if self.preprocess:
                rgb_no_robot = self.preprocessor.process(
                    raw_img_batch=raw_rgb_no_robot,
                    return_numpy=not use_tensor,
                    reconstruct_linear=True,
                )
            else:
                rgb_no_robot = self.preprocessor.transform_images(raw_rgb_no_robot)
            prepared_data["rgb_no_robot"] = rgb_no_robot

        if self.decomp_method == "oracle":
            num_goals_achieved = data["completed_tasks"]
            _, decomp_meta = decomp_trajectories(
                method_name=self.decomp_method,
                embeddings=None,
                goal_achieved_mask=num_goals_achieved,
                **self.decomp_kwargs or {},
            )
        else:
            decomp_kwargs = self.decomp_kwargs or {}
            if (
                self.decomp_method is not None
                and "embed_no_robot" in self.decomp_method
            ):
                decomp_kwargs["no_robot_embeddings"] = rgb_no_robot
            elif self.decomp_method == "embed2":
                decomp_kwargs["no_robot_embeddings"] = embeddings
                decomp_kwargs["task_name"] = str(data["reset_kwargs"]["task_elements"])
            _, decomp_meta = decomp_trajectories(
                method_name=self.decomp_method,
                embeddings=embeddings,
                **decomp_kwargs,
            )

        if self.preprocessor.use_language_goal:
            raise NotImplementedError

        if use_tensor:
            embeddings = U.any_to_torch_tensor(embeddings, device="cpu")

        milestone_indices = decomp_meta.milestone_indices
        prepared_data["milestone_indices"] = milestone_indices
        self.max_num_milestones = max(len(milestone_indices), self.max_num_milestones)

        diffs = np.diff(milestone_indices, prepend=0)
        milestone_step_mask = np.repeat(np.arange(len(diffs)), diffs)
        milestone_step_mask = np.concatenate(
            [np.array([0]), milestone_step_mask], dtype=np.int32
        )
        assert len(milestone_step_mask) == episode_length, (
            len(milestone_step_mask),
            episode_length,
        )

        milestone_embeddings = embeddings[milestone_indices]

        if sequential:
            if self.split_train_eval:
                if episode.split("/")[-2] not in self.train_tasks:
                    assert episode.split("/")[-2] in self.eval_tasks, episode.split(
                        "/"
                    )[-2]
                    return prepared_data, episode_length

            def maybe_tensor(x, dtype):
                return (
                    torch.tensor(x, device="cpu", dtype=dtype)
                    if use_tensor
                    else np.array(x, dtype=dtype)
                )

            for st in range(episode_length):
                seq_data = dict(
                    actions=actions[st],
                    milestones=milestone_embeddings,
                    milestone_indices=maybe_tensor(milestone_indices, dtype="int32"),
                    cur_milestone_idx=maybe_tensor(
                        milestone_step_mask[st], dtype="int32"
                    ),
                    episode_idx=maybe_tensor([episode_idx], dtype="int32"),
                    timesteps=maybe_tensor([st], dtype="int32"),
                )
                obs_i = {
                    k: v[st] if k != "rgb" else embeddings[st] for k, v in obs.items()
                }
                seq_data["obs"] = obs_i
                self.sequence_data.append(seq_data)
        else:
            raise NotImplementedError

        return prepared_data, episode_length

    @property
    def dataset_metadata(self) -> dict:
        assert self.max_num_milestones > 0, self.max_num_milestones

        dummy_env = KitchenBase(obs_keys=self.obs_keys)
        action_space = dummy_env.action_space
        observation_space = dummy_env.observation_space.spaces
        dummy_env.close()
        del dummy_env
        if "rgb" in observation_space and self.preprocess:
            observation_space["rgb"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.preprocessor_output_dim,
                dtype=np.float32,
            )
        observation_space = gym.spaces.Dict(observation_space)
        metadata = dict(action_space=action_space, observation_space=observation_space)
        metadata["observation_space"]["milestones"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_num_milestones,) + self.preprocessor_output_dim,
            dtype=np.float32,
        )
        return metadata

    def __getitem__(self, idx: int) -> dict:
        data = self.sequence_data[idx]
        milestones = data["milestones"]
        milestone_indices = data["milestone_indices"]
        num_milestones, d = milestones.shape
        assert num_milestones <= self.max_num_milestones
        pad_length = self.max_num_milestones - num_milestones
        if pad_length != 0:
            milestones = U.any_concat(
                # assume not use tensor
                [milestones, np.zeros((pad_length, d), dtype=milestones.dtype)],
                dim=0,
            )
            milestone_indices = U.any_concat(
                [milestone_indices, milestone_indices[-1][None] * pad_length]
            )
        # N, D
        data["milestones"] = milestones  # .reshape((self.max_num_milestones * d,))
        data["milestone_masks"] = np.array(
            [1] * num_milestones + [0] * pad_length, dtype=np.bool_
        )[
            :, None
        ]  # N, 1
        data["milestone_indices"] = milestone_indices
        return data


class MilestoneRandomSkipDataset(DatasetWithAug):
    def __init__(
        self, *, skip_ratio: 0.1, min_skip_n_milestones: int | None = None, **kwargs
    ):
        self.skip_ratio = skip_ratio
        self.min_skip_n_milestones = min_skip_n_milestones or 0
        super().__init__(**kwargs)

    def __getitem__(self, idx: int) -> dict:
        data = self.sequence_data[idx]
        milestones = data["milestones"]
        milestone_indices = data["milestone_indices"].copy()
        num_milestones = len(milestone_indices)
        assert len(milestones) == num_milestones, (
            len(milestones),
            num_milestones,
            milestone_indices,
        )
        cur_milestone_idx = data["cur_milestone_idx"].copy()

        do_skip = (
            len(milestone_indices) > self.min_skip_n_milestones
            and random.random() > self.skip_ratio
        )
        if do_skip:
            skip_back = (
                cur_milestone_idx > 0 and random.random() > 0.5
            ) or cur_milestone_idx == num_milestones - 1
            if skip_back:
                cur_milestone_idx -= 1
            else:
                cur_milestone_idx += 1
            assert 0 <= cur_milestone_idx <= num_milestones - 1, (
                cur_milestone_idx,
                milestone_indices,
            )

        pad_length = self.max_num_milestones - num_milestones
        if pad_length != 0:
            milestone_indices = U.any_concat(
                [data["milestone_indices"]]
                + [data["milestone_indices"][-1][None]] * pad_length
            )
        else:
            milestone_indices = data["milestone_indices"]

        return dict(
            obs=data["obs"],
            milestones=milestones[cur_milestone_idx],
            milestone_indices=milestone_indices,
            cur_milestone_idx=cur_milestone_idx,
            actions=data["actions"],
            episode_idx=data["episode_idx"],
            timesteps=data["timesteps"],
        )
