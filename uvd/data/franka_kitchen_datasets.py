from __future__ import annotations

import collections
import copy
import json
import random
from collections import OrderedDict
from typing import Literal

import gym
import numpy as np
import torch
import tqdm
import wandb
from torch.distributions.utils import lazy_property
from torch.utils.data import Dataset

import uvd.utils as U
from uvd.decomp import decomp_trajectories
from uvd.envs.franka_kitchen import KitchenBase
from uvd.envs.franka_kitchen.franka_kitchen_constants import ELEMENT_TO_IDX
from uvd.models.preprocessors import get_preprocessor
from .dataset_base import DatasetBase

__all__ = [
    "FrankaKitchenDataset",
    "FrankaKitchenRolloutDataset",
    "ALL_TASKS",
    "task_elements_to_prompt",
]


ALL_TASKS = [
    "bottom_burner-top_burner-light_switch-slide_cabinet",
    "bottom_burner-top_burner-slide_cabinet-hinge_cabinet",
    "kettle-bottom_burner-light_switch-hinge_cabinet",
    "kettle-bottom_burner-light_switch-slide_cabinet",
    "kettle-bottom_burner-slide_cabinet-hinge_cabinet",
    "kettle-bottom_burner-top_burner-hinge_cabinet",
    "kettle-bottom_burner-top_burner-light_switch",
    "kettle-bottom_burner-top_burner-slide_cabinet",
    "kettle-light_switch-slide_cabinet-hinge_cabinet",
    "kettle-top_burner-light_switch-slide_cabinet",
    "microwave-bottom_burner-light_switch-slide_cabinet",
    "microwave-bottom_burner-slide_cabinet-hinge_cabinet",
    "microwave-bottom_burner-top_burner-hinge_cabinet",
    "microwave-bottom_burner-top_burner-light_switch",
    "microwave-bottom_burner-top_burner-slide_cabinet",
    "microwave-kettle-bottom_burner-hinge_cabinet",
    "microwave-kettle-bottom_burner-slide_cabinet",
    "microwave-kettle-light_switch-hinge_cabinet",
    "microwave-kettle-light_switch-slide_cabinet",
    "microwave-kettle-slide_cabinet-hinge_cabinet",
    "microwave-kettle-top_burner-hinge_cabinet",
    "microwave-kettle-top_burner-light_switch",
    "microwave-light_switch-slide_cabinet-hinge_cabinet",
    "microwave-top_burner-light_switch-hinge_cabinet",
]


class FrankaKitchenDataset(DatasetBase):
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
        max_seq_length: int | None = None,
        include_no_robot: bool = False,
        del_preprocessor: bool = True,
        device: int | str | torch.device | None = None,
    ):
        super().__init__()

        self.split_train_eval = None
        if "random_" in specific_tasks:
            n_train = int(specific_tasks.split("_")[-1])
            specific_tasks = ALL_TASKS
            random.shuffle(specific_tasks)
            self.split_train_eval = n_train

        if isinstance(dataset_path, str):
            assert U.f_exists(dataset_path), dataset_path
            if specific_tasks is not None:
                if isinstance(specific_tasks, str):
                    specific_tasks = (specific_tasks,)
                dataset_path = [U.f_join(dataset_path, t) for t in specific_tasks]
            elif "metadata.json" in U.f_listdir(dataset_path):
                # single task
                dataset_path = (dataset_path,)
            else:
                # use all tasks
                dataset_path = [
                    U.f_join(dataset_path, d)
                    for d in U.f_listdir(dataset_path)
                    if "-" in d and "_" in d
                ]

        if self.split_train_eval is not None:
            assert specific_tasks is not None
            self.train_tasks = specific_tasks[: self.split_train_eval]
            self.eval_tasks = specific_tasks[self.split_train_eval :]
            U.rank_zero_print(
                f"{len(self.train_tasks)} training tasks, {len(self.eval_tasks)} unseen tasks",
                color="green",
            )
            if U.is_rank_zero() and wandb.run is not None:
                table = wandb.Table(
                    data=[[",\n".join(self.train_tasks), ",\n".join(self.eval_tasks)]],
                    columns=["training tasks", "unseen tasks"],
                )
                wandb.log(dict(partition=table))

        assert sum((U.f_exists(d) for d in dataset_path)) == len(dataset_path)
        self.include_no_robot = include_no_robot and decomp_method is not None
        if decomp_method == "embed_no_robot":
            assert self.include_no_robot and preprocess
        self.cache_scores = dict()

        episode_for_task = collections.defaultdict(list)
        for d in dataset_path:
            for f in U.f_listdir(d):
                if num_demos is not None and len(episode_for_task[d]) >= num_demos:
                    break
                if f.endswith(".pkl") and f.startswith("episode"):
                    episode_for_task[d].append(U.f_join(d, f))

        episodes = sorted([f for v in episode_for_task.values() for f in v])

        U.rank_zero_print(
            "# Train Demos:\n"
            + json.dumps(
                {
                    k.split("/")[-1]: len(v)
                    for k, v in episode_for_task.items()
                    if k.split("/")[-1] in self.train_tasks
                },
                indent=2,
            )
            + "\n# Eval Demos:\n"
            + json.dumps(
                {
                    k.split("/")[-1]: len(v)
                    for k, v in episode_for_task.items()
                    if k.split("/")[-1] in self.eval_tasks
                },
                indent=2,
            )
            + f"\nTotal: {len(episodes)}",
            color="blue",
        )

        if shuffle:
            random.shuffle(episodes)

        self.obs_keys = (
            tuple(obs_keys) if not isinstance(obs_keys, str) else (obs_keys,)
        )
        self.replace_last_goal_as_last_frame = replace_last_goal_as_last_frame
        self.num_demos = num_demos
        self.idx_to_episode_path = {}
        self._idx_to_episode_length = {}

        self.preprocess = preprocess
        self.device = device
        if preprocess or decomp_method == "embed":
            assert preprocess_name is not None
            self.preprocessor = get_preprocessor(
                name=preprocess_name, device=device, **preprocess_kwargs or {}
            )
            self.preprocessor_output_dim = self.preprocessor.output_dim
        else:
            # do transform only, with self.preprocess is False
            self.preprocessor = get_preprocessor(
                name=preprocess_name, device=device, **preprocess_kwargs or {}
            )
        self.decomp_method = decomp_method
        self.decomp_kwargs = decomp_kwargs
        self.use_language_goal = self.preprocessor.use_language_goal

        self.episode_data = {}
        self.sequence_data = []
        self.sequential = sequential
        for episode_idx, episode in tqdm.tqdm(
            enumerate(episodes),
            total=len(episodes),
            desc=f"load {self.__class__.__name__}",
        ):
            prepared_data, episode_length = self.prepare_episode_data(
                episode_idx, episode, sequential=sequential, use_tensor=False  # True
            )
            self.episode_data[episode_idx] = prepared_data

        assert len(self._idx_to_episode_length) == len(episodes)
        self._max_episode_length = max(self._idx_to_episode_length.values())
        self.max_seq_length = None
        self.idx_partition = None
        if max_seq_length is not None:
            assert max_seq_length <= self._max_episode_length
            if max_seq_length < self._max_episode_length:
                self.max_seq_length = max_seq_length
                self.idx_partition = [
                    (ep_idx, (max(0, step_end - max_seq_length), step_end))
                    for ep_idx in range(
                        len(self.sequence_data)
                    )  # only for training data
                    for step_end in range(
                        1,
                        self._idx_to_episode_length[
                            int(self.sequence_data[ep_idx]["episode_idx"])
                        ]
                        + 1,
                    )
                ]

        if del_preprocessor:
            # del after using
            del self.preprocessor

        U.rank_zero_print(
            f"TOTAL TRAINING DATA: {len(self.sequence_data)}", color="blue"
        )

    @property
    def max_episode_length(self) -> int:
        return self._max_episode_length

    @lazy_property
    def dataset_metadata(self) -> dict:
        dummy_env = KitchenBase(obs_keys=self.obs_keys)
        action_space = dummy_env.action_space
        observation_space = dummy_env.observation_space.spaces
        dummy_env.close()
        del dummy_env
        if "rgb" in observation_space and self.preprocess:
            observation_space["rgb"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.preprocessor_output_dim,)
                if isinstance(self.preprocessor_output_dim, int)
                else self.preprocessor_output_dim,
                dtype=np.float32,
            )
        observation_space = gym.spaces.Dict(observation_space)
        metadata = dict(action_space=action_space, observation_space=observation_space)
        if not self.sequential:  # causal
            metadata["max_episode_length"] = self.max_episode_length
            if self.max_seq_length is not None:
                metadata["max_seq_length"] = self.max_seq_length
        return metadata

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
        task_elements = data["reset_kwargs"]["task_elements"]
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
            oracle_milestones = data["oracle_milestones"]
            if self.replace_last_goal_as_last_frame:
                oracle_milestones[-1, ...] = rgb_traj[-1]  # full_obs_data["rgb"][-1]
            if self.preprocess:
                oracle_milestone_embeddings = self.preprocessor.process(
                    raw_img_batch=oracle_milestones, return_numpy=not use_tensor
                )
            else:
                oracle_milestone_embeddings = self.preprocessor.transform_images(
                    oracle_milestones
                )
            assert len(oracle_milestone_embeddings) == len(oracle_milestones)
            num_goals_achieved = data["completed_tasks"]
            milestone_embeddings, decomp_meta = decomp_trajectories(
                method_name=self.decomp_method,
                embeddings=oracle_milestone_embeddings,
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
            milestone_embeddings, decomp_meta = decomp_trajectories(
                method_name=self.decomp_method,
                embeddings=embeddings,
                **decomp_kwargs,
            )

        if self.preprocessor.use_language_goal:
            assert self.preprocess
            prompt = task_elements_to_prompt(task_elements)
            prompt_embed = self.preprocessor.encode_text(prompt)
            milestone_embeddings = prompt_embed.repeat(episode_length, 1)
            if not use_tensor:
                milestone_embeddings = U.any_to_numpy(milestone_embeddings)
            prepared_data["lang_embed"] = milestone_embeddings[-1]

        if use_tensor:
            embeddings = U.any_to_torch_tensor(embeddings, device="cpu")
            milestone_embeddings = U.any_to_torch_tensor(
                milestone_embeddings, device="cpu"
            )
        assert (
            milestone_embeddings.shape == embeddings.shape
        ), f"{milestone_embeddings.shape} != {embeddings.shape}"
        prepared_data["milestone_indices"] = decomp_meta.milestone_indices

        def maybe_tensor(x, dtype):
            return (
                torch.tensor(x, device="cpu", dtype=dtype)
                if use_tensor
                else np.array(x, dtype=dtype)
            )

        if sequential:
            if self.split_train_eval:
                if episode.split("/")[-2] not in self.train_tasks:
                    assert episode.split("/")[-2] in self.eval_tasks, episode.split(
                        "/"
                    )[-2]
                    return prepared_data, episode_length
            for st in range(episode_length):
                seq_data = dict(
                    actions=actions[st],
                    milestones=milestone_embeddings[st],
                    episode_idx=maybe_tensor([episode_idx], dtype="int32"),
                    timesteps=maybe_tensor([st], dtype="int32"),
                )
                obs_i = {
                    k: v[st] if k != "rgb" else embeddings[st] for k, v in obs.items()
                }
                seq_data["obs"] = obs_i
                self.sequence_data.append(seq_data)
        else:
            if self.split_train_eval:
                if episode.split("/")[-2] not in self.train_tasks:
                    assert episode.split("/")[-2] in self.eval_tasks, episode.split(
                        "/"
                    )[-2]
                    return prepared_data, episode_length
            self.sequence_data.append(
                dict(
                    actions=actions,
                    obs={k: v if k != "rgb" else embeddings for k, v in obs.items()},
                    milestones=milestone_embeddings,
                    episode_idx=maybe_tensor([episode_idx], dtype="int32"),
                    timesteps=maybe_tensor(list(range(episode_length)), dtype="int32"),
                )
            )
        return prepared_data, episode_length

    def __len__(self) -> int:
        if self.max_seq_length is None:
            return len(self.sequence_data)
        else:
            return len(self.idx_partition)

    def __getitem__(self, idx: int) -> dict:
        """Dict(obs, action, milestone, episode_idx, (maybe embedding))"""
        if self.sequential:
            return self.sequence_data[idx]

        if self.max_seq_length is not None:
            episode_idx, (start_idx, end_idx) = self.idx_partition[idx]
            data = copy.deepcopy(self.sequence_data[episode_idx])
            pad_len = self.max_seq_length - (end_idx - start_idx)
            target_mask = np.array(
                [1] * (end_idx - 1 - start_idx) + [0] * pad_len, dtype=np.int32
            )
            for k in data.keys():
                val = data[k]
                if k == "obs":
                    for obs_k, v in val.items():
                        if pad_len > 0:
                            data[k][obs_k] = U.any_concat(
                                [v[start_idx:end_idx, ...]]
                                + [U.any_zeros_like(v[0])[None]] * pad_len
                            )
                        else:
                            data[k][obs_k] = v[start_idx:end_idx, ...]
                elif k != "episode_idx":
                    if pad_len > 0:
                        data[k] = U.any_concat(
                            [val[start_idx:end_idx, ...]]
                            + [U.any_zeros_like(val[0])[None]] * pad_len
                        )
                    else:
                        data[k] = val[start_idx:end_idx]
            data["target_mask"] = target_mask
            return data

        data = self.sequence_data[idx]  # seems no need to deepcopy st only pad once
        actions = data["actions"]
        ep_len = len(actions)
        pad_len = self.max_episode_length - ep_len
        if torch.is_tensor(actions):
            target_mask = torch.tensor(
                [1] * ep_len + [0] * pad_len, dtype=torch.int32, device=actions.device
            )
        else:
            target_mask = np.array([1] * ep_len + [0] * pad_len, dtype=np.int32)
        if pad_len > 0:
            for k in data.keys():
                val = data[k]
                if k == "obs":
                    for obs_k, v in val.items():
                        if len(v) == ep_len:
                            data[k][obs_k] = U.any_concat(
                                [v] + [U.any_zeros_like(v[0])[None]] * pad_len
                            )
                elif len(val) == ep_len:
                    data[k] = U.any_concat(
                        [val] + [U.any_zeros_like(val[0])[None]] * pad_len
                    )
        data["target_mask"] = target_mask
        return data


class FrankaKitchenRolloutDataset(Dataset):
    """Reuse training dataset for in-domain rollout evaluation."""

    def __init__(
        self,
        train_dataset: FrankaKitchenDataset,
        specific_tasks: str | list | None = None,
        # has to be specific tasks
        num_episodes_each_task: int | None = None,
        num_tasks: int | None = None,
        use_no_robot_rgb_milestones: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert "rgb" in train_dataset.obs_keys

        self.is_hybrid = train_dataset.decomp_method == "embed_no_robot_extended"

        self.max_num_milestones = -1
        episode_data = copy.deepcopy(train_dataset.episode_data)
        del train_dataset.episode_data
        self.dataset_metadata = train_dataset.dataset_metadata

        self._idx_to_episode_path = train_dataset.idx_to_episode_path
        if specific_tasks is not None:
            if isinstance(specific_tasks, str):
                if "unseen" in specific_tasks:
                    specific_tasks = train_dataset.eval_tasks
                elif "all" in specific_tasks:
                    specific_tasks = ALL_TASKS
                else:
                    specific_tasks = (specific_tasks,)
            # assume all data in the same parent directory
            ds_base_path = "/".join(self._idx_to_episode_path[0].split("/")[:-2])
            spec_episode_paths = [U.f_join(ds_base_path, t) for t in specific_tasks]
            assert sum([U.f_exists(p) for p in spec_episode_paths]) == len(
                spec_episode_paths
            ), spec_episode_paths

            task_counter = {k: 1 for k in specific_tasks}
            _idx_to_episode_path = {}
            _n_episode_each_task = num_episodes_each_task or np.inf
            for i, p in self._idx_to_episode_path.items():
                task_name = p.split("/")[-2]
                if (
                    task_name not in specific_tasks
                    or task_counter[task_name] > _n_episode_each_task
                ):
                    if task_name not in specific_tasks:
                        U.rank_zero_print(
                            f"WARNING: MISSING {task_name} in {specific_tasks}",
                            color="red",
                        )
                    continue
                task_counter[task_name] += 1
                _idx_to_episode_path[i] = p
            self._idx_to_episode_path = _idx_to_episode_path
        use_no_robot_rgb_milestones = (
            use_no_robot_rgb_milestones and train_dataset.decomp_method is not None
        )
        self.use_no_robot_rgb_milestones = use_no_robot_rgb_milestones

        self.num_demos = len(self._idx_to_episode_path)
        self.num_tasks = min(self.num_demos, num_tasks or self.num_demos)

        self.episode_data = {}
        idx = -1
        for ds_idx in tqdm.tqdm(
            episode_data.keys(),
            desc=f"load {self.__class__.__name__}",
            total=len(self._idx_to_episode_path),
        ):
            if ds_idx not in self._idx_to_episode_path:
                continue
            # else:
            #     print(ds_idx)
            idx += 1
            prepared_data = episode_data[ds_idx]
            milestone_indices = prepared_data["milestone_indices"]
            rgb = prepared_data["obs"]["rgb"]
            rgb = (
                U.any_to_torch_tensor(rgb, device="cpu", copy=True)
                if torch.is_tensor(rgb)
                else rgb.copy()
            )

            data = U.load_pickle(self._idx_to_episode_path[ds_idx])
            reset_kwargs = {
                k: U.any_to_numpy(v, dtype="float32")[None]
                if k != "task_elements"
                else np.array([ELEMENT_TO_IDX[ele] for ele in v], dtype=np.uint8)[None]
                for k, v in data["reset_kwargs"].items()
            }

            if train_dataset.use_language_goal:
                milestones = prepared_data["lang_embed"][None]  # (1, embed_d)
            else:
                milestones = rgb[milestone_indices]  # use raw rgb anyway
            if torch.is_tensor(milestones):
                milestones = U.any_to_torch_tensor(milestones, device="cpu")
            if rgb.ndim != milestones.ndim:
                rgb_milestones = U.any_to_numpy(rgb)[milestone_indices]
            else:
                rgb_milestones = None
            rgb_no_robot = prepared_data.get("rgb_no_robot", None)

            self.max_num_milestones = max(len(milestones), self.max_num_milestones)
            if use_no_robot_rgb_milestones and rgb_no_robot is None:
                rgb_no_robot = data["no_robot_rgb"][:-1]

            rgb_no_robot_milestones = (
                rgb_no_robot[milestone_indices] if use_no_robot_rgb_milestones else None
            )
            if torch.is_tensor(rgb_no_robot_milestones):
                rgb_no_robot_milestones = U.any_to_torch_tensor(
                    rgb_no_robot_milestones, device="cpu"
                )
            if torch.is_tensor(milestones):
                milestone_indices = torch.tensor(
                    milestones, dtype=torch.int32, device=milestones.device
                )
            else:
                milestone_indices = np.array(milestone_indices, dtype=np.int32)
            self.episode_data[idx] = dict(
                milestones=milestones,
                reset_kwargs=reset_kwargs,
                rgb_milestones=rgb_milestones,
                rgb_no_robot_milestones=rgb_no_robot_milestones,
                milestone_indices=milestone_indices,
            )

        del episode_data
        if self.num_tasks != self.num_demos:
            self.episode_data = [
                self.episode_data[i % len(self.episode_data)]
                for i in range(self.num_tasks)
            ]

    def __len__(self) -> int:
        """Num tasks evaluated."""
        return self.num_tasks

    def __getitem__(
        self, idx: int
    ) -> OrderedDict[str, np.ndarray | dict[str, np.ndarray]]:
        data = self.episode_data[idx]
        milestones = data["milestones"]
        milestone_indices = data.get("milestone_indices")
        rgb_milestones = data["rgb_milestones"]
        rgb_no_robot_milestones = data["rgb_no_robot_milestones"]
        # assert milestones.shape[0] == rgb_milestones.shape[0]
        num_milestones = len(milestones)
        assert num_milestones <= self.max_num_milestones
        if num_milestones < self.max_num_milestones:
            # pad last
            pad_length = self.max_num_milestones - num_milestones
            milestones = U.any_concat(
                [milestones] + [milestones[-1][None]] * pad_length
            )
            milestone_indices = U.any_concat(
                [milestone_indices] + [milestone_indices[-1][None]] * pad_length
            )
            # if milestones.ndim != rgb_milestones.ndim:
            if rgb_milestones is not None:
                rgb_milestones = U.any_concat(
                    [rgb_milestones] + [rgb_milestones[-1][None]] * pad_length
                )
            if rgb_no_robot_milestones is not None:
                rgb_no_robot_milestones = U.any_concat(
                    [rgb_no_robot_milestones]
                    + [rgb_no_robot_milestones[-1][None]] * pad_length
                )
        rollout_data = OrderedDict(
            milestones=milestones,
            # rgb_milestones=rgb_milestones,
            reset_kwargs=data["reset_kwargs"],
            # rgb_no_robot_milestones=rgb_no_robot_milestones,
            milestone_indices=milestone_indices,
        )
        if rgb_milestones is not None:
            rollout_data["rgb_milestones"] = rgb_milestones
        if rgb_no_robot_milestones is not None:
            rollout_data["rgb_no_robot_milestones"] = rgb_no_robot_milestones
        return rollout_data


PROMPT_DICT = dict(
    microwave="open the microwave",
    kettle="move the kettle to the top left stove",
    light_switch="turn on the light",
    hinge_cabinet="open the left hinge cabinet",
    slide_cabinet="open the right slide cabinet",
    top_burner="turn on the top left burner",
    bottom_burner="turn on the bottom left burner",
)
PROMPT_DICT.update({k.replace("_", " "): v for k, v in PROMPT_DICT.items()})


def task_elements_to_prompt(task_elements: list) -> str | list[str]:
    if isinstance(task_elements[0], str):
        prompt = ", ".join([PROMPT_DICT[_] for _ in task_elements])
        return prompt[0].capitalize() + prompt[1:]
    # batch of task elements
    assert isinstance(task_elements[0][0], str), task_elements
    return [task_elements_to_prompt(ele) for ele in task_elements]
