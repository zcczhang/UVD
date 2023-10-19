from __future__ import annotations

import copy

import einops
import gym
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import uvd.utils as U
import wandb
from uvd.envs.franka_kitchen import KitchenBase
from uvd.models.policy import PolicyBase

from .inference_wrapper import InferenceWrapper
from .vec_envs.vec_env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from .visualize_wrapper import VisualizeWrapper
from ...models import Preprocessor

__all__ = ["VectorEnvEvaluator"]


class VectorEnvEvaluator:
    def __init__(
        self,
        *,
        env_name: str,
        reset_kwargs: dict,
        num_tasks: int,
        num_envs: int | None,
        max_horizon: int = 400,
        no_robot_rgb_milestones: np.ndarray | None = None,
        use_no_robot_milestones: bool = False,
        inference_kwargs: dict | None,
        random_skip_inference: bool = False,
        use_milestone_distances_normalization: bool = False,
        decomp_method: str | None = None,
        milestones: np.ndarray,
        rgb_milestones: np.ndarray | None = None,
        seed: int | None = None,
        seed_list: list[int] | None = None,
        save_video: bool = False,
        save_video_kwargs: dict | None = None,
        device: int | str | torch.device | None = None,
        use_milestone_compressor: bool = False,
        milestone_indices: np.ndarray | None = None,
    ):
        self.env_name = env_name
        self.reset_kwargs = reset_kwargs
        num_envs = min(num_envs or num_tasks, num_tasks, milestones.shape[0])
        self.num_envs = num_envs
        self.rollout_list = self.generate_rollout_list(num_envs, num_tasks)

        inference_kwargs = inference_kwargs or {}
        if not random_skip_inference:
            inference_kwargs.update(random_skip_ratio=0.0)
        # inference_kwargs["hybrid"] = decomp_method == "embed_no_robot_extended"
        inference_kwargs["hybrid"] = False
        if decomp_method == "embed_no_robot_extended":
            use_no_robot_milestones = False
        inference_kwargs["use_milestone_compressor"] = use_milestone_compressor
        self.inference_kwargs = inference_kwargs
        self.decomp_method = decomp_method
        self.max_horizon = max_horizon
        self.milestones = milestones
        self.rgb_milestones = rgb_milestones
        self.no_robot_rgb_milestones = no_robot_rgb_milestones
        if not use_no_robot_milestones:
            self.no_robot_rgb_milestones = None
        elif self.no_robot_rgb_milestones is None:
            U.rank_zero_print(
                f"WARNING: set to use no robot milestones {use_no_robot_milestones} but there is no input",
                color="red",
            )
        milestone_distances = None
        if use_milestone_distances_normalization and milestones.shape[1] > 1:
            if self.no_robot_rgb_milestones is not None:
                # assume only using if with linear embedding
                assert (
                    self.no_robot_rgb_milestones.ndim == 3
                ), self.no_robot_rgb_milestones.shape
                # N_ENVS, N_GOALS, D
                milestone_distances = np.linalg.norm(
                    self.no_robot_rgb_milestones[:, :-1]
                    - self.no_robot_rgb_milestones[:, 1:],
                    axis=-1,
                )
            else:
                assert self.milestones.ndim == 3, self.milestones.shape
                milestone_distances = np.linalg.norm(
                    self.milestones[:, :-1] - self.milestones[:, 1:], axis=-1
                )
            assert milestone_distances.shape == (
                milestones.shape[0],
                milestones.shape[1] - 1,
            ), milestone_distances.shape
        self.milestone_distances = milestone_distances

        if seed_list is None:
            _rng = np.random.default_rng(seed=seed)
            seed_list = [_rng.integers(0, 2**31 - 1) for _ in range(num_tasks)]
        assert len(seed_list) == num_tasks
        self._num_tasks = num_tasks
        self.seed_list = seed_list
        self.save_video = save_video
        if save_video:
            assert save_video_kwargs is not None
            self.video_fps = save_video_kwargs["fps"]
            self.add_debug_text = save_video_kwargs["add_debug_text"]
            self.wandb_logging = save_video_kwargs["wandb_logging"]
            self.save_locally = save_video_kwargs["save_locally"]
            if self.save_locally:
                self.save_path = save_video_kwargs["save_path"]
        self.device = device

        self.batched_env: BaseVectorEnv | None = None
        self.use_milestone_compressor = use_milestone_compressor
        self.milestone_indices = milestone_indices  # B, N

    @staticmethod
    def confirm_embedding(
        embedding: np.ndarray | torch.Tensor, preprocessor: Preprocessor | None
    ) -> np.ndarray:
        """make sure the embedding is linear for calculating embed distance during inference
        Args:
            embedding:
                if embedding.ndim == 2: B, d already;
                if embedding.ndim == 3: B, L, d for milestone embeddings;
                if embedding.ndim == 4: B, H, W, 3 for raw rgb;
                                        or without pooling, e.g. B, 2048, 7, 7
                if embedding.ndim == 5: B, L, H, W, 3 for L milestones;
            preprocessor: frozen preprocessor
        Return:
            embedding.ndim = 2 for single step obs, ndim=3 for milestone embeddings
        """
        if preprocessor is None:
            return U.any_to_numpy(embedding)
        preprocessor_output_dim = preprocessor.output_dim
        if embedding.ndim not in [2, 3]:
            input_dims = embedding.shape
            if (
                embedding.ndim in [4, 5]
                and embedding.shape[-3:] != preprocessor_output_dim
            ):
                # raw rgb case (B, (L) H, W, 3)
                if embedding.ndim == 5:
                    embedding = einops.rearrange(embedding, "b l h w c -> (b l) h w c")
                with torch.no_grad():
                    embedding = preprocessor.process(
                        embedding,
                        # embedding.reshape((np.prod(input_dims[:-3]), *input_dims[-3:])),
                        reconstruct_linear=True,
                    )
            # if preprocessor.remove_pool:
            if (
                embedding.ndim in [4, 5]
                and embedding.shape[-3:] == preprocessor_output_dim
            ):
                U.rank_zero_print("DEPRECATED!", color="red")
                # no pooling case, e.g. (B, (L), 2048, 7, 7)
                embedding = embedding.reshape(
                    (np.prod(input_dims[:-3]), *preprocessor.output_dim)
                )
                with torch.no_grad():
                    embedding = F.adaptive_avg_pool2d(embedding, output_size=(1, 1))
                    embedding = torch.flatten(embedding, 1)
                    if preprocessor.preprocessor_fc is not None:
                        embedding = preprocessor.preprocessor_fc(embedding)
            embedding = embedding.reshape((*input_dims[:-3], embedding.shape[-1]))
        if torch.is_tensor(embedding):
            embedding = embedding.cpu().numpy()
        assert embedding.ndim in [2, 3], embedding.shape
        return embedding

    def rollout(self, policy: PolicyBase, mode: str, epoch: int) -> dict:
        use_kv_cache = hasattr(policy, "use_kv_cache") and policy.use_kv_cache
        is_causal = hasattr(policy, "causal") and policy.causal
        cache_obs = is_causal and not use_kv_cache
        if cache_obs:  # means causal and not use kv cache
            U.rank_zero_print(
                "WARNING: `use_kv_cache=False` NOT RECOMMEND SINCE SOOOOOO INEFFICIENT!",
                color="red",
            )
            self.inference_kwargs["cache_history"] = True
        elif is_causal and use_kv_cache:
            # keep the bs the same so not terminate the episode
            self.inference_kwargs["dummy_rtn"] = True

        if self.batched_env is None:
            if cache_obs:  # dynamic leading dim obs for history
                self.batched_env = SubprocVectorEnv(self._create_env_fns())
            else:
                self.batched_env = ShmemVectorEnv(self._create_env_fns())
            self.batched_env.reset()

        results = {k: None for k in range(self._num_tasks)}

        seed_list = list(self.seed_list)
        logging_videos = self.save_video
        if logging_videos:
            self.batched_env.set_env_attr("recording", True)
        for i, env_ids in enumerate(self.rollout_list):
            if logging_videos and i != 0 and not self.save_locally:
                # logging wandb only 1st iter
                self.batched_env.set_env_attr("recording", False)
                logging_videos = False

            if is_causal and use_kv_cache:
                policy.reset_cache()

            num_env_this_batch = len(env_ids)
            self.batched_env.seed(
                seed=[seed_list.pop() for _ in range(num_env_this_batch)]
            )
            self.prepare_states_before_reset(env_ids, global_idx=i)
            if self.milestone_distances is not None:
                milestone_distances_this_batch = self.milestone_distances[
                    self.num_envs * i : self.num_envs * i + num_env_this_batch, ...
                ].copy()
                self.batched_env.set_env_attr(
                    "milestone_distances",
                    milestone_distances_this_batch,
                    id=env_ids,
                    diff_value=True,
                )

            if self.no_robot_rgb_milestones is not None:
                no_robot_milestones_this_batch = self.no_robot_rgb_milestones[
                    self.num_envs * i : self.num_envs * i + num_env_this_batch, ...
                ].copy()
                # num_env, num_milestone, (H, W, 3 or D)
                no_robot_milestones_this_batch = self.confirm_embedding(
                    no_robot_milestones_this_batch, preprocessor=policy.preprocessor
                )

                self.batched_env.set_env_attr(
                    "no_robot_milestones",
                    no_robot_milestones_this_batch,
                    id=env_ids,
                    diff_value=True,
                )
            # num_env_this_batch, num_goals, ...
            milestones_this_batch = self.milestones[
                self.num_envs * i : self.num_envs * i + num_env_this_batch, ...
            ].copy()
            # set milestones for this episode
            self.batched_env.set_env_attr(
                "milestones", milestones_this_batch, id=env_ids, diff_value=True
            )

            if self.milestone_indices is not None:
                milestone_indices_this_batch = self.milestone_indices[
                    self.num_envs * i : self.num_envs * i + num_env_this_batch, ...
                ].copy()
                self.batched_env.set_env_attr(
                    "milestone_indices",
                    milestone_indices_this_batch,
                    id=env_ids,
                    diff_value=True,
                )

            if milestones_this_batch.ndim != 3:
                # num_env_this_batch, num_goals, 3, H, W
                if i == 0:
                    U.rank_zero_print(f"{milestones_this_batch.shape=}", color="blue")
                self.batched_env.set_env_attr(
                    "milestone_embeddings",
                    self.confirm_embedding(
                        milestones_this_batch.copy(), preprocessor=policy.preprocessor
                    ),
                    id=env_ids,
                    diff_value=True,
                )
            if logging_videos:
                rgb_milestones_this_batch = self.rgb_milestones[
                    num_env_this_batch * i : num_env_this_batch * (i + 1), ...
                ].copy()
                self.batched_env.set_env_attr(
                    "rgb_milestones",
                    rgb_milestones_this_batch,
                    id=env_ids,
                    diff_value=True,
                )
            # reset: num_env_this_batch, h, w, 3 (or dict of obs)
            obs = self.batched_env.reset(id=env_ids)
            assert obs.shape[0] == num_env_this_batch, obs.shape

            running_env_ids = np.arange(num_env_this_batch)

            for st in tqdm.tqdm(
                range(self.max_horizon),
                initial=1,
                desc=f"rank {U.get_local_rank()}: Rollout {i * self.num_envs + len(env_ids)}/{self._num_tasks}",
                leave=False,
            ):
                self.batched_env.set_env_attr(
                    "cur_milestone_idx", value=None, id=running_env_ids
                )  # for random skipping, sample maybe next goal idx inside env every step

                if cache_obs:
                    current_milestone = self.batched_env.get_env_attr(
                        "cached_prev_milestones", id=running_env_ids
                    )
                else:
                    current_milestone = self.batched_env.get_env_attr(
                        "current_milestone", id=running_env_ids
                    )
                current_milestone = U.any_to_numpy(current_milestone)
                U.assert_(len(current_milestone), len(obs))

                with torch.no_grad():
                    # only batchify here, keep obs as list as length of running_env_ids anywhere else
                    batchify_obs = U.batch_observations(obs, device=self.device)
                    if cache_obs:
                        b, t, *_ = current_milestone.shape
                        if st == 0 and batchify_obs["rgb"].shape[:2] != (b, t):
                            for k in batchify_obs:
                                batchify_obs[k] = batchify_obs[k][
                                    :, None, ...
                                ]  # broadcast T dim
                        assert batchify_obs["rgb"].shape[:2] == (b, t), (
                            batchify_obs["rgb"].shape,
                            b,
                            t,
                        )
                    elif is_causal:
                        # broadcast T dim, B H W 3 or B D
                        assert current_milestone.ndim in [2, 4], current_milestone.shape
                        current_milestone = current_milestone[:, None, ...]
                        for k in batchify_obs:
                            batchify_obs[k] = batchify_obs[k][:, None, ...]
                    # L, n
                    action, obs_embed, goal_embed = policy(
                        batchify_obs,
                        goal=current_milestone,
                        deterministic=True,
                        return_embeddings=True,
                        timesteps=torch.tensor(
                            [[st]], dtype=torch.int32, device=self.device
                        ),
                        input_pos=torch.tensor([st], device=self.device)
                        if is_causal and use_kv_cache
                        else None,
                    )
                    if action.ndim == 3:  # has T dim
                        action = action[:, -1, :]
                # switch milestones inside the inference wrapper based on current obs embedding
                if self.no_robot_rgb_milestones is not None:
                    no_robot_rgbs = self.batched_env.get_env_attr(
                        "current_no_robot_frame", id=running_env_ids
                    )
                    cur_obs_embed = self.confirm_embedding(
                        np.stack(no_robot_rgbs), preprocessor=policy.preprocessor
                    )
                else:
                    cur_obs_embed = self.confirm_embedding(
                        batchify_obs["rgb"] if obs_embed.ndim != 2 else obs_embed,
                        preprocessor=policy.preprocessor,
                    )
                self.batched_env.set_env_attr(
                    "current_obs_embedding",
                    cur_obs_embed,
                    id=running_env_ids,
                    diff_value=True,
                )

                #  next_obs (n, dict/(256, 256, 3)) (n,) (n,)，list[dict]
                obs, r, done, info = self.batched_env.step(
                    action.cpu().numpy(), id=running_env_ids
                )
                prev_running_env_ids = copy.deepcopy(running_env_ids)
                if np.any(done):
                    if is_causal and use_kv_cache:
                        assert np.all(done), done  # keep bs the same for kv cache
                    # update running env id & collect results for envs that done
                    terminated_env_local_idx = np.where(done)[0]
                    masks = np.ones_like(running_env_ids, dtype=bool)
                    masks[terminated_env_local_idx] = False
                    running_env_ids = running_env_ids[masks]
                    obs = obs[masks]
                    terminated_local_ids = list(
                        set(prev_running_env_ids) - set(running_env_ids)
                    )
                    metrics = self.batched_env.get_env_attr(
                        "metrics", id=terminated_local_ids
                    )
                    try:
                        task_names = self.batched_env.get_env_attr(
                            "task_name", id=terminated_local_ids
                        )
                    except AttributeError:
                        task_names = None
                    for met_idx, local_id in enumerate(terminated_local_ids):
                        assert (
                            local_id not in running_env_ids
                        ), f"{local_id=} is done but still in {running_env_ids=}"
                        assert results[local_id + self.num_envs * i] is None, (
                            f"{local_id + self.num_envs * i}-th task should be only rollout once, "
                            f"{local_id=}, {running_env_ids=}, {prev_running_env_ids=}, {terminated_local_ids=}"
                        )
                        results[local_id + self.num_envs * i] = {
                            (
                                k
                                if ("success" not in k and "completed_tasks" not in k)
                                or task_names is None
                                else f"{k}/{task_names[met_idx]}"
                            ): v
                            for k, v in metrics[met_idx].items()
                        }

                if np.all(done):
                    assert results is not None
                    break

            # only save first num vec envs videos
            if logging_videos:  # gather videos after all env done
                # each video may have diff length among diff envs
                all_frames = self.batched_env.get_env_attr("frames", id=env_ids)
                try:
                    task_names = self.batched_env.get_env_attr("task_name", id=env_ids)
                except AttributeError:
                    task_names = None
                try:
                    self.logging_videos(
                        all_frames,
                        task_names=task_names,
                        global_task_ids=[
                            self.num_envs * i + _ for _ in range(num_env_this_batch)
                        ],
                        mode=mode,
                        global_step=epoch,
                    )
                except:
                    pass

        return self.process_results(results, mode=mode)

    def prepare_states_before_reset(self, local_env_ids: list[int], global_idx: int):
        if self.env_name == "franka_kitchen":
            reset_states = [
                dict(
                    init_qpos=self.reset_kwargs["init_qpos"][
                        env_id + self.num_envs * global_idx
                    ],
                    init_qvel=self.reset_kwargs["init_qvel"][
                        env_id + self.num_envs * global_idx
                    ],
                    task_elements=self.reset_kwargs["task_elements"][
                        env_id + self.num_envs * global_idx
                    ],
                )
                for env_id in local_env_ids
            ]
            self.batched_env.set_env_attr(
                "reset_states", reset_states, id=local_env_ids, diff_value=True
            )
        else:
            raise NotImplementedError(self.env_name)

    def process_results(self, results: dict, mode: str) -> dict:
        res = {}
        successes = []
        completions = []
        for result in results.values():
            for k, v in result.items():
                if "success" in k:
                    successes.append(v)
                elif "completed_tasks" in k:
                    completions.append(v)
                res.setdefault(k, []).append(v)
        assert (
            len(successes) == self._num_tasks
        ), f"{len(successes)=} != {self._num_tasks}"
        return {
            f"{mode}/success": np.mean(successes),
            f"{mode}/completed_tasks": np.mean(completions),
            f"{mode}/num_tasks": float(self._num_tasks),
            **{f"{mode}/{k}": np.mean(v) for k, v in res.items()},
        }

    def logging_videos(
        self,
        all_frames: list,
        *,
        task_names: str | None = None,
        global_task_ids: list,
        mode: str,
        global_step: int,
    ):
        cur_rank = U.get_local_rank()
        for i, ids in enumerate(global_task_ids):
            # wandb channel-first logging: episode_length, h, w, 3 -> ..., 3, h, w
            cur_video = U.any_to_numpy(all_frames[i])
            if cur_video.ndim != 4 or cur_video.shape[-1] != 3:
                U.rank_zero_print(
                    f"cur_video has unexpected shape {cur_video.shape}", color="red"
                )
                continue
            cur_video = cur_video.transpose([0, 3, 1, 2])
            assert cur_video.ndim == 4 and cur_video.shape[1] == 3, cur_video.shape
            if self.wandb_logging and cur_rank == 0:
                log_name = U.f_join(
                    f"{mode}", "rollout-videos", f"{task_names[i]}" or "", f"{ids}"
                )
                wandb.log(
                    {
                        log_name: wandb.Video(
                            cur_video, fps=self.video_fps, format="mp4"
                        )
                    },
                )
            if self.save_locally:
                save_path = U.f_mkdir(
                    self.save_path,
                    f"{mode}-rank_{cur_rank}"
                    + f"/{task_names[i] if task_names is not None else ''}",
                )
                U.save_video(
                    cur_video,
                    U.f_join(save_path, f"env_{ids}_epoch_{global_step}.mp4"),
                    fps=self.video_fps,
                )

    def _create_env_fns(self) -> list:
        def _create_env(env_kwargs: dict) -> gym.Env:
            if self.env_name == "franka_kitchen":
                # don't reset before creating vector env
                env = KitchenBase(frame_height=224, frame_width=224, **env_kwargs)
                if self.decomp_method is not None:
                    env.COMPLETE_IN_ANY_ORDER = True
                    env.TERMINATE_ON_WRONG_COMPLETE = False
                else:
                    env.COMPLETE_IN_ANY_ORDER = False
                    env.TERMINATE_ON_WRONG_COMPLETE = True
            else:
                raise NotImplementedError(self.env_name)
            env = InferenceWrapper(env, **self.inference_kwargs)
            if self.save_video:
                env = VisualizeWrapper(
                    env, add_goal=True, add_debug_text=self.add_debug_text
                )
                env.recording = True
            return env

        env_kwargs = dict(
            max_horizon=self.max_horizon,
            gpu_id=int(torch.device(self.device).index),
        )

        return [lambda: _create_env(env_kwargs) for _ in range(self.num_envs)]

    def close(self):
        self.batched_env.close()

    @staticmethod
    def generate_rollout_list(num_envs: int, num_tasks: int):
        assert num_envs <= num_tasks
        tasks = []
        k = 0
        for i in range(num_tasks):
            sub_list = []
            j = 0
            while j < num_envs and k < num_tasks:
                sub_list.append(j)
                j += 1
                k += 1
            tasks.append(sub_list)
            if k >= num_tasks:
                break
        return tasks
