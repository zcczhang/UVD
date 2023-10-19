from __future__ import annotations

from collections import OrderedDict

import gym
import numpy as np

import uvd.utils as U
from uvd.envs.evaluator.inference_wrapper import InferenceWrapper

__all__ = ["VisualizeWrapper"]


class VisualizeWrapper(gym.Wrapper):
    def __init__(
        self, env: InferenceWrapper, add_goal: bool = True, add_debug_text: bool = True
    ):
        super().__init__(env=env)
        self.env: InferenceWrapper = env
        self.add_goal = add_goal
        self.add_debug_text = add_debug_text
        self._frames = []
        self._rgb_milestones = None
        self._current_goal_image = None
        self._stepwise_extra_debug_texts = None

        self._is_init_reset = True
        self._recording = True

    @property
    def recording(self) -> bool:
        return self._recording

    @recording.setter
    def recording(self, val: bool):
        self._recording = val

    @property
    def frames(self):
        return self._frames

    def clear(self):
        self._frames = []

    @property
    def video_length(self):
        return len(self._frames)

    def reset(self, **kwargs) -> np.ndarray:
        obs = super().reset(**kwargs)
        self.clear()
        if self._is_init_reset:
            self._is_init_reset = False
            return obs
        self.add_frame(obs, info=None)
        return obs

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        o, r, d, i = super().step(action)
        self.add_frame(o, i)
        self.current_embedding_distance = None
        self.stepwise_extra_debug_texts = None
        return o, r, d, i

    def add_frame(
        self, obs: np.ndarray | OrderedDict[str, np.ndarray], info: dict | None
    ):
        if not self.recording:
            return
        if not isinstance(obs, np.ndarray):
            assert isinstance(obs, (dict, OrderedDict)) and "rgb" in obs, obs
            if self.env.cached_obs is not None:
                rgb = obs["rgb"][-1, ...].copy()
            else:
                rgb = obs["rgb"].copy()
        else:
            rgb = obs.copy()
        if self.add_goal:
            rgb = np.concatenate([rgb, self.current_goal_image], axis=-2)
        if self.add_debug_text:
            if info is not None:
                assert (
                    self.current_embedding_distance is not None
                ), f"should set before each step"
                txts = dict(
                    complete_milestones=self.completed_goals,
                    num_milstones=self.env.num_milestones,
                    current_embedding_distance=float(self.current_embedding_distance),
                )
                try:
                    txts.update(
                        current_sub_task=info["current_sub_task"],
                        step=info["episode_length"],
                        completed_tasks=info["completed_tasks"],
                        # success=int(info["success"]),
                    )
                    if "distances_left" in info["rewards"]:
                        distances_left = info["rewards"]["distances_left"]
                        if isinstance(distances_left, dict):
                            for ele, dist in distances_left.items():
                                txts[f"low_dim_distances_left/{ele}"] = float(
                                    np.sum(dist)
                                )
                        else:
                            txts["low_dim_distances_left"] = float(
                                np.sum(distances_left)
                            )
                except KeyError:
                    pass
            else:
                try:
                    current_sub_task = getattr(self.env, "current_subtask")
                    txts = dict(current_sub_task=current_sub_task, step=0)
                except AttributeError:
                    txts = dict(step=0)
            if self.stepwise_extra_debug_texts is not None:
                txts.update(self.stepwise_extra_debug_texts)
            rgb = U.debug_texts_to_frame(frame=rgb, debug_text=txts)
        self._frames.append(rgb)

    @property
    def current_goal_image(self):
        assert self._current_goal_image is not None
        if self.cur_milestone_idx is not None:
            return self.rgb_milestones[self.cur_milestone_idx]
        return self._current_goal_image

    @current_goal_image.setter
    def current_goal_image(self, image: np.ndarray | None):
        if image is not None:
            assert image.ndim == 3, image.shape
        self._current_goal_image = image

    @property
    def rgb_milestones(self) -> np.ndarray:
        assert self._rgb_milestones is not None, f"must set before using"
        return self._rgb_milestones

    @rgb_milestones.setter
    def rgb_milestones(self, rgbs: np.ndarray | None):
        """Set outside."""
        if rgbs is not None:
            # num_goal, h, w, 3
            assert (
                rgbs.ndim == 4 and rgbs.shape[0] == self.milestones.shape[0]
            ), rgbs.shape
            self._rgb_milestones = rgbs.copy()
            self._current_goal_image = self._rgb_milestones[0]
        else:
            self._rgb_milestones = None
            self._current_goal_image = None

    @property
    def stepwise_extra_debug_texts(self) -> dict | None:
        return self._stepwise_extra_debug_texts

    @stepwise_extra_debug_texts.setter
    def stepwise_extra_debug_texts(self, texts: dict | None):
        self._stepwise_extra_debug_texts = texts

    @property
    def milestones(self) -> np.ndarray:
        return self.env.milestones

    @milestones.setter
    def milestones(self, milestones: np.ndarray | None):
        self.env.milestones = milestones

    @property
    def milestone_embeddings(self) -> np.ndarray:
        return self.env.milestone_embeddings

    @milestone_embeddings.setter
    def milestone_embeddings(self, embeddings: np.ndarray | None):
        self.env.milestone_embeddings = embeddings

    @property
    def no_robot_milestones(self) -> np.ndarray:
        return self.env.no_robot_milestones

    @no_robot_milestones.setter
    def no_robot_milestones(self, milestones: np.ndarray | None):
        self.env.no_robot_milestones = milestones

    @property
    def current_milestone(self) -> np.ndarray:
        return self.env.current_milestone

    @property
    def current_no_robot_milestone(self) -> np.ndarray | None:
        return self.env.no_robot_milestones

    @property
    def current_embedding_distance(self) -> np.ndarray:
        return self.env.current_embedding_distance

    @current_embedding_distance.setter
    def current_embedding_distance(self, dist: float | None):
        self.env.current_embedding_distance = dist

    @property
    def current_obs_embedding(self) -> np.ndarray:
        return self.env.current_obs_embedding

    @current_obs_embedding.setter
    def current_obs_embedding(self, embedding: np.ndarray):
        before_set_achieved = self.completed_goals
        self.env.current_obs_embedding = embedding
        after_set_achieved = self.completed_goals
        # if after_set_achieved > before_set_achieved and self.recording:
        if self.recording:
            # switch rgb current goal image
            self.current_goal_image = self.rgb_milestones[
                min(after_set_achieved, len(self.rgb_milestones) - 1)
            ]

    @property
    def cur_milestone_idx(self) -> int | None:
        return self.env.cur_milestone_idx

    @cur_milestone_idx.setter
    def cur_milestone_idx(self, idx: int | None):
        self.env.cur_milestone_idx = idx

    @property
    def completed_goals(self) -> int:
        return self.env.completed_goals

    @property
    def reset_states(self) -> dict:
        return self.env.reset_states

    @reset_states.setter
    def reset_states(self, state: dict):
        self.env.reset_states = state

    @property
    def task_name(self) -> str:
        return self.env.task_name

    @property
    def metrics(self) -> dict:
        return self.env.metrics

    @property
    def milestone_distances(self) -> np.ndarray | None:
        return self.env.milestone_distances

    @milestone_distances.setter
    def milestone_distances(self, milestone_distances: np.ndarray | None):
        self.env.milestone_distances = milestone_distances

    @property
    def current_no_robot_frame(self) -> np.ndarray:
        return self.env.current_no_robot_frame

    @property
    def current_rgb_frame(self) -> np.ndarray:
        return self.env.current_rgb_frame

    @property
    def num_milestones(self):
        return self.env.num_milestones

    @property
    def milestone_indices(self):
        return self.env.milestone_indices

    @milestone_indices.setter
    def milestone_indices(self, milestone_indices: np.ndarray | None):
        self.env.milestone_indices = milestone_indices
