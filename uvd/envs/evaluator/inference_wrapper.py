from __future__ import annotations

import random

import gym
import numpy as np

import uvd.utils as U
from uvd.envs.franka_kitchen import KitchenBase

__all__ = ["InferenceWrapper"]


class InferenceWrapper(gym.Wrapper):
    """Set all milestones => reset => set current_embedding_distance => step =>
    set current_embedding_distance => ...

    current subgoal milestone should not be set but automatically be
    changed after setting current embedding distance
    """

    def __init__(
        self,
        env: gym.Env,
        thresh: float = 0.2,
        hybrid: bool = False,
        use_milestone_compressor: bool = False,
        random_skip_ratio: float = 0.0,
        delay_steps: int | None = 0,
        cache_history: bool = False,
        dummy_rtn: bool = False,
    ):
        super().__init__(env=env)
        self._thresh = thresh if thresh is not None else np.inf
        self._random_skip_ratio = random_skip_ratio
        self._cur_milestone_idx = None
        self._delay_steps = delay_steps or 0
        self._n_steps_after_pass_thresh = -1

        self.num_milestones = None
        self._milestones = None
        self._current_milestone = None
        self._current_obs_embedding = None
        self._achieved_goals = 0
        self._current_embedding_distance = None
        self._unnormalized_embedding_distance = None
        self.milestone_distances = None

        self._no_robot_milestones = None
        self._current_no_robot_milestone = None

        self._milestone_embeddings = None
        self._current_milestone_embedding = None

        self._subgoal_init_distance = None

        self.hybrid_phase = 0 if hybrid else None
        self._cur_rgb_frame = None
        # U.rank_zero_print(f"{use_milestone_compressor=}", color="blue")
        self.use_milestone_compressor = use_milestone_compressor
        self._n_steps = 0
        self._n_steps_this_milestone = 0
        self._milestone_indices = None
        self.cache_history = cache_history
        if cache_history:
            self.cached_obs = []
            self.cached_prev_milestones = []
        else:
            self.cached_obs = None
            self.cached_prev_milestones = None
        self._has_dummy_reset = True

        if dummy_rtn:
            self.dummy_rtn = []
        else:
            self.dummy_rtn = None

    def dummy_step(self) -> tuple:
        self._n_steps += 1
        return self.dummy_rtn

    def step(self, action: np.ndarray) -> tuple:
        if (
            self.dummy_rtn is not None
            and len(self.dummy_rtn) > 0
            and self._n_steps + 1 >= self.env.max_horizon
        ):
            o, r, _, i = self.dummy_rtn
            self.dummy_rtn = []
            return o, r, True, i
        elif self.dummy_rtn is not None and len(self.dummy_rtn) > 0:
            return self.dummy_step()

        o, r, d, i = super().step(action)
        self._n_steps += 1
        self._n_steps_this_milestone += 1
        o = self.proc_obs(o)
        self._cur_rgb_frame = o.get("rgb", None)
        # switch delay triggered
        if self._n_steps_after_pass_thresh >= 0:
            self._n_steps_after_pass_thresh += 1

        if d and self.dummy_rtn is not None:
            self.dummy_rtn = (o, r, False, i)
            d = self._n_steps >= self.env.max_horizon

        if d:
            # force set every episode
            self.milestones = None
            if self.dummy_rtn is not None:
                self.dummy_rtn = []
        return o, r, d, i

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if self.cached_obs is not None:
            self.cached_obs = []
            self.cached_prev_milestones = []
        obs = self.proc_obs(obs)
        self._achieved_goals = 0
        self._n_steps_after_pass_thresh = -1
        self._subgoal_init_distance = None
        self.hybrid_phase = 0 if self.hybrid_phase is not None else None
        self._cur_rgb_frame = obs.get("rgb", None)
        self._n_steps = 0
        self._n_steps_this_milestone = 0
        self._has_dummy_reset = False
        return obs

    def proc_obs(self, obs: dict):
        if (
            self.cached_obs is not None and not self._has_dummy_reset
        ):  # in case dummy reset call
            # this fn should be call only once for every step and reset
            self.cached_obs.append(obs)
            # steps, ...
            obs = U.batch_observations(self.cached_obs, to_tensor=False)
            self.cached_prev_milestones.append(self.current_milestone)
        return obs

    @property
    def completed_goals(self):
        return self._achieved_goals

    @property
    def milestones(self) -> np.ndarray:
        assert (
            self._milestones is not None
        ), f"must set before using, step: {self.env.episode_length}"
        return self._milestones

    @milestones.setter
    def milestones(self, milestones: np.ndarray | None):
        """Set outside.

        can be rgb or embedding
        """
        if milestones is not None:
            self._milestones = milestones
            self.num_milestones = len(milestones)
            # set initial sub goal
            self._current_milestone = self._milestones[0]
        else:
            self._milestones = None
            self._current_milestone = None
            self._subgoal_init_distance = None

    @property
    def milestone_embeddings(self) -> np.ndarray:
        if self.milestones.ndim == 2:
            # preprocessed embedding already
            return self.milestones
        assert self._milestone_embeddings is not None
        return self._milestone_embeddings

    @milestone_embeddings.setter
    def milestone_embeddings(self, embeddings: np.ndarray | None):
        if embeddings is not None:
            assert embeddings.ndim == 2, embeddings.shape
            self._milestone_embeddings = embeddings
            self._current_milestone_embedding = self.milestone_embeddings[0]
        else:
            self._milestone_embeddings = self._current_milestone_embedding = None

    @property
    def no_robot_milestones(self) -> np.ndarray:
        return self._no_robot_milestones

    @no_robot_milestones.setter
    def no_robot_milestones(self, milestones: np.ndarray | None):
        """If set (outside), sub-goal switches will depend on this, while the
        rollout inference is still based on the goal embed with arm."""
        if milestones is not None:
            self._no_robot_milestones = milestones.copy()
            self._current_no_robot_milestone = milestones[0]
        else:
            self._no_robot_milestones = None
            self._current_no_robot_milestone = None
            self._subgoal_init_distance = None

    @property
    def cur_milestone_idx(self) -> int | None:
        return self._cur_milestone_idx

    @cur_milestone_idx.setter
    def cur_milestone_idx(self, idx: int | None):
        self._cur_milestone_idx = idx

    def maybe_random_skip(self, cur, all_cur) -> np.ndarray:
        if self._random_skip_ratio > 0:
            if (
                self._cur_milestone_idx is None
                and self._random_skip_ratio > random.random()
            ):
                self.cur_milestone_idx = min(
                    self._achieved_goals + 1, self.num_milestones - 1
                )
            elif self._cur_milestone_idx is None:
                self.cur_milestone_idx = min(
                    self._achieved_goals, self.num_milestones - 1
                )
            assert (
                self.cur_milestone_idx
                in [
                    self._achieved_goals,
                    self._achieved_goals + 1,
                    self.num_milestones - 1,
                ]
                and self.cur_milestone_idx < self.num_milestones
            ), f"WARN: {self.cur_milestone_idx}, {self._achieved_goals}"
            return all_cur[self._cur_milestone_idx]
        self._cur_milestone_idx = min(self._achieved_goals, self.num_milestones - 1)
        return cur

    @property
    def current_milestone(self) -> np.ndarray:
        if self.use_milestone_compressor:
            return self.milestone_embeddings
        # current sub-goal, switch automatically when set embedding distance
        assert self._current_milestone is not None
        return self.maybe_random_skip(self._current_milestone, self._milestones)

    @property
    def current_milestone_embedding(self) -> np.ndarray:
        if self.current_milestone.ndim == 1:
            return self.current_milestone
        assert self._current_milestone_embedding is not None
        return self.maybe_random_skip(
            self._current_milestone_embedding, self._milestone_embeddings
        )

    def get_current_milestone_embedding_without_skip(self) -> np.ndarray:
        if self.current_milestone.ndim == 1:
            return self._current_milestone
        assert self._current_milestone_embedding is not None
        return self._current_milestone_embedding

    @property
    def current_no_robot_milestone(self) -> np.ndarray | None:
        if self._current_no_robot_milestone is not None:
            assert (
                self._current_no_robot_milestone.ndim == 1
            ), self.current_milestone.shape
        # no need for random skip
        return self._current_no_robot_milestone

    @property
    def current_embedding_distance(self) -> np.ndarray:
        return self._current_embedding_distance

    @current_embedding_distance.setter
    def current_embedding_distance(self, dist: float | None):
        if dist is None or self._milestones is None:
            self._current_embedding_distance = dist
            return
        switch = False
        if dist is not None and dist < self._thresh:
            switch = True
            if self._delay_steps > 0 and self._n_steps_after_pass_thresh == -1:
                switch = False
                # trigger the delay
                self._n_steps_after_pass_thresh = 0
        if 0 < self._delay_steps <= self._n_steps_after_pass_thresh:
            assert self._delay_steps == self._n_steps_after_pass_thresh
            switch = True
            self._n_steps_after_pass_thresh = -1

        current_milestone_budget = self.current_milestone_budget()
        if switch and current_milestone_budget is not None:
            if current_milestone_budget > self._n_steps_this_milestone + 3:
                switch = False  # make sure not switch too fast by noise of the embed
        elif not switch and current_milestone_budget is not None:
            if current_milestone_budget < self._n_steps_this_milestone - 3:
                switch = True  # in case not switch due to noise of the embed

        if switch:
            self._n_steps_this_milestone = 0
            if self.hybrid_phase is not None:
                self.hybrid_phase += 1
            # switch milestones
            self._achieved_goals += 1
            self._achieved_goals = min(self._achieved_goals, self.num_milestones)
            # self._achieved_goals = self.env.num_goals_achieved
            self._cur_milestone_idx = min(
                self._achieved_goals, self.num_milestones - 1
            )  # next idx
            self._current_milestone = self._milestones[self._cur_milestone_idx]
            if self._current_milestone_embedding is not None:
                # switch no next milestone embedding for conditioning
                self._current_milestone_embedding = self.milestone_embeddings[
                    self._cur_milestone_idx
                ]
            if self.current_no_robot_milestone is not None:
                # switch to next milestone for calculating embed distance
                assert len(self.milestones) == len(self.no_robot_milestones)
                self._current_no_robot_milestone = self.no_robot_milestones[
                    self._cur_milestone_idx
                ]
                if self.milestone_distances is not None:
                    self._subgoal_init_distance = self.milestone_distances[
                        min(self._achieved_goals - 1, self.num_milestones - 2)
                    ]
                else:
                    # use no skipping milestone for distance check
                    # self._subgoal_init_distance = np.linalg.norm(
                    #     self.current_obs_embedding - self._current_no_robot_milestone
                    # )
                    self._subgoal_init_distance = None  # set next step back
            else:
                if self.milestone_distances is not None:
                    # self._subgoal_init_distance = self.milestone_distances[
                    #     min(self._achieved_goals - 1, self.num_milestones - 2)
                    # ]
                    self._subgoal_init_distance = None
                else:
                    # use no skipping milestone for distance check
                    self._subgoal_init_distance = np.linalg.norm(
                        self.current_obs_embedding - self._current_milestone_embedding
                    )
            self._current_embedding_distance = 1.0
        else:
            self._current_embedding_distance = dist

    @property
    def current_obs_embedding(self) -> np.ndarray:
        assert self._current_obs_embedding is not None
        return self._current_obs_embedding

    @current_obs_embedding.setter
    def current_obs_embedding(self, embedding: np.ndarray):
        """Set outside, should be embedding already for calculating embed
        distance."""
        assert embedding.ndim == 1, embedding.shape
        self._current_obs_embedding = embedding
        if self.hybrid_phase is None:
            # calculate embed dist automatically after setting embedding
            # use no skipping milestone for distance check
            cur_milestone = (
                self.current_no_robot_milestone
                if self.current_no_robot_milestone is not None
                else self.get_current_milestone_embedding_without_skip()
            )
        else:
            cur_milestone = (
                self.get_current_milestone_embedding_without_skip()
                if self.hybrid_phase % 2 == 0
                else self.current_no_robot_milestone
            )
        assert (
            cur_milestone.ndim == embedding.ndim == 1
        ), f"{cur_milestone.shape} != {embedding.shape} != (n,)"
        embed_distance = np.linalg.norm(embedding - cur_milestone)
        self._unnormalized_embedding_distance = embed_distance

        if self._subgoal_init_distance is None:
            # first step after reset here
            self._subgoal_init_distance = max(embed_distance, 1e-14)
        # property setter
        self.current_embedding_distance = embed_distance / self._subgoal_init_distance

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
        self.env: KitchenBase
        metrics = self.env.metrics
        achieved = (
            self.completed_goals if not metrics["success"] else self.num_milestones
        )
        achieved_percentage = achieved / self.num_milestones
        metrics["achieved_milestones"] = achieved_percentage
        return metrics

    @property
    def current_no_robot_frame(self) -> np.ndarray:
        if self.hybrid_phase is not None and self.hybrid_phase % 2 == 0:
            # hack for hybrid return rgb again
            assert self._cur_rgb_frame is not None
            return self._cur_rgb_frame
        else:
            return self.env.current_no_robot_frame

    @property
    def current_rgb_frame(self) -> np.ndarray:
        return self._cur_rgb_frame

    @property
    def milestone_indices(self):
        return self._milestone_indices

    @milestone_indices.setter
    def milestone_indices(self, milestone_indices: np.ndarray | None):
        self._milestone_indices = milestone_indices

    def current_milestone_budget(self) -> int | None:
        if self.cur_milestone_idx is None:
            self.cur_milestone_idx = 0
        if self.milestone_indices is None:
            return None
        return (
            self.milestone_indices[self.cur_milestone_idx] + 1  # indices start from 0
            if self.cur_milestone_idx == 0
            else self.milestone_indices[self.cur_milestone_idx]
            - self.milestone_indices[self.cur_milestone_idx - 1]
        )
