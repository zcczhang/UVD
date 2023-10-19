"""Environments using kitchen and Franka robot."""
from __future__ import annotations

import copy
from collections import OrderedDict

import gym
import numpy as np


from uvd.envs.franka_kitchen.franka_kitchen_constants import *
from mujoco_py import MjRenderContextOffscreen

from adept_envs.franka.kitchen_multitask_v0 import KitchenV0
from adept_envs.simulation.renderer import RenderMode, DMRenderer

__all__ = ["KitchenBase"]


class KitchenBase(KitchenV0):
    ALL_TASKS = FRANKA_KITCHEN_ALL_TASKS
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    TERMINATE_ON_WRONG_COMPLETE = False
    COMPLETE_IN_ANY_ORDER = False

    frame_height = 224
    frame_width = 224

    def __init__(
        self,
        *,
        task_elements: list | None = None,
        reward_config: dict | None = FRANKA_KITCHEN_REWARD_CONFIG,
        goal_masking: bool = True,
        obs_keys: str | tuple[str, ...] = ("rgb", "proprio"),
        max_horizon: int = 1000,
        frame_height: int = 224,
        frame_width: int = 224,
        robot_params: dict | None = None,
        frame_skip: int = 40,
        gpu_id: int = -1,
    ):
        task_elements = (
            list(task_elements) if task_elements is not None else list(self.ALL_TASKS)
        )
        self._task_elements = [
            e if isinstance(e, str) else IDX_TO_ELEMENT[e] for e in task_elements
        ]
        self.goal_masking = goal_masking
        # workaround for vector env preventing render
        self._init_step = True
        super(KitchenBase, self).__init__(
            robot_params=robot_params or {}, frame_skip=frame_skip
        )
        self._num_goals_achieved = 0
        self.tasks_to_complete = copy.deepcopy(self._task_elements)
        self.reward_config = reward_config
        self.obs_keys = (
            tuple(obs_keys) if not isinstance(obs_keys, str) else (obs_keys,)
        )
        dict_obs_space = OrderedDict()
        for key in obs_keys:
            if key == "rgb":
                dict_obs_space[key] = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(frame_height, frame_width, 3),
                    dtype=np.uint8,
                )
            elif key == "proprio":
                dict_obs_space[key] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.obs_dict["qp"].shape[0],),
                    dtype=np.float32,
                )
            else:
                raise NotImplementedError(key)
        self.observation_space = gym.spaces.Dict(dict_obs_space)
        self.max_horizon = max_horizon
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.gpu_id = gpu_id
        self._renderer_setup = False
        self.env_info = None
        self.episode_length = 0

    @property
    def task_elements(self) -> list[str]:
        return self._task_elements

    @property
    def task_name(self) -> str:
        return "-".join(self.task_elements).replace(" ", "_")

    @property
    def current_subtask(self) -> str:
        if self.num_goals_achieved == len(self.task_elements):
            return "null"  # success
        return self.task_elements[self.num_goals_achieved]

    @property
    def metrics(self):
        """Accessible metrics for vector env."""
        assert self.env_info is not None, f"try to call metrics without stepping"
        metrics = dict(
            episode_length=self.episode_length,
            success=int(self.env_info["success"]),
            completed_tasks=self.env_info["completed_tasks"],
        )
        return metrics

    def _get_task_goal(
        self, task: list[str] | None = None, actually_return_goal: bool = True
    ) -> np.ndarray:
        if task is None:
            task = self.task_elements
        new_goal = np.zeros_like(self.goal)
        if self.goal_masking and not actually_return_goal:
            return new_goal
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal
        return new_goal

    def reset(
        self,
        init_qpos: np.ndarray | None = None,
        init_qvel: np.ndarray | None = None,
        task_elements: list[str] | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> OrderedDict[str, np.ndarray]:
        """Return visual obs."""
        if seed is not None:
            self.seed(seed)
        # set initial states if not None
        self.reset_states = dict(
            init_qpos=init_qpos, init_qvel=init_qvel, task_elements=task_elements
        )
        assert len(self.task_elements) > 0
        self.tasks_to_complete = copy.deepcopy(self.task_elements)
        self._num_goals_achieved = 0

        self.env_info = None
        self.episode_length = 0

        super().reset()
        return self.get_dict_obs(**kwargs)

    def get_dict_obs(self, **kwargs):
        """Observation sensor for dict observation space."""
        obs = OrderedDict()
        for k in self.obs_keys:
            if k == "rgb":
                if "frame_height" in kwargs:
                    self.frame_height = kwargs.pop("frame_height")
                if "frame_width" in kwargs:
                    self.frame_width = kwargs.pop("frame_width")
                obs[k] = self.render(
                    mode="rgb_array", height=self.frame_height, width=self.frame_width
                )
            elif k == "proprio":
                obs[k] = self.obs_dict["qp"]
            else:
                raise NotImplementedError(k)
        return obs

    @property
    def reset_states(self) -> dict:
        return dict(
            init_qpos=self.init_qpos,
            init_qvel=self.init_qvel,
            task_elements=self.task_elements,
        )

    @reset_states.setter
    def reset_states(self, state: dict | None):
        """Set state in vector env for convenience."""
        if state.get("init_qpos", None) is not None:
            assert (
                state.get("init_qvel", None) is not None
            ), "qpos and qvel set together"
            qpos = self._squeeze_state_vec(state["init_qpos"].copy())
            qvel = self._squeeze_state_vec(state["init_qvel"].copy())
            self.set_state(qpos=qpos, qvel=qvel)
            self.init_qpos = self.data.qpos.ravel().copy()
            self.init_qvel = self.data.qvel.ravel().copy()
        if state.get("task_elements", None) is not None:
            self._task_elements = [
                e if isinstance(e, str) else IDX_TO_ELEMENT[e]
                for e in self._squeeze_state_vec(
                    state["task_elements"].copy(), to_list=True
                )
            ]

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
        if isinstance(self.sim_robot.renderer, DMRenderer):
            assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
            state = np.concatenate([qpos, qvel])
            self.sim.set_state(state)
            self.sim.forward()
        else:
            super().set_state(qpos=qpos, qvel=qvel)

    def state_vector(self) -> np.ndarray:
        if isinstance(self.sim_robot.renderer, DMRenderer):
            return self.sim.get_state()
        else:
            return super().state_vector()

    @staticmethod
    def _squeeze_state_vec(vec: np.ndarray, to_list: bool = False) -> np.ndarray | list:
        if not isinstance(vec, np.ndarray):
            return vec
        if vec.ndim != 1:
            vec = vec.reshape((vec.shape[-1],))
        if to_list:
            return vec.tolist()
        return vec

    def _get_reward_n_score(self, obs_dict: dict) -> tuple[dict, float]:
        """Score here means whether complete a new goal this step."""
        reward_dict = {"true_reward": 0.0}

        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        next_goal = self._get_task_goal(
            task=self.task_elements, actually_return_goal=True
        )  # obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        distances_dict = {}
        # # for element in self.tasks_to_complete:
        # make metrics consistent for sync in DDP rollout
        for i, element in enumerate(self.task_elements):
            element_idx = OBS_ELEMENT_INDICES[element]
            # distance = np.linalg.norm(
            #     next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx]
            # )
            distance = abs(
                next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx]
            )  # keep state dims
            distances_dict[element] = distance
            # complete = distance < BONUS_THRESH
            complete = distance < OBS_ELEMENT_THRESH[element]  # diff criteria
            if isinstance(complete, np.ndarray):
                complete = np.all(complete)

            if (
                self.REMOVE_TASKS_WHEN_COMPLETE
                and element not in self.tasks_to_complete
            ):
                # already achieved task(s), and
                # edge case for knobs that distances increasing later though rendered states not changed
                condition = True
            else:
                condition = (
                    complete and all_completed_so_far
                    if not self.COMPLETE_IN_ANY_ORDER
                    else complete
                )
            if condition:
                completions.append(element)
            all_completed_so_far = all_completed_so_far and condition
        prev_tasks_to_complete = list(self.tasks_to_complete)
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [
                self.tasks_to_complete.remove(element)
                for element in completions
                if element in self.tasks_to_complete
            ]
            num_goals_achieved = len(self.task_elements) - len(self.tasks_to_complete)
        else:
            num_goals_achieved = len(completions)

        complete_new_goal_this_step = False
        if num_goals_achieved > self._num_goals_achieved:
            self._num_goals_achieved += 1
            # edge case for knobs
            if num_goals_achieved != self._num_goals_achieved:
                self._num_goals_achieved = len(self.task_elements) - len(
                    self.tasks_to_complete
                )

            reward_dict["true_reward"] += self.reward_config["progress"]
            complete_new_goal_this_step = True

        if self.num_goals_achieved == len(self.task_elements):
            reward_dict["true_reward"] += self.reward_config["terminal"]

        reward_dict["distances_left"] = distances_dict
        score = int(complete_new_goal_this_step)
        return reward_dict, score

    @property
    def num_goals_achieved(self):
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            assert self._num_goals_achieved == len(self.task_elements) - len(
                self.tasks_to_complete
            ), f"{self._num_goals_achieved}, {self.task_elements}, {self.tasks_to_complete}"
        return self._num_goals_achieved

    def step(
        self, action: np.ndarray, **kwargs
    ) -> tuple[OrderedDict[str, np.ndarray], float, bool, dict]:
        action = np.clip(action, -1.0, 1.0)
        if not self.initializing:
            action = self.act_mid + action * self.act_amp  # mean center and scale
        else:
            self.goal = self._get_task_goal()  # update goal if init
        self.robot.step(self, action, step_duration=self.skip * self.model.opt.timestep)

        low_dim_obs = self._get_obs()
        # workaround for vector env
        if self._init_step:
            self._init_step = False
            return low_dim_obs, 0.0, False, {}

        self.episode_length += 1

        reward_dict, score = self._get_reward_n_score(self.obs_dict)
        success = (
            not self.tasks_to_complete
            if self.REMOVE_TASKS_WHEN_COMPLETE
            else self.num_goals_achieved == len(self.tasks_to_complete)
        )

        env_info = {
            "time": self.obs_dict["t"],
            "obs_dict": self.obs_dict,
            "rewards": reward_dict,
            "score": score,
            "success": success,
            # self.task_elements[:self.num_goals_achieved - 1]
            "completed_tasks": self.num_goals_achieved,
            "episode_length": self.episode_length,
            "current_sub_task": self.current_subtask,
        }
        done = self.episode_length >= self.max_horizon
        if self.TERMINATE_ON_TASK_COMPLETE and success:
            done = True
        if self.TERMINATE_ON_WRONG_COMPLETE:
            all_goal = self._get_task_goal(task=self.ALL_TASKS)
            for wrong_task in list(set(self.ALL_TASKS) - set(self.task_elements)):
                element_idx = OBS_ELEMENT_INDICES[wrong_task]
                distance = np.linalg.norm(
                    low_dim_obs[..., element_idx] - all_goal[element_idx]
                )
                complete = distance < BONUS_THRESH
                if complete:
                    done = True
                    break

        obs = self.get_dict_obs(**kwargs)
        env_info["done"] = done
        self.env_info = env_info
        return obs, reward_dict["true_reward"], done, env_info

    def close(self):
        super().close()
        self.close_env()

    def render(
        self,
        mode: str = "human",
        height: int | None = None,
        width: int | None = None,
        camera_id: int = -1,
        set_robot_alpha: float | None = None,
    ) -> np.ndarray | None:
        height = height or self.frame_height
        width = width or self.frame_width
        if not self._renderer_setup:
            self.set_gpu_id(gpu_id=self.gpu_id)
            camera_settings = dict(
                distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
            )
            self.sim_robot.renderer._camera_settings = camera_settings
        alpha_map = None
        if set_robot_alpha is not None:
            alpha_map = self.set_robot_alpha(alpha=set_robot_alpha)

        height = height or self.frame_height
        width = width or self.frame_width
        if mode in ["rgb_array", "rgb"]:
            frame = self.sim_robot.renderer.render_offscreen(
                width=width, height=height, mode=RenderMode.RGB, camera_id=camera_id
            )
        elif mode == "human":
            frame = self.sim_robot.renderer.render_to_window()
        else:
            frame = super(KitchenV0, self).render(
                mode=mode, height=height, width=width, camera_id=camera_id
            )
        if set_robot_alpha is not None:
            self.unset_robot_alpha(alpha_map)
        return frame

    def set_robot_alpha(self, alpha: float) -> dict[int, float]:
        alpha_map = dict()
        if not USE_DM_CONTROL:
            for name in self.sim.model.site_names:
                if "end_effector" in name:
                    orig_val = self.sim.model.site_rgba[
                        self.sim.model.site_name2id(name)
                    ][-1]
                    alpha_map[name] = max(orig_val, alpha_map.get(name, -1))
                    self.sim.model.site_rgba[self.sim.model.site_name2id(name)][
                        -1
                    ] = alpha
        for dof_id in self.sim.model.dof_jntid:
            alpha_map[dof_id] = max(
                self.sim.model.geom_rgba[dof_id][-1], alpha_map.get(dof_id, -1)
            )
            self.sim.model.geom_rgba[dof_id][-1] = alpha
        for dof_id in self.sim.model.dof_bodyid:
            alpha_map[dof_id] = max(
                self.sim.model.geom_rgba[dof_id][-1], alpha_map.get(dof_id, -1)
            )
            self.sim.model.geom_rgba[dof_id][-1] = alpha
        return alpha_map

    def unset_robot_alpha(self, alpha_map: dict[int, float]):
        for dof_id, alpha in alpha_map.items():
            if isinstance(dof_id, str) and "end_effector" in dof_id:
                self.sim.model.site_rgba[self.sim.model.site_name2id(dof_id)][
                    -1
                ] = alpha
            else:
                self.sim.model.geom_rgba[dof_id][-1] = alpha

    @property
    def current_no_robot_frame(self) -> np.ndarray:
        return self.render(mode="rgb_array", set_robot_alpha=0.0)

    def set_gpu_id(self, gpu_id: int):
        self._renderer_setup = True
        # if not USE_DM_CONTROL:
        if not isinstance(self.sim_robot.renderer, DMRenderer):
            self.sim_robot.renderer._offscreen_renderer = MjRenderContextOffscreen(
                self.sim_robot.renderer._sim, device_id=gpu_id
            )
