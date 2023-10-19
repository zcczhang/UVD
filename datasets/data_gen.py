from __future__ import annotations

import copy
import glob
import multiprocessing as mp
import shutil
from multiprocessing import get_logger
from typing import Any

import hydra
import imageio
import numpy as np
import tqdm

import uvd.utils as U
from uvd.envs.franka_kitchen import KitchenBase


def parse_task_name(task_name: str):
    sub_tasks = task_name.split("-")[1].split(",")
    parsed_tasks = []
    for sub_task in sub_tasks:
        if sub_task == "bottomknob":
            parsed_tasks.append("bottom burner")
        elif sub_task == "topknob":
            parsed_tasks.append("top burner")
        elif sub_task == "switch":
            parsed_tasks.append("light switch")
        elif sub_task == "slide":
            parsed_tasks.append("slide cabinet")
        elif sub_task == "hinge":
            parsed_tasks.append("hinge cabinet")
        else:
            parsed_tasks.append(sub_task)
    return "-".join(parsed_tasks)


def _generate_one_task_demo(
    *,
    raw_data_path: str,
    raw_task_dir: str,
    # same seeds for diff tasks for now
    seed_start: int,
    output_path: str,
    max_demos: int | None = None,
    frame_height: int = 256,
    frame_width: int = 256,
    terminal_if_done: bool = True,
    include_no_robot: bool = False,
    max_reject_sampling: int = 10,
    render: bool = False,
    save_mp4: bool = False,
    mp4_fps: int | None = None,
    copy_raw_data: bool = False,
    env: KitchenBase | None = None,
    gpu_id,
):
    task_path = U.f_join(raw_data_path, raw_task_dir)
    parsed_task_name = parse_task_name(raw_task_dir)
    trajectories = glob.glob(U.f_join(task_path, "*.pkl"))
    assert len(trajectories) > 0, task_path
    # include space, connect goals with "-"
    task_elements = parsed_task_name.split("-")
    n_demos = min(max_demos or len(trajectories), len(trajectories))
    saved_episode_dir = f"{parsed_task_name.replace(' ', '_')}"
    episode_path = U.f_mkdir(output_path, saved_episode_dir)

    close_env = False
    if env is None:
        env = KitchenBase(
            task_elements=task_elements,
            frame_height=frame_height,
            frame_width=frame_width,
            obs_keys=("rgb", "proprio"),
            gpu_id=gpu_id,
        )
        close_env = True

    num_demos_gen = 0
    seed = seed_start - 1
    traj_length_map = {}
    unsuccessful_traj = []
    for i, traj in tqdm.tqdm(
        enumerate(trajectories[:n_demos]), total=n_demos, desc=parsed_task_name
    ):
        data = U.load_pickle(traj)
        path = data["path"]
        actions = path["actions"]
        ctrls = data["ctrl"]
        raw_data_length, action_dim = actions.shape
        assert action_dim == env.action_space.shape[0]

        data_dict, info = {}, {}  # placeholders
        for _ in range(max_reject_sampling):
            seed += 1
            env.reset(init_qpos=data["qpos"][0], init_qvel=data["qvel"][0])
            obs = env.reset(task_elements=task_elements, seed=seed)
            obs_dict = env.obs_dict
            # slightly difference than directly using
            init_qpos = env.init_qpos.copy()
            init_qvel = env.init_qvel.copy()
            if render:
                env.render()
            data_dict: dict[str, Any] = {
                "reset_kwargs": dict(
                    task_elements=task_elements,
                    init_qpos=init_qpos,
                    init_qvel=init_qvel,
                ),
                "obs_full": [obs],  # {str: L+1, h, w, 3}
                "actions": [],  # L, A
                "rewards": [],  # L,
                "dones": [],  # L,
                "completed_tasks": [],  # L, x
                "scores": [],  # L,
                "seed": seed,
                "oracle_milestones": [],  # n h, w, 3
            }
            if include_no_robot:
                data_dict["no_robot_rgb"] = [
                    env.render(mode="rgb_array", set_robot_alpha=0.0)
                ]
            for s in tqdm.trange(raw_data_length):
                # Construct the action
                ctrl = (ctrls[s] - obs_dict["qp"]) / (
                    env.frame_skip * env.model.opt.timestep
                )
                act = (ctrl - env.act_mid) / env.act_amp
                act = np.clip(act, -1.0, 1.0)
                obs, reward, done, info = env.step(act)
                if render:
                    env.render()
                obs_dict = info["obs_dict"]
                # 1 if complete a new goal this step
                score = info["score"]
                # num goals achieved so far
                completed_tasks = info["completed_tasks"]
                if score == 1:
                    data_dict["oracle_milestones"].append(obs["rgb"])

                data_dict["obs_full"].append(obs)
                data_dict["actions"].append(np.array(act))
                data_dict["rewards"].append(float(reward))
                data_dict["dones"].append(int(done))
                data_dict["scores"].append(float(score))
                data_dict["completed_tasks"].append(np.array(completed_tasks))

                if include_no_robot:
                    data_dict["no_robot_rgb"].append(
                        env.render(mode="rgb_array", set_robot_alpha=0.0)
                    )

                if terminal_if_done and done:
                    break
            if len(env.tasks_to_complete) != 0:
                get_logger().warning(
                    f"reject sampling, with task left: {env.tasks_to_complete}, "
                    f"distance left: {info['rewards']['distances_left']}"
                )
                continue
            else:
                break
        if len(data_dict["dones"]) == 0 or data_dict["dones"][-1] != 1:
            # raise ValueError(traj)
            get_logger().warning(
                f"unsuccessful data {traj}, distance left: {info['rewards']['distances_left']}"
            )
            debug_dir = U.f_mkdir(U.f_join(episode_path, "reject_sampling"))
            # noinspection PyUnboundLocalVariable
            imageio.imsave(U.f_join(debug_dir, f"{task_elements}_{i}.png"), obs["rgb"])
            unsuccessful_traj.append(traj)
            continue

        traj_length = len(data_dict["dones"])
        traj_length_map[i] = traj_length
        # save accurate data
        data_dict["obs_full"] = U.batch_observations(
            data_dict["obs_full"], to_tensor=False
        )
        data_dict["actions"] = np.stack(data_dict["actions"])
        data_dict["rewards"] = np.array(data_dict["rewards"])
        data_dict["dones"] = np.array(data_dict["dones"])
        data_dict["scores"] = np.array(data_dict["scores"])
        data_dict["completed_tasks"] = np.array(data_dict["completed_tasks"])
        data_dict["ctrls"] = ctrls[:traj_length]
        data_dict["last_distances_to_goal"] = info["rewards"]["distances_left"]
        data_dict["oracle_milestones"] = np.stack(data_dict["oracle_milestones"])
        if include_no_robot:
            data_dict["no_robot_rgb"] = np.stack(data_dict["no_robot_rgb"])
            assert data_dict["obs_full"]["rgb"].shape == data_dict["no_robot_rgb"].shape

        assert (
            len(data_dict["obs_full"]["rgb"]) - 1
            == len(data_dict["obs_full"]["proprio"]) - 1
            == len(data_dict["actions"])
            == len(data_dict["rewards"])
            == len(data_dict["scores"])
            == len(data_dict["completed_tasks"])
            == len(data_dict["ctrls"])
            == traj_length
        )
        assert len(data_dict["oracle_milestones"]) == len(task_elements)

        num_demos_gen += 1
        U.save_pickle(data_dict, U.f_join(episode_path, f"episode_{i}.pkl"))
        if copy_raw_data:
            out_raw_path = U.f_mkdir(output_path, "raw_data", saved_episode_dir)
            shutil.copy(traj, U.f_join(out_raw_path, traj.split("/")[-1]))
        if save_mp4:
            video_dir = U.f_mkdir(U.f_join(episode_path, "videos"))
            U.save_video(
                video=data_dict["obs_full"]["rgb"],
                fname=U.f_join(video_dir, f"episode_{i}.mp4"),
                fps=mp4_fps,
                compress=True,
            )
            if include_no_robot:
                U.save_video(
                    video=data_dict["no_robot_rgb"],
                    fname=U.f_join(video_dir, f"episode_{i}_no_robot.mp4"),
                    fps=mp4_fps,
                    compress=True,
                )
    U.dump_json(
        dict(
            task_elements=task_elements,
            seed_start=seed_start,
            action_dim=env.action_space.shape[0],
            frame_height=frame_height,
            frame_width=frame_width,
            num_demos=num_demos_gen,
            trajectory_lengths=traj_length_map,
            unsuccessful_traj=unsuccessful_traj,
        ),
        episode_path,
        f"metadata.json",
    )
    if close_env:
        env.close()


def generate_one_task_demo(kwargs):
    _generate_one_task_demo(**kwargs)


@hydra.main(config_path=".", config_name="data_gen", version_base="1.1")
def main(cfg):
    if cfg.debug:
        cfg = copy.deepcopy(cfg)
        cfg.num_processes = 1
        cfg.output_path += "_debug"
        # cfg.render = True
        cfg.max_demos = 3
    U.ask_if_overwrite(cfg.output_path)
    if cfg.dm_backend or (cfg.render and (not cfg.mp or cfg.num_processes == 1)):
        import adept_envs.mujoco_env

        adept_envs.mujoco_env.USE_DM_CONTROL = True

    raw_data_path = cfg.raw_data_path
    raw_task_dirs = U.f_listdir(raw_data_path)
    assert raw_task_dirs
    if cfg.mp:
        num_processes = min(mp.cpu_count() - 2, cfg.num_processes or len(raw_task_dirs))
        _, num_gpus = U.parse_gpu_devices(cfg.gpus)
        with mp.Pool(num_processes) as pool:
            pool.map(
                generate_one_task_demo,
                [
                    dict(
                        raw_data_path=cfg.raw_data_path,
                        raw_task_dir=raw_task_dirs[i],
                        seed_start=cfg.seed_start,
                        output_path=cfg.output_path,
                        max_demos=cfg.max_demos,
                        frame_height=cfg.frame_height,
                        frame_width=cfg.frame_width,
                        terminal_if_done=cfg.terminal_if_done,
                        max_reject_sampling=cfg.max_reject_sampling,
                        include_no_robot=cfg.include_no_robot,
                        render=cfg.render,
                        save_mp4=cfg.save_mp4,
                        mp4_fps=cfg.mp4_fps,
                        copy_raw_data=cfg.copy_raw_data,
                        env=None,
                        gpu_id=i % num_gpus,
                    )
                    for i in range(len(raw_task_dirs))
                ],
            )
    else:
        env = KitchenBase(frame_height=cfg.frame_height, frame_width=cfg.frame_width)
        for raw_task_dir in tqdm.tqdm(
            raw_task_dirs, desc="Generate Franka Kitchen Demos"
        ):
            _generate_one_task_demo(
                raw_data_path=cfg.raw_data_path,
                raw_task_dir=raw_task_dir,
                seed_start=cfg.seed_start,
                output_path=cfg.output_path,
                max_demos=cfg.max_demos,
                frame_height=cfg.frame_height,
                frame_width=cfg.frame_width,
                terminal_if_done=cfg.terminal_if_done,
                max_reject_sampling=cfg.max_reject_sampling,
                include_no_robot=cfg.include_no_robot,
                render=cfg.render,
                save_mp4=cfg.save_mp4,
                mp4_fps=cfg.mp4_fps,
                copy_raw_data=cfg.copy_raw_data,
                env=env,
                gpu_id=-1,
            )


if __name__ == "__main__":
    main()
