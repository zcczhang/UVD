import argparse
import copy
import time

import gym
import numpy as np
import torch
import yaml
from omegaconf import DictConfig

import uvd.utils as U
from uvd.models.preprocessors import get_preprocessor
from uvd.decomp.decomp import embedding_decomp, DEFAULT_DECOMP_KWARGS
from uvd.envs.evaluator.inference_wrapper import InferenceWrapper
from uvd.envs.franka_kitchen.franka_kitchen_base import KitchenBase

MLP_CFG = """\
policy:
  _target_: uvd.models.policy.MLPPolicy
  observation_space: ???
  action_space: ???
  preprocessor: ???
  obs_encoder:
    __target__: uvd.models.nn.MLP
    hidden_dims: [1024, 512, 256]
    activation: ReLU
    normalization: false
    input_normalization: BatchNorm1d
    input_normalization_full_obs: false
    proprio_output_dim: 512
    proprio_add_layernorm: true
    proprio_activation: Tanh
    proprio_add_noise_eval: false
    actor_act: Tanh
  act_head:
    __target__: uvd.models.distributions.DeterministicHead
"""

GPT_CFG = """\
policy:
  _target_: uvd.models.policy.GPTPolicy
  observation_space: ???
  action_space: ???
  preprocessor: ???
  use_kv_cache: true
  max_seq_length: 10
  obs_add: false
  proprio_hidden_dim: 512
  obs_encoder:
    __target__: uvd.models.nn.GPT
    use_wte: true
  gpt_config:
    block_size: 10
    vocab_size: null
    n_embd: 768
    n_layer: 8
    n_head: 8
    dropout: 0.1
    bias: false
    use_llama_impl: true
    position_embed: rotary
  act_head:
    __target__: uvd.models.distributions.DeterministicHead
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="gpt")
    parser.add_argument("--preprocessor_name", default="vip")
    parser.add_argument("--use_uvd", action="store_true")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print("NO GPU FOUND")
    preprocessor = get_preprocessor(
        args.preprocessor_name, device="cuda" if use_gpu else None
    )
    policy_name = args.policy.lower()
    assert policy_name in ["mlp", "gpt"]
    is_causal = policy_name == "gpt"

    env = KitchenBase(frame_height=224, frame_width=224)
    env = InferenceWrapper(env, dummy_rtn=is_causal)
    env.reset()

    observation_space = gym.spaces.Dict(
        rgb=gym.spaces.Box(-np.inf, np.inf, preprocessor.output_dim, np.float32),
        proprio=gym.spaces.Box(-1, 1, (9,), np.float32),
        milestones=gym.spaces.Box(
            -np.inf, np.inf, (6,) + preprocessor.output_dim, np.float32
        ),
    )
    action_space = env.action_space

    cfg = yaml.safe_load(MLP_CFG if policy_name == "mlp" else GPT_CFG)
    cfg = DictConfig(cfg)
    policy = U.hydra_instantiate(
        cfg.policy,
        observation_space=observation_space,
        action_space=action_space,
        preprocessor=preprocessor,
    )
    policy = policy.to(preprocessor.device).eval()
    U.debug_model_info(policy)
    if is_causal:
        assert policy.causal and policy.use_kv_cache

    preprocessor = policy.preprocessor
    # Or load FrankaKitchen dummy datas
    dummy_data = np.random.random((300, 224, 224, 3)).astype(np.float32)
    emb = preprocessor.process(dummy_data, return_numpy=True)
    if args.use_uvd:
        _, decomp_meta = embedding_decomp(
            embeddings=emb,
            fill_embeddings=False,
            return_intermediate_curves=False,
            **DEFAULT_DECOMP_KWARGS["embed"],
        )
        milestones = emb[decomp_meta.milestone_indices]  # nhw3
    else:
        milestones = emb[-1][None, ...]
    env.milestones = milestones

    MAX_HORIZON = 300
    totals = []
    for _ in range(args.n):
        obs = env.reset()
        if is_causal:
            policy.reset_cache()

        times = []
        for st in range(MAX_HORIZON):
            t = time.time()
            obs = copy.deepcopy(obs)
            batchify_obs = U.batch_observations([obs], device=policy.device)
            if is_causal:
                # B, T, ...
                cur_milestone = env.current_milestone[None, None, ...]
                for k in batchify_obs:
                    batchify_obs[k] = batchify_obs[k][:, None, ...]
            else:
                # B, ...
                cur_milestone = env.current_milestone[None, ...]
            with torch.no_grad():
                action, obs_embed, goal_embed = policy(
                    batchify_obs,
                    goal=torch.as_tensor(cur_milestone, device=policy.device),
                    deterministic=True,
                    return_embeddings=True,
                    input_pos=torch.tensor([st], device=policy.device)
                    if is_causal
                    else None,
                )
            env.current_obs_embedding = obs_embed[0].cpu().numpy()
            obs, r, done, info = env.step(action[0].cpu().numpy())
            step_t = time.time() - t
            times.append(step_t)
        times = np.sum(times)
        print(times)
        totals.append(times)
    print(np.mean(totals))
