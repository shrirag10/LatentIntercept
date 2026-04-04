#!/usr/bin/env python3
"""
scripts/train.py
================
Main LatentIntercept training script.

Integrates the Genesis AirHockey environment with the TD-MPC2 agent
from third_party/tdmpc2.

Usage:
    # Basic run (256 parallel envs, RTX 4060)
    python scripts/train.py

    # Override config values  (dot-notation key=value pairs)
    python scripts/train.py training.total_steps=5000000 tdmpc2.batch_size=128

    # Resume from checkpoint
    python scripts/train.py --checkpoint trained_models/step_1000000.pt

Architecture overview:
  ┌─────────────────────────────────────────┐
  │  Genesis AirHockeyEnv (256 parallel)    │
  │  → 24D obs, 8D action, dense reward     │
  └──────────────┬──────────────────────────┘
                 │ obs, reward, done
  ┌──────────────▼──────────────────────────┐
  │  GenesisTDMPC2Wrapper                   │
  │  → action scaling, auto-reset           │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │  TD-MPC2 Agent (third_party/tdmpc2)     │
  │  • World model (encoder + dynamics)     │
  │  • MPPI planner (horizon=12)            │
  │  • Policy network (for seeding MPPI)    │
  │  • Buffer (TensorDict episode storage)  │
  └─────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Make tdmpc2 importable from third_party ───────────────────────────────────
from src.utils.tdmpc2_path import setup_tdmpc2_path
setup_tdmpc2_path()

from src.environments.air_hockey_env import AirHockeyEnv
from src.environments.wrapper import GenesisTDMPC2Wrapper
from src.agent.tdmpc2_config import build_tdmpc2_config, print_config
from src.utils.logger import TrainingLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LatentIntercept with TD-MPC2")
    p.add_argument("--config",     type=str, default="configs/base_config.yaml")
    p.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--device",     type=str, default="cuda")
    # Allow arbitrary overrides: key=value pairs appended to CLI
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


def apply_cli_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides like training.total_steps=5000000."""
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = obj.setdefault(part, {})
        try:
            obj[parts[-1]] = int(val)
        except ValueError:
            try:
                obj[parts[-1]] = float(val)
            except ValueError:
                obj[parts[-1]] = val
    return cfg


def main() -> None:
    args = parse_args()

    # ── Load YAML config ─────────────────────────────────────────────────────
    cfg_path = ROOT / args.config
    base_cfg: dict = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    base_cfg = apply_cli_overrides(base_cfg, args.overrides)

    train_cfg = base_cfg.get("training", {})
    total_steps         = train_cfg.get("total_steps",         10_000_000)
    log_interval        = train_cfg.get("log_interval",        1_000)
    eval_interval       = train_cfg.get("eval_interval",       50_000)
    eval_episodes       = train_cfg.get("eval_episodes",       20)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 1_000_000)

    paths_cfg = base_cfg.get("paths", {})
    checkpoint_dir = ROOT / paths_cfg.get("checkpoints", "trained_models")
    runs_dir       = ROOT / paths_cfg.get("tensorboard",  "runs")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ── Build TD-MPC2 config ─────────────────────────────────────────────────
    agent_cfg = build_tdmpc2_config(base_cfg)
    # Sync steps and device with training config
    from omegaconf import OmegaConf
    OmegaConf.update(agent_cfg, "steps", total_steps, merge=True)
    OmegaConf.update(agent_cfg, "device", args.device, merge=True)
    print_config(agent_cfg)

    # ── Build environment ─────────────────────────────────────────────────────
    n_envs    = base_cfg.get("n_envs",       256)
    sim_dt    = base_cfg.get("sim_dt",       0.01)
    ctrl_freq = base_cfg.get("control_freq", 10)
    ep_len    = base_cfg.get("episode_len",  200)

    # TD-MPC2's MPPI planner uses a single _prev_mean buffer and must be
    # called sequentially per env.  Large n_envs dramatically slows training.
    # Consider n_envs=4-8 for MPPI, or set tdmpc2.mpc=False to use the
    # policy network directly (much faster, viable after seed phase).
    if n_envs > 32 and agent_cfg.get("mpc", True):
        print(f"[WARN] n_envs={n_envs} with MPPI planning — this will be slow. "
              f"Consider reducing n_envs or setting tdmpc2.mpc=False.")

    env = AirHockeyEnv(
        n_envs=n_envs,
        dt=sim_dt,
        control_freq=ctrl_freq,
        episode_len=ep_len,
        device=args.device,
        show_viewer=False,
        seed=args.seed,
        cfg=base_cfg,
    )
    wrapper = GenesisTDMPC2Wrapper(env, cfg=base_cfg)
    print(f"Environment ready: {wrapper}\n")

    # ── Instantiate TD-MPC2 agent + buffer ────────────────────────────────────
    from tdmpc2 import TDMPC2
    from common.buffer import Buffer

    agent  = TDMPC2(agent_cfg)
    buffer = Buffer(agent_cfg)
    print(f"Agent ready:  TDMPC2 (latent_dim={agent_cfg.latent_dim})")

    # ── Optionally resume from checkpoint ────────────────────────────────────
    start_step = 0
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            agent.load(ckpt_path)
            try:
                start_step = int(ckpt_path.stem.split("_")[-1])
            except ValueError:
                start_step = 0
                print(f"[WARN] Could not parse step from '{ckpt_path.name}', resuming from step 0")
            print(f"Resumed from checkpoint: {ckpt_path} (step {start_step})\n")

    # ── Logger ───────────────────────────────────────────────────────────────
    logger = TrainingLogger(log_dir=runs_dir / "latent_intercept")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"Starting training. Total steps: {total_steps:,}\n")
    t_start       = time.time()
    ep_reward_sum = torch.zeros(n_envs, device=args.device)

    # Episode-level accumulators used to build TensorDict episodes for Buffer
    from tensordict import TensorDict

    obs = wrapper.reset()   # (N, 24)

    # Per-env episode storage: lists of tensors
    ep_obs     = [[] for _ in range(n_envs)]
    ep_act     = [[] for _ in range(n_envs)]
    ep_rew     = [[] for _ in range(n_envs)]
    ep_done    = [[] for _ in range(n_envs)]

    # Record initial obs for each env
    for i in range(n_envs):
        ep_obs[i].append(obs[i].cpu())

    global_step = start_step
    iters_since_log = 0

    while global_step < total_steps:
        seed_phase = global_step < agent_cfg.seed_steps

        # ─ Act ───────────────────────────────────────────────────────────────
        if seed_phase:
            action = torch.rand(n_envs, wrapper.action_dim, device=args.device) * 2 - 1
        else:
            # t0=True because the agent's _prev_mean is a single buffer and
            # cannot track per-env planning state.  This disables MPPI
            # warm-starting but ensures correctness with parallel envs.
            actions_list = []
            for env_i in range(n_envs):
                act = agent.act(obs[env_i], t0=True, eval_mode=False)
                actions_list.append(act.to(args.device))
            action = torch.stack(actions_list, dim=0)  # (N, 8)

        # ─ Step ──────────────────────────────────────────────────────────────
        next_obs, reward, terminated, info = wrapper.step(action)

        # ─ Store transition for each env ──────────────────────────────────────
        for i in range(n_envs):
            ep_act[i].append(action[i].cpu())
            ep_rew[i].append(reward[i].cpu())
            ep_done[i].append(terminated[i].cpu())
            if terminated[i]:
                ep_obs[i].append(info["terminal_obs"][i].cpu())
            else:
                ep_obs[i].append(next_obs[i].cpu())

        # ─ Flush completed episodes into the Buffer ───────────────────────────
        done_ids = terminated.nonzero(as_tuple=True)[0].tolist()
        for i in done_ids:
            T = len(ep_act[i])
            if T < 2:
                ep_obs[i]  = [next_obs[i].cpu()]
                ep_act[i]  = []
                ep_rew[i]  = []
                ep_done[i] = []
                continue

            dummy_act  = torch.zeros_like(ep_act[i][0])
            dummy_rew  = torch.zeros_like(ep_rew[i][0])
            dummy_done = torch.zeros_like(ep_done[i][0])

            ep_td = TensorDict({
                "obs":        torch.stack(ep_obs[i]),
                "action":     torch.stack([dummy_act]  + ep_act[i]),
                "reward":     torch.stack([dummy_rew]  + ep_rew[i]),
                "terminated": torch.stack([dummy_done] + ep_done[i]).float(),
            }, batch_size=[T + 1])

            buffer.add(ep_td)

            # Reset accumulators for this env
            ep_obs[i]  = [next_obs[i].cpu()]
            ep_act[i]  = []
            ep_rew[i]  = []
            ep_done[i] = []

        obs = next_obs

        # ─ Update world model ─────────────────────────────────────────────────
        train_info = {}
        if not seed_phase and buffer.num_eps > 0:
            try:
                train_info_td = agent.update(buffer)
                train_info = {k: v.item() for k, v in train_info_td.items()
                              if hasattr(v, "item")}
            except RuntimeError:
                pass  # not enough long trajectories in buffer yet

        ep_reward_sum += reward
        global_step += n_envs
        iters_since_log += 1

        # ─ Logging (fires when global_step crosses a log_interval boundary) ──
        if global_step % log_interval < n_envs and global_step > start_step + n_envs:
            elapsed = time.time() - t_start
            fps     = (global_step - start_step) / elapsed if elapsed > 0 else 0
            mean_r  = ep_reward_sum.mean().item() / max(iters_since_log, 1)
            ep_reward_sum.zero_()
            iters_since_log = 0

            logger.log_scalar("train/mean_reward",  mean_r,               global_step)
            logger.log_scalar("train/success_rate", wrapper.success_rate, global_step)
            logger.log_scalar("train/fps",          fps,                  global_step)
            for k, v in train_info.items():
                logger.log_scalar(f"train/{k}", v, global_step)

            print(
                f"step {global_step:>10,} / {total_steps:,} | "
                f"reward {mean_r:+.4f} | "
                f"success {wrapper.success_rate:.2%} | "
                f"fps {fps:,.0f} | "
                f"buf_eps {buffer.num_eps}"
            )
            wrapper.reset_stats()

        # ─ Evaluation ────────────────────────────────────────────────────────
        if global_step % eval_interval < n_envs and global_step > n_envs:
            saved_successes = wrapper._successes
            saved_episodes  = wrapper._episodes
            saved_ep_reward = wrapper._ep_reward.clone()
            eval_success, eval_reward = evaluate(agent, wrapper, eval_episodes, args.device)
            wrapper._successes = saved_successes
            wrapper._episodes  = saved_episodes
            wrapper._ep_reward = saved_ep_reward
            logger.log_scalar("eval/success_rate", eval_success, global_step)
            logger.log_scalar("eval/mean_reward",  eval_reward,  global_step)
            print(
                f"\n[EVAL] step {global_step:,} | "
                f"success={eval_success:.2%} | "
                f"reward={eval_reward:.4f}\n"
            )

            obs = wrapper.reset()
            ep_obs  = [[obs[i].cpu()] for i in range(n_envs)]
            ep_act  = [[] for _ in range(n_envs)]
            ep_rew  = [[] for _ in range(n_envs)]
            ep_done = [[] for _ in range(n_envs)]
            ep_reward_sum.zero_()

        # ─ Checkpoint ────────────────────────────────────────────────────────
        if global_step % checkpoint_interval < n_envs and global_step > n_envs:
            ckpt_path = checkpoint_dir / f"step_{global_step:08d}.pt"
            agent.save(ckpt_path)
            print(f"[CKPT] Saved → {ckpt_path}")

    # ── Final save ───────────────────────────────────────────────────────────
    final_ckpt = checkpoint_dir / "final.pt"
    agent.save(final_ckpt)
    print(f"\nTraining complete. Final checkpoint → {final_ckpt}")
    wrapper.close()
    logger.close()


@torch.no_grad()
def evaluate(
    agent,
    wrapper: GenesisTDMPC2Wrapper,
    num_episodes: int,
    device: str,
) -> tuple[float, float]:
    """
    Run deterministic evaluation episodes.

    Returns (success_rate, mean_episode_reward).
    """
    obs = wrapper.reset()
    total_reward = 0.0
    done_envs    = 0
    success_envs = 0

    n_envs   = wrapper.n_envs
    complete = torch.zeros(n_envs, dtype=torch.bool, device=device)

    step          = 0
    max_eval_steps = wrapper._env.episode_len + 10

    while done_envs < num_episodes and step < max_eval_steps:
        actions_list = []
        for env_i in range(n_envs):
            if complete[env_i]:
                actions_list.append(torch.zeros(wrapper.action_dim, device=device))
            else:
                act = agent.act(obs[env_i], t0=True, eval_mode=True)
                actions_list.append(act.to(device))
        action = torch.stack(actions_list, dim=0)

        obs, reward, terminated, info = wrapper.step(action)

        newly_done = terminated & ~complete
        if newly_done.any():
            total_reward  += info["ep_reward"][newly_done].sum().item()
            success_envs  += info["caught"][newly_done].sum().item()
        done_envs     += newly_done.sum().item()
        complete      |= terminated

        if complete.all():
            obs = wrapper.reset()
            complete[:] = False

        step += 1

    success_rate = success_envs / max(done_envs, 1)
    mean_reward  = total_reward  / max(done_envs, 1)
    return success_rate, mean_reward


if __name__ == "__main__":
    main()
