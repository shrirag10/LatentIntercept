#!/usr/bin/env python3
"""
scripts/validate_env.py
========================
Quick visual smoke-test of the Genesis environment.

Opens the Genesis interactive viewer, spawns the block with a random
impulse, and runs for `num_steps` control steps.

Usage:
    # Passive mode — arm stays still, block slides
    python scripts/validate_env.py [--n_envs 1] [--steps 200] [--device cuda]

    # Zero-shot mode — untrained TD-MPC2 agent controls the arm
    python scripts/validate_env.py --zero_shot [--steps 200] [--device cuda]

    # Zero-shot with a trained checkpoint
    python scripts/validate_env.py --zero_shot --checkpoint trained_models/final.pt

The viewer opens a real-time 3D window. Press Escape to close.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# ── Ensure project root is on the path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environments.air_hockey_env import AirHockeyEnv
from src.environments.wrapper import GenesisTDMPC2Wrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Genesis AirHockey environment")
    p.add_argument("--n_envs",     type=int,  default=1,     help="Number of parallel environments")
    p.add_argument("--steps",      type=int,  default=200,   help="Number of control steps to run")
    p.add_argument("--no_viewer",  action="store_true",      help="Run headless (no Genesis viewer)")
    p.add_argument("--config",     type=str,  default="configs/base_config.yaml")
    p.add_argument("--device",     type=str,  default="cuda")
    p.add_argument("--zero_shot",  action="store_true",      help="Run TD-MPC2 agent (random if no checkpoint)")
    p.add_argument("--checkpoint", type=str,  default=None,  help="Path to trained checkpoint (used with --zero_shot)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = ROOT / args.config
    cfg: dict = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    mode = "zero-shot agent" if args.zero_shot else "passive (zero action)"

    print(f"\n{'='*55}")
    print("  LatentIntercept — Environment Validation")
    print(f"{'='*55}")
    print(f"  mode       : {mode}")
    print(f"  n_envs     : {args.n_envs}")
    print(f"  steps      : {args.steps}")
    print(f"  viewer     : {not args.no_viewer}")
    print(f"  device     : {args.device}")
    if args.checkpoint:
        print(f"  checkpoint : {args.checkpoint}")
    print(f"{'='*55}\n")

    sim_dt    = cfg.get("sim_dt", 0.01)
    ctrl_freq = cfg.get("control_freq", 10)
    ep_len    = cfg.get("episode_len", 200)

    env = AirHockeyEnv(
        n_envs=args.n_envs,
        dt=sim_dt,
        control_freq=ctrl_freq,
        episode_len=ep_len,
        show_viewer=not args.no_viewer,
        device=args.device,
        cfg=cfg,
    )
    wrapper = GenesisTDMPC2Wrapper(env, cfg=cfg)

    print(f"Environment built: {env}")
    print(f"  obs_dim    = {env.OBS_DIM}")
    print(f"  action_dim = {env.ACTION_DIM}\n")

    # ── Load TD-MPC2 agent if zero-shot mode ─────────────────────────────────
    agent = None
    if args.zero_shot:
        from src.utils.tdmpc2_path import setup_tdmpc2_path
        setup_tdmpc2_path()
        from src.agent.tdmpc2_config import build_tdmpc2_config
        from omegaconf import OmegaConf

        agent_cfg = build_tdmpc2_config(cfg)
        OmegaConf.update(agent_cfg, "device", args.device, merge=True)

        from tdmpc2 import TDMPC2
        agent = TDMPC2(agent_cfg)

        if args.checkpoint is not None:
            ckpt = Path(args.checkpoint)
            if ckpt.exists():
                agent.load(ckpt)
                print(f"  Loaded checkpoint: {ckpt}")
            else:
                print(f"  [WARN] Checkpoint not found: {ckpt} — using random weights")
        else:
            print("  Using randomly initialized agent (no checkpoint)")
        print()

    # ── Reset ─────────────────────────────────────────────────────────────────
    obs = wrapper.reset()
    print(f"Initial obs shape : {obs.shape}")
    print(f"Initial obs[:3]   : {obs[0, :3].cpu().numpy().round(3)}\n")

    successes = 0
    episodes  = 0
    pbar = tqdm(range(args.steps), desc="Validating", unit="step", dynamic_ncols=True)

    for step in pbar:
        # ── Choose action ─────────────────────────────────────────────────
        if agent is not None:
            with torch.no_grad():
                actions = []
                for i in range(args.n_envs):
                    act = agent.act(obs[i], t0=True, eval_mode=True)
                    actions.append(act.to(args.device))
                action = torch.stack(actions, dim=0)
        else:
            action = torch.zeros(args.n_envs, env.ACTION_DIM, device=args.device)
            action[:, 7] = 1.0  # open gripper

        obs, reward, terminated, info = wrapper.step(action)

        block_pos = obs[0, 15:18].cpu().numpy().round(3)
        block_vel = obs[0, 18:21].cpu().numpy().round(3)
        pbar.set_postfix(
            bpos=f"{block_pos}",
            bvel=f"{block_vel}",
            rew=f"{reward[0].item():.4f}",
        )

        done_ids = terminated.nonzero(as_tuple=True)[0]
        if done_ids.numel() > 0:
            episodes  += done_ids.numel()
            successes += info["caught"][done_ids].sum().item()

    wrapper.close()

    print(f"\n{'='*55}")
    print("  Validation complete")
    print(f"  Mode     : {mode}")
    print(f"  Episodes : {episodes}")
    print(f"  Catches  : {int(successes)}/{max(episodes, 1)}")
    if episodes > 0:
        print(f"  Success  : {successes / episodes:.1%}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
