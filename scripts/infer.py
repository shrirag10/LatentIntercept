#!/usr/bin/env python3
"""
scripts/infer.py
================
Load a trained TD-MPC2 checkpoint and run inference demo.

Usage:
    python scripts/infer.py --checkpoint trained_models/final.pt [--render]

The script runs the agent in deterministic (eval_mode=True) mode for
a configurable number of episodes and prints per-episode statistics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.tdmpc2_path import setup_tdmpc2_path
setup_tdmpc2_path()

from src.environments.air_hockey_env import AirHockeyEnv
from src.environments.wrapper import GenesisTDMPC2Wrapper
from src.agent.tdmpc2_config import build_tdmpc2_config
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LatentIntercept inference demo")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config",     type=str, default="configs/base_config.yaml")
    p.add_argument("--episodes",   type=int, default=10)
    p.add_argument("--render",     action="store_true", help="Open Genesis viewer")
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = ROOT / args.config
    base_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    sim_dt    = base_cfg.get("sim_dt", 0.01)
    ctrl_freq = base_cfg.get("control_freq", 10)
    ep_len    = base_cfg.get("episode_len", 200)

    env = AirHockeyEnv(
        n_envs=1,
        dt=sim_dt,
        control_freq=ctrl_freq,
        episode_len=ep_len,
        device=args.device,
        show_viewer=args.render,
        cfg=base_cfg,
    )
    wrapper = GenesisTDMPC2Wrapper(env, cfg=base_cfg)

    agent_cfg = build_tdmpc2_config(base_cfg)

    try:
        from tdmpc2 import TDMPC2
    except ImportError:
        print("[ERROR] TD-MPC2 not installed.")
        sys.exit(1)

    agent = TDMPC2(agent_cfg)
    agent.load(Path(args.checkpoint))
    agent.eval()
    print(f"Loaded checkpoint from: {args.checkpoint}\n")

    successes = 0
    for ep in range(args.episodes):
        obs = wrapper.reset()
        ep_reward = 0.0
        caught = False

        for step in range(env.episode_len):
            with torch.no_grad():
                action = agent.act(obs[0], t0=(step == 0), eval_mode=True).unsqueeze(0)
            obs, reward, terminated, info = wrapper.step(action)
            ep_reward += reward[0].item()
            if terminated[0]:
                caught = info["caught"][0].item()
                break

        successes += int(caught)
        status = "✓ CAUGHT" if caught else "✗ MISSED"
        print(f"  Episode {ep+1:2d}/{args.episodes}  {status}  reward={ep_reward:.3f}")

    print(f"\n  Success rate: {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)")
    wrapper.close()


if __name__ == "__main__":
    main()
