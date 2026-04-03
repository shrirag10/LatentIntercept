#!/usr/bin/env python3
"""
scripts/visualize_trajectories.py
==================================
Run a trained TD-MPC2 agent for one episode and produce a publication-quality
3-D trajectory plot:

  • Red dashed   — predicted block trajectory (constant-velocity extrapolation)
  • Blue solid   — executed end-effector (EE) trajectory
  • Green star   — catch / intercept point (if successful)

Usage:
    python scripts/visualize_trajectories.py \
        --checkpoint trained_models/final.pt \
        --output outputs/trajectory_3d.png

The script can also run *without* a checkpoint: it records the block sliding
across while the arm stays at home position (useful for initial testing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ── Matplotlib config (Agg backend for headless servers) ────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.tdmpc2_path import setup_tdmpc2_path
setup_tdmpc2_path()

from src.environments.air_hockey_env import AirHockeyEnv
from src.environments.wrapper import GenesisTDMPC2Wrapper
from src.agent.tdmpc2_config import build_tdmpc2_config
import yaml


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-D trajectory visualisation")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained TD-MPC2 checkpoint. If omitted, "
                        "the arm stays at home pose (block-only demo).")
    p.add_argument("--zero_shot", action="store_true",
                   help="Run an untrained TD-MPC2 agent (random actions)")
    p.add_argument("--config",  type=str, default="configs/base_config.yaml")
    p.add_argument("--output",  type=str, default="outputs/trajectory_3d.png")
    p.add_argument("--pdf",     action="store_true", help="Also save as PDF")
    p.add_argument("--device",  type=str, default="cuda")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Data collection
# ────────────────────────────────────────────────────────────────────────────

def collect_episode(
    wrapper: GenesisTDMPC2Wrapper,
    env: AirHockeyEnv,
    agent=None,
    device: str = "cuda",
) -> dict:
    """
    Run one episode and record EE + block positions at every control step.

    Returns dict with numpy arrays:
        ee_traj   : (T, 3)
        block_traj: (T, 3)
        caught    : bool
        catch_idx : int | None   (timestep of catch, or None)
    """
    obs = wrapper.reset()

    ee_positions: list[np.ndarray] = []
    block_positions: list[np.ndarray] = []
    caught = False
    catch_idx = None

    for step in range(env.episode_len):
        # ── Read positions from the obs vector ──────────────────────────
        # obs layout: [0:7 q, 7:14 qd, 14 grip, 15:18 p_block, 18:21 v_block, 21:24 rel]
        p_block = obs[0, 15:18].cpu().numpy()
        rel     = obs[0, 21:24].cpu().numpy()
        p_ee    = p_block + rel  # p_ee = p_block + (p_ee - p_block)

        ee_positions.append(p_ee.copy())
        block_positions.append(p_block.copy())

        # ── Choose action ───────────────────────────────────────────────
        if agent is not None:
            with torch.no_grad():
                action = agent.act(obs[0], t0=(step == 0), eval_mode=True).unsqueeze(0)
        else:
            # No agent — zero torque, arm stays at home
            action = torch.zeros(1, env.ACTION_DIM, device=device)

        obs, reward, terminated, info = wrapper.step(action)

        if terminated[0]:
            # Use terminal_obs (pre-reset) — obs already contains the new episode's initial state
            terminal_obs = info["terminal_obs"]
            p_block_f = terminal_obs[0, 15:18].cpu().numpy()
            rel_f     = terminal_obs[0, 21:24].cpu().numpy()
            p_ee_f    = p_block_f + rel_f
            ee_positions.append(p_ee_f.copy())
            block_positions.append(p_block_f.copy())

            caught = bool(info["caught"][0].item())
            if caught:
                catch_idx = len(ee_positions) - 1
            break

    return {
        "ee_traj":    np.array(ee_positions),
        "block_traj": np.array(block_positions),
        "caught":     caught,
        "catch_idx":  catch_idx,
    }


# ────────────────────────────────────────────────────────────────────────────
# Predicted block trajectory (constant-velocity extrapolation)
# ────────────────────────────────────────────────────────────────────────────

def predict_block_path(
    p0: np.ndarray,
    v0: np.ndarray,
    dt: float = 0.1,
    n_steps: int = 200,
) -> np.ndarray:
    """Straight-line constant-velocity prediction from initial state."""
    t = np.arange(n_steps)[:, None] * dt           # (T, 1)
    return p0[None, :] + v0[None, :] * t            # (T, 3)


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def plot_trajectory(
    data: dict,
    predicted_block: np.ndarray,
    table_size: tuple[float, float, float],
    save_path: str,
    save_pdf: bool = False,
) -> None:
    """Create publication-quality top-down trajectory figure."""

    ee    = data["ee_traj"]
    block = data["block_traj"]

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

    bg_color = "#0d1117"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor("#131926")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    # ── Table boundary ──────────────────────────────────────────────────
    hl, hw = table_size[0] / 2, table_size[1] / 2
    table_rect = plt.Rectangle(
        (-hl, -hw), table_size[0], table_size[1],
        linewidth=1.5, edgecolor="cyan", facecolor="#1a2233", alpha=0.5,
    )
    ax.add_patch(table_rect)

    # ── Predicted block path (red dashed) ───────────────────────────────
    ax.plot(
        predicted_block[:, 0], predicted_block[:, 1],
        color="#ff4444", linestyle="--", linewidth=1.5, alpha=0.6,
        label="Predicted block path",
        path_effects=[pe.Stroke(linewidth=3, foreground="#ff444440"), pe.Normal()],
    )

    # ── Actual block path (orange solid) ────────────────────────────────
    ax.plot(
        block[:, 0], block[:, 1],
        color="#ff8c00", linewidth=2, alpha=0.9,
        label="Actual block path",
    )

    # ── EE trajectory (blue solid) ──────────────────────────────────────
    ax.plot(
        ee[:, 0], ee[:, 1],
        color="#4488ff", linewidth=2.5,
        label="End-effector path",
        path_effects=[pe.Stroke(linewidth=4, foreground="#4488ff60"), pe.Normal()],
    )

    # ── Start markers ───────────────────────────────────────────────────
    ax.scatter(ee[0, 0], ee[0, 1], color="#4488ff", s=80, marker="o",
               zorder=5, edgecolors="white", linewidths=0.5, label="EE start")
    ax.scatter(block[0, 0], block[0, 1], color="#ff8c00", s=80, marker="o",
               zorder=5, edgecolors="white", linewidths=0.5, label="Block start")

    # ── End markers (triangles) ─────────────────────────────────────────
    ax.scatter(ee[-1, 0], ee[-1, 1], color="#4488ff", s=80, marker="v",
               zorder=5, edgecolors="white", linewidths=0.5)
    ax.scatter(block[-1, 0], block[-1, 1], color="#ff8c00", s=80, marker="v",
               zorder=5, edgecolors="white", linewidths=0.5)

    # ── Catch point (green star) ────────────────────────────────────────
    if data["caught"] and data["catch_idx"] is not None:
        ci = data["catch_idx"]
        ax.scatter(
            ee[ci, 0], ee[ci, 1],
            color="#00ff88", s=300, marker="*", zorder=10,
            edgecolors="white", linewidths=0.8,
            label="Intercept point",
        )
        ax.annotate(
            "CATCH", (ee[ci, 0], ee[ci, 1]),
            textcoords="offset points", xytext=(0, 12),
            color="#00ff88", fontsize=9, fontweight="bold", ha="center",
        )

    # ── Franka base marker ──────────────────────────────────────────────
    ax.scatter(-0.65, 0.0, color="white", s=100, marker="s", zorder=6,
               edgecolors="cyan", linewidths=1.0, label="Franka base")

    # ── Labels & legend ─────────────────────────────────────────────────
    ax.set_xlabel("X (m)", fontsize=11, color="white", labelpad=8)
    ax.set_ylabel("Y (m)", fontsize=11, color="white", labelpad=8)
    ax.set_aspect("equal")
    margin = 0.15
    ax.set_xlim(-hl - margin, hl + margin)
    ax.set_ylim(-hw - margin, hw + margin)
    ax.grid(True, alpha=0.15, color="white")

    ax.set_title(
        "LatentIntercept — Top-Down Trajectory View",
        color="white", fontsize=14, fontweight="bold", pad=15,
    )

    legend = ax.legend(
        loc="upper right", fontsize=9, framealpha=0.4,
        facecolor="#1a1a2e", edgecolor="white", labelcolor="white",
    )

    # ── Save ────────────────────────────────────────────────────────────
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", facecolor=bg_color)
    print(f"  Saved PNG: {out}")

    if save_pdf:
        pdf_path = out.with_suffix(".pdf")
        fig.savefig(str(pdf_path), bbox_inches="tight", facecolor=bg_color)
        print(f"  Saved PDF: {pdf_path}")

    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg_path = ROOT / args.config
    base_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    env = AirHockeyEnv(
        n_envs=1,
        device=args.device,
        show_viewer=False,
        seed=args.seed,
        cfg=base_cfg,
    )
    wrapper = GenesisTDMPC2Wrapper(env, cfg=base_cfg)

    # Optionally load agent
    agent = None
    if args.checkpoint is not None or args.zero_shot:
        agent_cfg = build_tdmpc2_config(base_cfg)
        from omegaconf import OmegaConf
        OmegaConf.update(agent_cfg, "device", args.device, merge=True)
        try:
            from tdmpc2 import TDMPC2
        except ImportError:
            print("[ERROR] TD-MPC2 not installed — running without agent.")
        else:
            agent = TDMPC2(agent_cfg)
            if args.checkpoint is not None:
                agent.load(Path(args.checkpoint))
                print(f"  Loaded checkpoint: {args.checkpoint}")
            else:
                print("  Zero-shot mode: using untrained TD-MPC2 agent")
            agent.eval()

    print("  Collecting episode...")
    data = collect_episode(wrapper, env, agent=agent, device=args.device)
    print(f"  Episode length: {len(data['ee_traj'])} steps  |  "
          f"Caught: {'YES ✓' if data['caught'] else 'NO ✗'}")

    # Predicted block path from initial state
    control_dt = env.dt * env.control_freq
    p0 = data["block_traj"][0]
    v0 = (data["block_traj"][1] - data["block_traj"][0]) / control_dt
    predicted = predict_block_path(p0, v0, dt=control_dt, n_steps=len(data["block_traj"]))

    print("  Plotting...")
    plot_trajectory(
        data, predicted,
        table_size=env.TABLE_SIZE,
        save_path=args.output,
        save_pdf=args.pdf,
    )
    print("  Done.")

    wrapper.close()


if __name__ == "__main__":
    main()