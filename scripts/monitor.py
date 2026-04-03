#!/usr/bin/env python3
"""
scripts/monitor.py
==================
Training health-check script.  Reads TensorBoard event files from the
latest run directory and prints key metric summaries.

Usage:
    python scripts/monitor.py                 # auto-detect latest run
    python scripts/monitor.py --logdir runs/  # explicit directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("[ERROR] tensorboard not installed. Run: pip install tensorboard")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training health monitor")
    p.add_argument("--logdir", type=str, default="runs/",
                   help="TensorBoard log directory")
    p.add_argument("--last", type=int, default=20,
                   help="Show last N data points per metric")
    return p.parse_args()


def find_latest_run(logdir: str) -> Path:
    """Return the most recently modified subdirectory in logdir."""
    base = Path(logdir)
    if not base.exists():
        print(f"[ERROR] Log directory '{logdir}' does not exist.")
        sys.exit(1)
    subdirs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    subdirs = [d for d in subdirs if d.is_dir()]
    if not subdirs:
        # Maybe the events are directly in logdir
        return base
    return subdirs[0]


def print_metric_summary(ea: EventAccumulator, tag: str, last_n: int) -> None:
    """Print summary statistics for a single scalar tag."""
    events = ea.Scalars(tag)
    if not events:
        print(f"  {tag}: (no data)")
        return

    values = [e.value for e in events]
    recent = values[-last_n:]

    print(f"  {tag}:")
    print(f"    Total points : {len(values)}")
    print(f"    Latest value : {values[-1]:.4f}")
    print(f"    Last {last_n} mean : {sum(recent) / len(recent):.4f}")
    print(f"    Last {last_n} min  : {min(recent):.4f}")
    print(f"    Last {last_n} max  : {max(recent):.4f}")
    print(f"    All-time best: {max(values):.4f}")


def main() -> None:
    args = parse_args()
    run_dir = find_latest_run(args.logdir)
    print(f"Reading logs from: {run_dir}\n")

    ea = EventAccumulator(str(run_dir))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print("[WARN] No scalar data found. Training may not have started yet.")
        return

    print(f"Found {len(tags)} metric(s):\n")

    # Priority order — show these first if they exist
    priority = ["train/mean_reward", "train/success_rate",
                "train/fps", "train/value_loss", "train/policy_loss"]
    ordered_tags = [t for t in priority if t in tags]
    ordered_tags += [t for t in sorted(tags) if t not in ordered_tags]

    for tag in ordered_tags:
        print_metric_summary(ea, tag, args.last)
        print()

    # Quick health verdict
    if "train/mean_reward" in tags:
        rewards = [e.value for e in ea.Scalars("train/mean_reward")]
        if len(rewards) >= 100:
            early = sum(rewards[:20]) / 20
            late = sum(rewards[-20:]) / 20
            if late > early * 1.2:
                print("✅ HEALTHY: Reward is trending upward")
            elif late < early * 0.8:
                print("⚠️  WARNING: Reward is declining — check hyperparameters")
            else:
                print("🔄 PLATEAU: Reward is flat — may need more steps or tuning")
        else:
            print("🕐 TOO EARLY: Not enough data for health assessment")


if __name__ == "__main__":
    main()
