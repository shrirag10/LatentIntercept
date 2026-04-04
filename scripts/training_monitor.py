#!/usr/bin/env python3
"""Live training progress monitor — reads TensorBoard event files."""

import time
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("pip install tensorboard")
    sys.exit(1)

from tqdm import tqdm

LOG_DIR = Path(__file__).resolve().parents[1] / "runs" / "latent_intercept"
TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
POLL_INTERVAL = 10  # seconds


def read_latest(ea):
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    latest = {}
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            latest[tag] = (events[-1].step, events[-1].value)
    return latest


def main():
    print(f"Monitoring: {LOG_DIR}")
    print(f"Target: {TOTAL_STEPS:,} steps  |  Polling every {POLL_INTERVAL}s\n")

    ea = EventAccumulator(str(LOG_DIR))

    pbar = tqdm(
        total=TOTAL_STEPS,
        unit="step",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} steps "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
        dynamic_ncols=True,
    )

    prev_step = 0

    while True:
        try:
            data = read_latest(ea)
        except Exception:
            pbar.set_postfix_str("waiting for event file…")
            time.sleep(POLL_INTERVAL)
            continue

        step = 0
        reward = fps = success = 0.0

        if "train/fps" in data:
            step, fps = data["train/fps"]
        if "train/mean_reward" in data:
            _, reward = data["train/mean_reward"]
        if "train/success_rate" in data:
            _, success = data["train/success_rate"]

        if step == 0:
            pbar.set_postfix_str("waiting for first log entry…")
            time.sleep(POLL_INTERVAL)
            continue

        delta = step - prev_step
        if delta > 0:
            pbar.update(delta)
            prev_step = step

        pbar.set_postfix_str(
            f"rew={reward:+.3f}  success={success:.1%}  fps={fps:.0f}"
        )

        if step >= TOTAL_STEPS:
            pbar.close()
            print("\nTraining complete!")
            break

        time.sleep(POLL_INTERVAL)

    pbar.close()


if __name__ == "__main__":
    main()
