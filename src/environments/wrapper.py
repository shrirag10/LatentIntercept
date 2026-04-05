"""
LatentIntercept — GenesisTDMPC2Wrapper
======================================
Bridges AirHockeyEnv to the TD-MPC2 training API.

TD-MPC2 expects a minimal Gym-compatible interface:
  • env.obs_space  / env.act_space  (gymnasium.spaces)
  • env.reset()    → obs
  • env.step(act)  → (obs, reward, done, info)

This wrapper normalises actions from [-1, 1] (TD-MPC2 convention) to
position deltas for the arm joints and exposes `obs_dim` / `action_dim`
attributes that TD-MPC2 reads from the config dictionary.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from src.environments.air_hockey_env import AirHockeyEnv




class GenesisTDMPC2Wrapper:
    """
    Vectorised wrapper adapting AirHockeyEnv for the TD-MPC2 agent.

    Responsibilities
    ----------------
    1. Expose `obs_dim`, `action_dim`, `obs_space`, `act_space` attributes.
    2. Normalise actions from [-1, 1] → position deltas (rad) for joints
       and [-1, 1] pass-through for gripper.
    3. Auto-reset individual environments upon episode termination.
    4. Keep a running success counter for logging.

    Parameters
    ----------
    env : AirHockeyEnv
        The underlying Genesis environment instance.
    cfg : dict, optional
        Config dict; used to pull reward weights from YAML.
    """

    def __init__(self, env: AirHockeyEnv, cfg: Optional[dict] = None) -> None:
        self._env = env
        self._cfg = cfg or {}

        # ── Reward weights (from config or defaults) ─────────────────────────
        reward_cfg = self._cfg.get("reward", {})
        self._w_dist        = float(reward_cfg.get("w_dist", 1.0))
        self._w_vel         = float(reward_cfg.get("w_vel", 0.1))
        self._success_bonus = float(reward_cfg.get("success_bonus", 10.0))
        self._dist_scale    = float(reward_cfg.get("dist_scale", 5.0))

        # ── Gym-compatible observation / action spaces ────────────────────────
        obs_lo = np.full(AirHockeyEnv.OBS_DIM, -np.inf, dtype=np.float32)
        obs_hi = np.full(AirHockeyEnv.OBS_DIM,  np.inf, dtype=np.float32)
        self.obs_space = gym.spaces.Box(obs_lo, obs_hi, dtype=np.float32)

        act_lo = np.full(AirHockeyEnv.ACTION_DIM, -1.0, dtype=np.float32)
        act_hi = np.full(AirHockeyEnv.ACTION_DIM, +1.0, dtype=np.float32)
        self.act_space = gym.spaces.Box(act_lo, act_hi, dtype=np.float32)

        # Shorthand expected by TD-MPC2's environment resolver
        self.obs_dim    = AirHockeyEnv.OBS_DIM
        self.action_dim = AirHockeyEnv.ACTION_DIM
        self.n_envs     = env.n_envs
        self.device     = env.device

        # ── Position-delta scale on device ────────────────────────────────────
        self._delta_scale = env.POSITION_DELTA_MAX

        # ── Running stats ─────────────────────────────────────────────────────
        self._ep_reward = torch.zeros(env.n_envs, device=self.device)
        self._successes = 0
        self._episodes  = 0

    # ────────────────────────────────────────────────────────────────────────
    # Core Gym interface
    # ────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reset all (or selected) environments and return initial obs."""
        obs = self._env.reset(env_ids=env_ids)
        if env_ids is None:
            self._ep_reward[:] = 0.0
        else:
            self._ep_reward[env_ids] = 0.0
        return obs  # (N, OBS_DIM)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Scale action from [-1, +1] and forward to AirHockeyEnv.

        TD-MPC2 outputs actions in [-1, 1]. We map them as:
          • action[:, :7]  → position deltas in [-delta_max, +delta_max] rad
          • action[:, 7]   → gripper in [-1, +1] (passed through unchanged)

        Returns
        -------
        obs        : (N, OBS_DIM)
        reward     : (N,)
        terminated : (N,)   bool
        info       : dict
        """
        action = self._scale_action(action)
        obs, reward, terminated, info = self._env.step(action)

        # Recompute reward only if config weights differ from env defaults
        if (self._w_dist != 1.0 or self._w_vel != 0.1
                or self._success_bonus != 10.0 or self._dist_scale != 5.0):
            reward = self._recompute_reward_weighted(info["caught"])

        self._ep_reward += reward

        # Log episode stats for completed environments
        done_ids = terminated.nonzero(as_tuple=True)[0]
        if done_ids.numel() > 0:
            self._episodes  += done_ids.numel()
            self._successes += info["caught"][done_ids].sum().item()

        info["ep_reward"] = self._ep_reward.clone()
        info["terminal_obs"] = obs.clone()

        # Auto-reset terminated environments and zero their reward accumulators
        if done_ids.numel() > 0:
            reset_obs = self._env.reset(env_ids=done_ids)
            obs[done_ids] = reset_obs
            self._ep_reward[done_ids] = 0.0

        return obs, reward, terminated, info

    # ────────────────────────────────────────────────────────────────────────
    # Action scaling
    # ────────────────────────────────────────────────────────────────────────

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] normalised action → physical action space."""
        arm_raw     = action[:, :7]   # (N, 7) in [-1, 1]
        gripper_raw = action[:, 7:8]  # (N, 1) in [-1, 1]

        arm_deltas = arm_raw * self._delta_scale
        # gripper_cmd passed through unchanged — env interprets in [-1, 1]
        return torch.cat([arm_deltas, gripper_raw], dim=1)

    # ────────────────────────────────────────────────────────────────────────
    # Weighted reward computation
    # ────────────────────────────────────────────────────────────────────────

    def _recompute_reward_weighted(
        self, caught: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recompute the dense reward with cfg-specified weights.

        This is called *after* env.step() so the scene is at the new state.
        Accepts pre-computed caught tensor to avoid redundant contact queries.
        """
        return self._env._compute_reward(
            w_dist=self._w_dist,
            w_vel=self._w_vel,
            success_bonus=self._success_bonus,
            dist_scale=self._dist_scale,
            _caught=caught,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Statistics
    # ────────────────────────────────────────────────────────────────────────

    @property
    def success_rate(self) -> float:
        """Episode success rate since last reset_stats() call."""
        if self._episodes == 0:
            return 0.0
        return self._successes / self._episodes

    def reset_stats(self) -> None:
        self._successes = 0
        self._episodes  = 0

    # ────────────────────────────────────────────────────────────────────────
    # Pass-through helpers
    # ────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._env.close()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GenesisTDMPC2Wrapper("
            f"n_envs={self.n_envs}, "
            f"obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, "
            f"success_rate={self.success_rate:.2%})"
        )
