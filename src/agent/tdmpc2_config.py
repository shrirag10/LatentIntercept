"""
LatentIntercept — TD-MPC2 Agent Configuration
==============================================
Builds an OmegaConf DictConfig that TD-MPC2 reads at agent construction.
All fields required by TDMPC2.__init__, WorldModel, and Buffer are included.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def build_tdmpc2_config(base_cfg: dict | None = None) -> DictConfig:
    """
    Construct and return the full TD-MPC2 OmegaConf config for LatentIntercept.

    Parameters
    ----------
    base_cfg : dict, optional
        The parsed base_config.yaml dict. Values from the ``tdmpc2``
        sub-key override the defaults below.

    Returns
    -------
    cfg : DictConfig
        A flat OmegaConf config passed directly to ``TDMPC2.__init__()``.
    """

    # Pull episode_len from top-level config if present
    episode_len = 200
    if base_cfg is not None:
        episode_len = base_cfg.get("episode_len", base_cfg.get("episode_length", 200))

    defaults: dict = {
        # ── Task identity ────────────────────────────────────────────────────
        "task":      "latent_intercept",
        "obs":       "state",           # state-based (no pixels)
        "multitask": False,
        "episodic":  False,             # set True to enable termination model

        # ── Episode / buffer sizing ───────────────────────────────────────────
        "episode_length":  episode_len, # required by TDMPC2._get_discount()
        "steps":           10_000_000,  # required by Buffer._capacity = min(buffer_size, steps)
        "seed_steps":      50_000,      # random exploration before planning

        # ── Network dimensions ───────────────────────────────────────────────
        "obs_dim":       24,            # must match AirHockeyEnv.OBS_DIM
        "action_dim":    8,             # must match AirHockeyEnv.ACTION_DIM
        "obs_shape":     {"state": [24]},  # encoder iterates keys; "state" → mlp path
        "latent_dim":    256,
        "mlp_dim":       256,
        "enc_dim":       128,
        "num_enc_layers": 2,
        "num_channels":  32,            # pixel obs (unused but field required)
        "task_dim":      0,             # must be 0 when multitask=False; encoder adds this to input dim
        "num_q":         5,             # Q-ensemble size
        "simnorm_dim":   8,             # SimNorm group size in world model
        "dropout":       0.01,

        # ── MPPI planner ─────────────────────────────────────────────────────
        "mpc":          True,           # use MPPI planning in agent.act()
        "horizon":      8,             # steps × 10Hz ≈ 0.8 s look-ahead
        "num_samples":  256,
        "num_elites":   64,
        "num_pi_trajs": 24,
        "iterations":   4,
        "min_std":      0.05,
        "max_std":      1.5,
        "temperature":  0.5,

        # ── Training ─────────────────────────────────────────────────────────
        "batch_size":      256,
        "buffer_size":     1_000_000,
        "lr":              3.0e-4,
        "enc_lr_scale":    0.3,         # encoder lr = lr * enc_lr_scale
        "grad_clip_norm":  20.0,
        "tau":             0.01,        # target Q EMA decay (soft update)
        "rho":             0.75,        # consistency / value loss decay per step

        # ── Loss weights ─────────────────────────────────────────────────────
        "consistency_coef":  20.0,
        "reward_coef":       0.5,
        "value_coef":        0.2,
        "termination_coef":  1.0,

        # ── Discount ─────────────────────────────────────────────────────────
        "discount_denom": 5,
        "discount_min":   0.95,
        "discount_max":   0.995,

        # ── Policy ───────────────────────────────────────────────────────────
        "log_std_min":    -10,
        "log_std_max":    2,
        "entropy_coef":   1e-4,

        # ── Critic / two-hot encoding ─────────────────────────────────────────
        "num_bins": 201,
        "vmin":     -100,
        "vmax":     +100,
        "bin_size": 200 / (201 - 1),  # (vmax - vmin) / (num_bins - 1)

        # ── Hardware ─────────────────────────────────────────────────────────
        "device":  "cuda",
        "fp16":    False,
        "compile": False,

        # ── Logging ──────────────────────────────────────────────────────────
        "log_freq":     1_000,
        "eval_freq":    50_000,
        "save_freq":    1_000_000,
        "eval_episodes": 20,
    }

    # Apply any overrides from base_config.yaml[tdmpc2]
    if base_cfg is not None:
        overrides = base_cfg.get("tdmpc2", {})
        defaults.update(overrides)

    cfg = OmegaConf.create(defaults)
    return cfg


def print_config(cfg: DictConfig) -> None:  # pragma: no cover
    """Pretty-print the config to stdout."""
    print("=" * 60)
    print("TD-MPC2 Config — LatentIntercept")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
