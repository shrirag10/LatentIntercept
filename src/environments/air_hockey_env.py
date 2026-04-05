"""
LatentIntercept — Genesis Air Hockey Environment
================================================
Builds a frictionless-table scene with a Franka Emika arm and a sliding
cube block. Supports N parallel environments via Genesis's native batched
simulation API.

Scene layout (top-down):
  ┌─────────────────────────────┐
  │ Block spawns here → → → → →│
  │            TABLE            │
  │         ← ← Franka arm     │
  └─────────────────────────────┘

Coordinate convention (Genesis default):
  X: along table length (arm → block spawn direction)
  Y: table width (lateral)
  Z: vertical (up)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

import genesis as gs


class AirHockeyEnv:
    """
    Parallel Genesis environment — Franka intercepts a sliding cube block.

    Parameters
    ----------
    n_envs : int
        Number of parallel simulation environments.
    dt : float
        Physics timestep (default 0.01 s → 100 Hz).
    control_freq : int
        Agent action frequency divider; if control_freq=10 the agent acts
        every 10 physics steps (10 Hz effective control).
    episode_len : int
        Maximum number of *control* steps per episode.
    device : str
        Torch device for tensor output ("cuda" or "cpu").
    show_viewer : bool
        If True, open the Genesis interactive 3-D viewer.
    seed : int
        Random seed for domain randomisation.
    cfg : dict, optional
        Optional nested config dict (loaded from base_config.yaml) to
        override default geometry / reward parameters.
    """

    # ── Default geometry & physics constants ────────────────────────────────
    TABLE_SIZE   = (1.5, 0.8, 0.02)   # metres [L, W, H]
    TABLE_POS    = (0.0, 0.0, 0.0)    # table surface at Z≈0
    TABLE_FRICTION = 0.01             # Genesis minimum (1e-2)

    BLOCK_SIZE   = (0.06, 0.06, 0.06) # 6-cm cube
    BLOCK_MASS   = 0.15               # 150 g
    BLOCK_SPAWN_X_RANGE = (0.4, 0.7)  # spawn zone in +X
    BLOCK_SPEED_RANGE   = (1.5, 3.5)  # m/s
    BLOCK_ANGLE_RANGE   = (-30.0, 30.0)  # degrees
    # Terminate when block slides in +X (away from arm); see _block_moving_away
    RESET_ON_BLOCK_MOVING_AWAY = True
    BLOCK_MOVING_AWAY_VX_THRESHOLD = 0.25  # m/s — ignore small numerical drift

    FRANKA_HOME_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    GRIPPER_OPEN  = 0.04             # metres
    CONTACT_FORCE_THRESHOLD = 5.0   # N — contact force for "catch" detection

    # Franka Panda joint limits (rad) — from official datasheet
    FRANKA_JOINT_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    FRANKA_JOINT_UPPER = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]

    POSITION_DELTA_MAX = 0.1  # max joint displacement per control step (rad)

    REWARD_DIST_SCALE = 5.0  # scale factor inside distance exponential

    # ── Observation / action sizes ──────────────────────────────────────────
    OBS_DIM    = 24   # 7 qpos + 7 qvel + 1 gripper + 3 p_block + 3 v_block + 3 rel
    ACTION_DIM = 8    # 7 joint position deltas + 1 gripper command

    def __init__(
        self,
        n_envs: int = 256,
        dt: float = 0.01,
        control_freq: int = 10,
        episode_len: int = 200,
        device: str = "cuda",
        show_viewer: bool = False,
        seed: int = 0,
        cfg: Optional[dict] = None,
    ) -> None:
        self.n_envs       = n_envs
        self.dt           = dt
        self.control_freq = control_freq
        self.episode_len  = episode_len
        self.device       = device
        self.show_viewer  = show_viewer

        # Apply optional config overrides
        if cfg is not None:
            self._apply_cfg(cfg)

        self._rng = np.random.default_rng(seed)
        self._step_count = torch.zeros(n_envs, dtype=torch.int32, device=device)

        self._build_scene()

    # ────────────────────────────────────────────────────────────────────────
    # Scene construction
    # ────────────────────────────────────────────────────────────────────────

    def _build_scene(self) -> None:
        """Initialise Genesis, create the scene, add entities, build."""
        gs.init(backend=gs.cuda if self.device == "cuda" else gs.cpu)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=16),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720),
                camera_pos=(0.0, -2.0, 1.8),
                camera_lookat=(0.0, 0.0, 0.3),
            ),
            show_viewer=self.show_viewer,
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                gravity=(0.0, 0.0, -9.81),
            ),
        )

        # ── Ground plane (invisible, infinite) ──────────────────────────────
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # ── Table surface (visible, frictionless) ───────────────────────────
        table_half = [s / 2 for s in self.TABLE_SIZE]
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size=self.TABLE_SIZE,
                pos=(0.0, 0.0, table_half[2]),
                fixed=True,
            ),
            surface=gs.surfaces.Default(vis_mode="visual"),
            material=gs.materials.Rigid(rho=2700.0, friction=self.TABLE_FRICTION),
        )

        # Table surface Z-offset used for spawning entities on top of the table
        self._table_surface_z = self.TABLE_SIZE[2] + self.BLOCK_SIZE[2] / 2

        # ── Franka Emika arm ────────────────────────────────────────────────
        # Genesis ships with Franka URDF / MJCF; use the built-in morph.
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml",
                           pos=(-0.65, 0.0, self.TABLE_SIZE[2])),
        )

        # Identify arm DOF indices (7 joints) and gripper DOF indices (2 fingers)
        # Genesis names these in order after building; we resolve after build.
        self._arm_dof_indices   = list(range(7))
        self._gripper_dof_indices = [7, 8]   # panda_finger_joint1, panda_finger_joint2

        # ── Block (cube) ─────────────────────────────────────────────────────
        self.block = self.scene.add_entity(
            gs.morphs.Box(
                size=self.BLOCK_SIZE,
                pos=(0.4, 0.0, self._table_surface_z),
            ),
            material=gs.materials.Rigid(
                rho=self.BLOCK_MASS / np.prod(self.BLOCK_SIZE),
                friction=self.TABLE_FRICTION,
            ),
            surface=gs.surfaces.Default(color=(0.9, 0.3, 0.1, 1.0)),
        )

        # ── Table boundary walls ────────────────────────────────────────────
        wall_height = 0.06
        half_L = self.TABLE_SIZE[0] / 2
        half_W = self.TABLE_SIZE[1] / 2
        L_tab = self.TABLE_SIZE[0]
        W_tab = self.TABLE_SIZE[1]
        wall_center_z = self.TABLE_SIZE[2] + wall_height / 2
        wall_surface = gs.surfaces.Default(vis_mode="collision")
        wall_material = gs.materials.Rigid(rho=2700.0, friction=0.01)

        # +X (far end)
        self.scene.add_entity(
            gs.morphs.Box(
                size=(0.02, W_tab, wall_height),
                pos=(half_L + 0.01, 0.0, wall_center_z),
                fixed=True,
            ),
            surface=wall_surface,
            material=wall_material,
        )
        # -X (arm end)
        self.scene.add_entity(
            gs.morphs.Box(
                size=(0.02, W_tab, wall_height),
                pos=(-half_L - 0.01, 0.0, wall_center_z),
                fixed=True,
            ),
            surface=wall_surface,
            material=wall_material,
        )
        # +Y (right)
        self.scene.add_entity(
            gs.morphs.Box(
                size=(L_tab, 0.02, wall_height),
                pos=(0.0, half_W + 0.01, wall_center_z),
                fixed=True,
            ),
            surface=wall_surface,
            material=wall_material,
        )
        # -Y (left)
        self.scene.add_entity(
            gs.morphs.Box(
                size=(L_tab, 0.02, wall_height),
                pos=(0.0, -half_W - 0.01, wall_center_z),
                fixed=True,
            ),
            surface=wall_surface,
            material=wall_material,
        )

        # ── Build with N parallel environments ───────────────────────────────
        self.scene.build(n_envs=self.n_envs)

        # Cache end-effector link index (panda_hand) for FK queries.
        # get_link() returns a Link object; its local index is used with
        # get_links_pos / get_links_vel for efficient batched queries.
        self._ee_link = self.franka.get_link("hand")
        self._ee_link_idx = [self._ee_link.idx_local]

        # Cache joint limits as tensors for position-delta clamping
        self._joint_lower = torch.tensor(
            self.FRANKA_JOINT_LOWER, dtype=torch.float32, device=self.device
        )
        self._joint_upper = torch.tensor(
            self.FRANKA_JOINT_UPPER, dtype=torch.float32, device=self.device
        )

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reset specified environments (or all if env_ids is None).

        Returns
        -------
        obs : torch.Tensor  shape (n_reset, OBS_DIM)
        """
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)

        n_reset = env_ids.shape[0]
        self._step_count[env_ids] = 0

        # ── Reset Franka to home pose ────────────────────────────────────────
        home_q = torch.tensor(
            self.FRANKA_HOME_Q + [self.GRIPPER_OPEN, self.GRIPPER_OPEN],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).expand(n_reset, -1)

        self.franka.set_dofs_position(home_q, envs_idx=env_ids)
        self.franka.set_dofs_velocity(
            torch.zeros(n_reset, 9, device=self.device), envs_idx=env_ids
        )

        # ── Randomise block spawn & launch ───────────────────────────────────
        spawn_pos, launch_vel = self._random_launch(n_reset)

        # Set block pose with zero_velocity=False so we can assign
        # our own initial velocity immediately after.
        self.block.set_pos(spawn_pos, envs_idx=env_ids, zero_velocity=False)
        self.block.set_quat(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0).expand(n_reset, -1),
            envs_idx=env_ids,
            zero_velocity=False,
        )

        # For a free-floating rigid body the 6 DOFs are:
        #   [vx, vy, vz, ωx, ωy, ωz]   (3 linear + 3 angular)
        block_vel_6 = torch.cat(
            [launch_vel, torch.zeros(n_reset, 3, device=self.device)], dim=1
        )  # (n_reset, 6)
        self.block.set_dofs_velocity(block_vel_6, envs_idx=env_ids)

        # One forward simulation step to settle state
        self.scene.step()

        return self._get_obs(env_ids)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Apply action for one control period (control_freq physics steps).

        Parameters
        ----------
        action : torch.Tensor  shape (n_envs, ACTION_DIM)
            [Δq_0..Δq_6, gripper_cmd] — joint position deltas (rad)
            and gripper command in [-1, +1] where +1 = open, -1 = close.

        Returns
        -------
        obs       : (n_envs, OBS_DIM)
        reward    : (n_envs,)
        terminated: (n_envs,)  bool — caught, OOB, block moving away (+X), or timeout
        info      : dict
        """
        action = action.to(self.device).float()
        arm_deltas   = action[:, :7]        # (N, 7) position deltas
        gripper_cmd  = action[:, 7:8]       # (N, 1) in [-1, 1]

        # Compute arm position targets from current pose + delta
        current_q = self.franka.get_dofs_position()[:, :7]
        arm_targets = torch.clamp(
            current_q + arm_deltas,
            self._joint_lower,
            self._joint_upper,
        )

        # Convert normalised gripper command → position target
        gripper_width = self.GRIPPER_OPEN * (0.5 * gripper_cmd + 0.5)  # [0, 0.04] m
        gripper_target = torch.cat([gripper_width, gripper_width], dim=1)  # (N, 2)

        # Apply arm position control (Genesis PD controller handles gravity)
        self.franka.control_dofs_position(
            arm_targets, dofs_idx_local=self._arm_dof_indices
        )
        # Apply gripper position control
        self.franka.control_dofs_position(
            gripper_target, dofs_idx_local=self._gripper_dof_indices
        )

        # Step physics control_freq times
        for _ in range(self.control_freq):
            self.scene.step()

        self._step_count += 1

        obs       = self._get_obs()
        caught    = self._detect_catch()
        out_of_bounds = self._block_out_of_bounds()
        moving_away = self._block_moving_away()
        reward    = self._compute_reward(_caught=caught)
        timeout = self._step_count >= self.episode_len

        terminated = caught | out_of_bounds | moving_away | timeout

        info = {
            "caught":        caught,
            "out_of_bounds": out_of_bounds,
            "moving_away":   moving_away,
            "timeout":       timeout,
        }

        return obs, reward, terminated, info

    # ────────────────────────────────────────────────────────────────────────
    # Observation extraction
    # ────────────────────────────────────────────────────────────────────────

    def _get_obs(
        self,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the 24-dimensional state vector.

          [0 : 7]  — Franka joint angles (rad)
          [7 :14]  — Franka joint velocities (rad/s)
          [14:15]  — Gripper finger mean position (m)
          [15:18]  — Block position [X, Y, Z] (m)
          [18:21]  — Block velocity [Vx, Vy, Vz] (m/s)
          [21:24]  — EE→Block relative vector (m)
        """
        envs_idx = None if env_ids is None else env_ids

        # Joint state
        q    = self.franka.get_dofs_position(envs_idx=envs_idx)  # (N, 9)
        qd   = self.franka.get_dofs_velocity(envs_idx=envs_idx)  # (N, 9)
        arm_q   = q[:, :7]
        arm_qd  = qd[:, :7]
        gripper = q[:, 7:9].mean(dim=1, keepdim=True)  # mean of both finger joints

        # Block state
        p_block = self.block.get_pos(envs_idx=envs_idx)           # (N, 3)
        v_block = self.block.get_vel(envs_idx=envs_idx)           # (N, 3)

        # End-effector position (FK via Genesis link query)
        # get_links_pos returns (N, n_links, 3); we index the single EE link.
        p_ee = self.franka.get_links_pos(
            links_idx_local=self._ee_link_idx, envs_idx=envs_idx
        )[:, 0, :]  # (N, 3)

        rel = p_ee - p_block  # (N, 3)

        obs = torch.cat([arm_q, arm_qd, gripper, p_block, v_block, rel], dim=1)
        return obs  # (N, 24)

    # ────────────────────────────────────────────────────────────────────────
    # Reward
    # ────────────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        env_ids: Optional[torch.Tensor] = None,
        w_dist: float = 1.0,
        w_vel: float  = 0.1,
        success_bonus: float = 10.0,
        dist_scale: Optional[float] = None,
        _caught: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dense composite reward:

          R = w_dist · exp(-dist_scale · ||p_ee - p_block||)
            + w_vel  · exp(-||v_ee - v_block||)
            + success_bonus · I[caught]

        Both exponential terms are always positive and bounded in (0, 1].
        """
        if dist_scale is None:
            dist_scale = self.REWARD_DIST_SCALE
        envs_idx = None if env_ids is None else env_ids

        p_block = self.block.get_pos(envs_idx=envs_idx)
        v_block = self.block.get_vel(envs_idx=envs_idx)
        p_ee    = self.franka.get_links_pos(
            links_idx_local=self._ee_link_idx, envs_idx=envs_idx
        )[:, 0, :]
        v_ee    = self.franka.get_links_vel(
            links_idx_local=self._ee_link_idx, envs_idx=envs_idx
        )[:, 0, :]

        dist = torch.norm(p_ee - p_block, dim=-1)  # (N,)
        vel_diff = torch.norm(v_ee - v_block, dim=-1)  # (N,)

        r_dist = torch.exp(-dist_scale * dist)
        r_vel  = torch.exp(-vel_diff)

        if _caught is not None:
            caught = _caught.float()
        else:
            caught = self._detect_catch(env_ids=envs_idx).float()

        reward = w_dist * r_dist + w_vel * r_vel + success_bonus * caught
        return reward  # (N,)

    # ────────────────────────────────────────────────────────────────────────
    # Termination detection
    # ────────────────────────────────────────────────────────────────────────

    def _detect_catch(
        self,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        A 'catch' is registered when Genesis reports a contact force between
        the block and either Franka finger link that exceeds the threshold,
        AND the gripper is commanded to close (fingers < GRIPPER_OPEN * 0.5).

        Returns bool tensor (N,).
        """
        envs_idx = None if env_ids is None else env_ids

        # Contact forces on the block — returns (N, n_links, 3) for all
        # block links (just one for a primitive box).  Sum across links.
        contact_forces = self.block.get_links_net_contact_force(envs_idx=envs_idx)
        # Shape: (N, n_links, 3) → sum across links → (N, 3)
        if contact_forces.dim() == 3:
            contact_forces = contact_forces.sum(dim=1)
        force_mag = torch.norm(contact_forces, dim=-1)  # (N,)

        # Gripper finger positions
        q_gripper = self.franka.get_dofs_position(envs_idx=envs_idx)[:, 7:9]  # (N, 2)
        finger_mean = q_gripper.mean(dim=1)

        gripper_closing = finger_mean < (self.GRIPPER_OPEN * 0.5)
        caught = (force_mag > self.CONTACT_FORCE_THRESHOLD) & gripper_closing

        return caught

    def _block_out_of_bounds(
        self,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns True for environments where the block has left the table
        area (fell off or slid past the arm side).
        """
        envs_idx = None if env_ids is None else env_ids

        p_block = self.block.get_pos(envs_idx=envs_idx)  # (N, 3)

        half_l = self.TABLE_SIZE[0] / 2
        half_w = self.TABLE_SIZE[1] / 2

        x_oob = (p_block[:, 0] < -half_l - 0.15) | (p_block[:, 0] > half_l + 0.15)
        y_oob = (p_block[:, 1] < -half_w - 0.15) | (p_block[:, 1] > half_w + 0.15)
        z_oob = p_block[:, 2] < -0.05  # fell below table

        return x_oob | y_oob | z_oob

    def _block_moving_away(
        self,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        True when the block's centre-of-mass velocity along +X exceeds the
        threshold. The nominal intercept direction is -X (toward the arm);
        sustained +X motion means the puck is sliding away (e.g. after a miss
        or wall bounce) and the episode can reset early.
        """
        if not self.RESET_ON_BLOCK_MOVING_AWAY:
            envs_idx = None if env_ids is None else env_ids
            n = self.n_envs if envs_idx is None else env_ids.numel()
            return torch.zeros(n, dtype=torch.bool, device=self.device)

        envs_idx = None if env_ids is None else env_ids
        v_block = self.block.get_vel(envs_idx=envs_idx)  # (N, 3)
        return v_block[:, 0] > self.BLOCK_MOVING_AWAY_VX_THRESHOLD

    # ────────────────────────────────────────────────────────────────────────
    # Domain randomisation helpers
    # ────────────────────────────────────────────────────────────────────────

    def _random_launch(
        self,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample randomised spawn position and initial velocity for the block.

        Returns
        -------
        pos : (n, 3)  spawn position on the table surface
        vel : (n, 3)  initial linear velocity (block slides toward arm)
        """
        # Spawn X: somewhere in the far half of the table (+X direction)
        x_lo, x_hi = self.BLOCK_SPAWN_X_RANGE
        spawn_x = self._rng.uniform(x_lo, x_hi, size=n)

        # Spawn Y: random lateral offset within 60% of table width
        half_w = self.TABLE_SIZE[1] / 2 * 0.6
        spawn_y = self._rng.uniform(-half_w, half_w, size=n)

        # Z: on top of the table surface
        spawn_z = np.full(n, self._table_surface_z)

        pos = torch.tensor(
            np.stack([spawn_x, spawn_y, spawn_z], axis=1),
            dtype=torch.float32, device=self.device,
        )

        # Launch speed and angle
        speed = self._rng.uniform(*self.BLOCK_SPEED_RANGE, size=n)
        angle_deg = self._rng.uniform(*self.BLOCK_ANGLE_RANGE, size=n)
        angle_rad = np.deg2rad(angle_deg)

        # Block slides in -X direction toward the arm
        vx = -speed * np.cos(angle_rad)
        vy =  speed * np.sin(angle_rad)
        vz = np.zeros(n)

        vel = torch.tensor(
            np.stack([vx, vy, vz], axis=1),
            dtype=torch.float32, device=self.device,
        )

        return pos, vel

    # ────────────────────────────────────────────────────────────────────────
    # Config helpers
    # ────────────────────────────────────────────────────────────────────────

    def _apply_cfg(self, cfg: dict) -> None:
        """Override class defaults from a parsed YAML config dict."""
        env_cfg   = cfg.get("env_cfg", cfg)   # accept either nested or flat
        block_cfg = env_cfg.get("block", {})
        table_cfg = env_cfg.get("table", {})
        franka_cfg = env_cfg.get("franka", {})
        reward_cfg = env_cfg.get("reward", {})

        if "size" in table_cfg:
            self.TABLE_SIZE = tuple(table_cfg["size"])
        if "friction" in table_cfg:
            self.TABLE_FRICTION = table_cfg["friction"]

        if "size" in block_cfg:
            self.BLOCK_SIZE = tuple(block_cfg["size"])
        if "mass" in block_cfg:
            self.BLOCK_MASS = block_cfg["mass"]
        if "spawn_x_range" in block_cfg:
            self.BLOCK_SPAWN_X_RANGE = tuple(block_cfg["spawn_x_range"])
        if "speed_range" in block_cfg:
            self.BLOCK_SPEED_RANGE = tuple(block_cfg["speed_range"])
        if "angle_range" in block_cfg:
            self.BLOCK_ANGLE_RANGE = tuple(block_cfg["angle_range"])
        if "reset_if_moving_away" in block_cfg:
            self.RESET_ON_BLOCK_MOVING_AWAY = bool(block_cfg["reset_if_moving_away"])
        if "moving_away_vx_threshold" in block_cfg:
            self.BLOCK_MOVING_AWAY_VX_THRESHOLD = float(
                block_cfg["moving_away_vx_threshold"]
            )

        if "home_q" in franka_cfg:
            self.FRANKA_HOME_Q = list(franka_cfg["home_q"])
        if "gripper_open" in franka_cfg:
            self.GRIPPER_OPEN = franka_cfg["gripper_open"]
        if "contact_threshold" in franka_cfg:
            self.CONTACT_FORCE_THRESHOLD = franka_cfg["contact_threshold"]
        if "position_delta_max" in franka_cfg:
            self.POSITION_DELTA_MAX = franka_cfg["position_delta_max"]
        if "joint_lower" in franka_cfg:
            self.FRANKA_JOINT_LOWER = list(franka_cfg["joint_lower"])
        if "joint_upper" in franka_cfg:
            self.FRANKA_JOINT_UPPER = list(franka_cfg["joint_upper"])

        if "dist_scale" in reward_cfg:
            self.REWARD_DIST_SCALE = reward_cfg["dist_scale"]

    # ────────────────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Cleanly shut down Genesis."""
        # Genesis Scene does not expose a close() method.
        # Deletion of the scene object handles cleanup implicitly.
        del self.scene

    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        return (self.OBS_DIM,)

    @property
    def action_space_shape(self) -> tuple[int, ...]:
        return (self.ACTION_DIM,)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AirHockeyEnv(n_envs={self.n_envs}, "
            f"obs={self.OBS_DIM}D, action={self.ACTION_DIM}D)"
        )
