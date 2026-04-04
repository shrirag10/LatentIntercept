# TD-MPC2 Zero-Shot Inference Analysis

## Setup

| Parameter | Value |
|-----------|-------|
| Agent | TD-MPC2 (untrained / zero-shot) |
| MPPI horizon | 12 steps (1.2 s lookahead at 10 Hz) |
| MPPI samples | 512 candidates, 64 elites, 6 iterations |
| Policy-seeded trajectories | 24 |
| Latent dim / MLP dim | 512 / 512 |
| Obs space | 24-D (joint pos + vel + gripper + block pos + vel + EE→block relative) |
| Action space | 8-D (7 joint-position deltas + 1 gripper) |
| Block speed | 1.5–3.5 m/s, ±30° launch angle |

All three runs use **zero-shot TD-MPC2** — the world model, encoder, and policy are at their **random initialisation**. MPPI planning is active but operates over a **completely untrained latent dynamics model**, so the 12-step lookahead generates trajectories in a latent space that has no meaningful structure yet. The runs differ only in **episode length** (number of control steps the agent is allowed).

---

## 500 Steps — `trajectory_zero_shot_500.png`

![500](../outputs/trajectory_zero_shot_500.png)

**What happens:**
- The block launches from the right side (~0.6 m) and travels diagonally across the full table, bouncing off the far boundary and ending in the upper-left.
- The end-effector (blue) stays **trapped in a tight cluster** near (-0.5, -0.1), exhibiting dense **high-frequency jitter** with no net displacement toward the block.

**TD-MPC2 interpretation:**
- With a **random world model**, the MPPI rollouts in latent space are essentially **noise**. The 512 sampled action sequences yield reward estimates that are meaningless, so the elite selection picks **arbitrary** trajectories each step.
- The **policy network** (which seeds 24 of the 512 MPPI samples) outputs near-zero-mean actions from random weights, anchoring the arm near home pose.
- The dense jitter pattern is the hallmark of **MPPI with an uninformative cost landscape**: each planning iteration picks a slightly different random direction, producing small oscillatory torques that cancel out over time.
- The arm **never crosses** toward the block's half of the table. No distance-reward gradient is being exploited because the learned value function V(z) is flat at initialisation.

**Verdict:** Pure random-initialisation behavior. MPPI planning adds **no useful lookahead** — the world model latent space is unstructured and the value head is uninformative.

---

## 700 Steps — `trajectory_zero_shot_700.png`

![700](../outputs/trajectory_zero_shot_700.png)

**What happens:**
- Same block trajectory (right → diagonal → upper-left bounce).
- The EE now traces a **large sweeping arc**: it drops to (-0.6, -0.1), swings down and around through the lower-left, curves upward past the block path into the upper-left (~(-0.65, 0.35)), then loops back. The **terminal EE position** ends up in the same quadrant as the block's final location.

**TD-MPC2 interpretation:**
- With 200 more steps of episode budget, the arm has **more time to drift**. The sweeping arc is still **stochastic** (note the jagged edges on the blue curve), but the longer horizon allows accumulated random torques to produce larger excursions.
- The fact that the EE **reaches the block's neighborhood** by the end is **coincidental geometry**, not learned tracking — the arm's reachable workspace naturally extends into the upper-left where the block also ends up after bouncing.
- The MPPI planner is still operating over **random latent dynamics**. The arc shape comes from the Franka's joint-space kinematics under small random delta commands compounding over 700 steps, not from any predictive model of block motion.
- Unlike 500 steps, the jitter here is superimposed on a **net drift** — suggesting the random policy's slight biases in certain joint directions accumulate over the longer episode.

**Verdict:** Extended episode length reveals the **kinematic drift** of random actions through the Franka's joint space. The EE covers more of the workspace but intercept proximity is **accidental**, not purposeful.

---

## 1000 Steps — `trajectory_zero_shot_1000.png`

![1000](../outputs/trajectory_zero_shot_1000.png)

**What happens:**
- Same block trajectory pattern.
- The EE starts at home (-0.35, 0), drifts down and left with jitter, then **pushes into the upper-left** region where the block's path terminates. The EE's final cluster sits at roughly (-0.65, 0.2) — **close to the block's late-stage position** and near the Franka base.

**TD-MPC2 interpretation:**
- At 1000 steps the arm has **even more time** to wander. The trajectory shows two phases: an initial **jittery drift** downward (steps ~0–400), then a **decisive shift** into the upper-left quadrant (steps ~400–1000) where the EE oscillates densely.
- The second phase looks "intentional" but is better explained by **joint-limit saturation**: after enough random deltas accumulate, several joints approach their limits, and the remaining free joints tend to push the EE toward the extremes of the reachable workspace — which happens to overlap with the block's post-bounce region.
- The **constant-velocity block prediction** (red dashed) continues to diverge after the block bounces, confirming that the simple extrapolation overlay is only valid for the first ~0.5 s of block flight. A trained world model would need to learn the bounce dynamics to predict accurately.
- No **CATCH** marker appears — the distance/velocity thresholds (dist < 5 cm, vel_diff < 0.5 m/s) were never simultaneously satisfied, confirming that proximity here is **geometric coincidence**, not a successful intercept.

**Verdict:** Longer episodes let random actions explore more of the workspace, and kinematic structure funnels the EE toward the Franka's natural reach boundaries. **Zero true intercepts** — the untrained world model provides no planning advantage.

---

## Summary: Zero-Shot TD-MPC2 Progression

| Metric | 500 steps | 700 steps | 1000 steps |
|--------|-----------|-----------|------------|
| EE workspace coverage | Tiny cluster near home | Large arc, ~60 % of reachable space | Moderate drift + dense cluster in upper-left |
| Proximity to block end | Far (~0.3 m gap) | Close (same quadrant) | Close (same quadrant) |
| Intercept (CATCH) | No | No | No |
| Motion character | Pure jitter | Jitter + sweeping arc | Jitter + directional drift |
| Source of EE motion | Random MPPI over flat value landscape | Accumulated random deltas through Franka kinematics | Same + joint-limit saturation bias |

**Key takeaway:** TD-MPC2's MPPI planner is only as good as its **learned world model**. At zero-shot (random init), the 12-step latent rollouts carry no predictive information, so the 512-sample, 6-iteration planning loop **degenerates to random search**. Any apparent "pursuit" of the block is an artifact of the Franka's kinematic structure and episode length, not learned behavior. **Training the world model and value function is essential** before MPPI planning can exploit the 1.2 s lookahead for anticipatory interception.

---

## Associated GIFs

| Training steps | GIF (60 s clip) | Time window in recording |
|---------------|------------------|--------------------------|
| 500 | `videos/zero_shot_trajectory_500.gif` | 0:00 – 1:00 |
| 700 | `videos/zero_shot_trajectory_700.gif` | 1:00 – 2:00 |
| 1000 | `videos/zero_shot_trajectory_1000.gif` | 2:00 – 3:00 |

Source: `~/Videos/Screencasts/Screencast from 04-03-2026 03:01:10 PM.webm` (~3 m 19 s)
