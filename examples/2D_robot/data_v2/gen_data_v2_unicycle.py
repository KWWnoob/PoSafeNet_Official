#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio

# ----------------------------
# cbfpy imports (package)
# ----------------------------
from jax import Array
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cbfpy import CBF, CBFConfig


# ============================================================
# Nominal controller (numpy + jax versions)
# ============================================================

def wrap_pi_jnp(a: Array) -> Array:
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi

def u_nom_jnp(z: Array, x_goal: float) -> Array:
    """JAX nominal controller using dst_y stored in the state z[4]."""
    px, py, th, v, dst_y = z
    dx = x_goal - px
    dy = dst_y - py
    th_des = jnp.arctan2(dy, dx)
    e_th = wrap_pi_jnp(th_des - th)
    dist = jnp.sqrt(dx * dx + dy * dy)

    omega = 2.0 * e_th
    v_des = jnp.clip(1.5 * dist, 0.0, 2.0)
    a = 2.0 * (v_des - v)
    return jnp.stack([omega, a])


# ============================================================
# cbfpy Config for unicycle obstacle avoidance (CBF-QP filter)
# ============================================================

class UnicycleObsConfig(CBFConfig):
    """A CLF-CBF config that behaves like a *CBF safety filter* for a unicycle."""

    def __init__(
        self,
        *,
        x_goal: float,
        omega_max: float,
        a_max: float,
        obs_list,
        k1: float,
        k2: float,
        beta: float = 10.0,
        robot_radius: float = 0.0,
    ):
        self.x_goal = float(x_goal)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.beta = float(beta)
        self.robot_radius = float(robot_radius)

        # z = [px, py, theta, v, dst_y], u = [omega, a]
        u_min = (-float(omega_max), -float(a_max))
        u_max = ( float(omega_max),  float(a_max))
        
        obs = jnp.asarray(obs_list, dtype=jnp.float32)
        # CLFCBFConfig API varies slightly by cbfpy version; in the official demo
        # you only pass n, m, u_min, u_max, plus relaxation options.
        super().__init__(
            n=5,
            m=2,
            init_args=(obs,),
            u_min=u_min,
            u_max=u_max,
        )

        # If your cbfpy version requires these explicitly, uncomment:
        # self.num_cbf = 1
        # self.num_clf = 0

    # -------------------------
    # Dynamics: z_dot = f(z) + g(z) u
    # -------------------------
    def f(self, z: ArrayLike) -> Array:
        px, py, th, v, dst_y = z
        return jnp.array([v * jnp.cos(th), v * jnp.sin(th), 0.0, 0.0, 0.0])

    def g(self, z: ArrayLike) -> Array:
        # omega affects theta_dot; a affects v_dot
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])

    def h_2(self, z, obs):
        px, py, th, v, *_ = z

        ox = obs[:, 0]
        oy = obs[:, 1]
        R  = obs[:, 2]

        R_eff = R + self.robot_radius + 0.8 * v

        h = (px - ox)**2 + (py - oy)**2 - R_eff**2
        return h.reshape(-1)

    # CBF class-K function used by cbfpy: alpha(h) >= 0
    def alpha(self, h: ArrayLike) -> Array:
        return self.k2 * jnp.asarray(h)
    
    def H(self, z):
        return jnp.diag(jnp.array([1, 5], dtype=jnp.float64))
    

# ============================================================
# Rollout + dataset
# ============================================================

def step_unicycle(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    px, py, th, v = x
    omega, a = u
    px2 = px + dt * v * np.cos(th)
    py2 = py + dt * v * np.sin(th)
    th2 = th + dt * omega
    v2 = max(0.0, v + dt * a)
    return np.array([px2, py2, th2, v2], dtype=np.float64)

def check_barrier_positive(z: np.ndarray, obs_arr: np.ndarray) -> float:
    """
    Returns min h over all obstacles.
    h = (px-ox)^2 + (py-oy)^2 - R^2
    """
    px, py = z[0], z[1]
    ox = obs_arr[:, 0]
    oy = obs_arr[:, 1]
    R  = obs_arr[:, 2]
    h = (px - ox)**2 + (py - oy)**2 - R**2
    return float(np.min(h))

def rollout_episode(
    *,
    ep_id: int,
    dt: float,
    T: int,
    x0: np.ndarray,
    dst_y: float,
    x_goal: float,
    terminate_dist: float,
    keep_min_steps: int,
    obs_list: list[tuple[float, float, float]],
    ctrl,
) -> np.ndarray | None:
    obs_arr = np.asarray(obs_list, dtype=np.float64).reshape(-1, 3)
    data_rows = []
    x = x0.copy()   

    @jax.jit
    def safe_controller(z, x_goal, z_obs):
        u = u_nom_jnp(z, x_goal)
        return ctrl.safety_filter(z, u, z_obs)
    
    for t in range(T):
        # nominal
        # u_nom = u_nom_np(x, x_goal, dst_y)

        # cbfpy state includes dst_y
        z = np.array([x[0], x[1], x[2], x[3], dst_y], dtype=np.float64)
        u_safe = safe_controller(z, x_goal, obs_arr).reshape(2,) 
        
            # ---------- NEW: barrier check ----------
        h_min = check_barrier_positive(z, obs_arr)
        if h_min < -1e-6:
            print(f"[Barrier violation] ep={ep_id}, t={t}, h_min={h_min:.3e}")
            return None   # ❗️直接丢弃整个 episode
        # --------------------------------------

        # record one row: [px,py,theta,v,dst_y,omega,a]
        data_rows.append([ep_id, t, x[0], x[1], x[2], x[3], dst_y, u_safe[0], u_safe[1]])

        # integrate
        x = step_unicycle(x, u_safe, dt)

        # terminate if near goal (but keep at least some steps)
        if t + 1 >= keep_min_steps:
            if np.hypot(x_goal - x[0], dst_y - x[1]) <= terminate_dist:
                break

    if len(data_rows) == 0:
        return None
    
    # ---- NEW: drop episode if not reached goal ----
    reached = (np.hypot(x_goal - x[0], dst_y - x[1]) <= terminate_dist)
    if not reached:
        # 可选：打印一下，方便你看丢弃率
        print(f"[Drop: not reached] ep={ep_id}, final_dist={np.hypot(x_goal - x[0], dst_y - x[1]):.3f}, steps={len(data_rows)}")
        return None
    # ----------------------------------------------

    return np.asarray(data_rows, dtype=np.float64)

def build_dataset(
    *,
    n_episodes: int,
    dt: float,
    T: int,
    seed: int,
    x_goal: float,
    init_px_low: float,
    init_px_high: float,
    init_py_low: float,
    init_py_high: float,
    v_low: float,
    v_high: float,
    terminate_dist: float,
    keep_min_steps: int,
    obs_list: list[tuple[float, float, float]],
    ctrl,
):
    rng = np.random.default_rng(seed)
    episodes = []
    for ep_id in range(n_episodes):
        px0 = rng.uniform(init_px_low, init_px_high)
        py0 = rng.uniform(init_py_low, init_py_high)
        th0 = rng.uniform(-np.pi, np.pi)
        v0 = rng.uniform(v_low, v_high)
        dst_y = rng.uniform(0.0, 30.0)  # same distribution as many of your scripts

        x0 = np.array([px0, py0, th0, v0], dtype=np.float64)
        ep = rollout_episode(
            ep_id=ep_id,
            dt=dt,
            T=T,
            x0=x0,
            dst_y=float(dst_y),
            x_goal=x_goal,
            terminate_dist=terminate_dist,
            keep_min_steps=keep_min_steps,
            obs_list=obs_list,
            ctrl=ctrl,
        )
        if ep is not None:
            episodes.append(ep)

    if not episodes:
        return np.zeros((0, 7), dtype=np.float64)

    return np.concatenate(episodes, axis=0)

def split_and_save_by_episode(data: np.ndarray, out_dir: Path, prefix: str,
                              train_ratio: float, valid_ratio: float, seed: int = 0):
    # data columns: [ep_id, t, px, py, th, v, dst_y, omega, a]
    ep_ids = np.unique(data[:, 0].astype(int))
    rng = np.random.default_rng(seed)
    rng.shuffle(ep_ids)

    n_ep = len(ep_ids)
    n_train_ep = int(np.floor(train_ratio * n_ep))
    n_valid_ep = int(np.floor(valid_ratio * n_ep))

    train_ids = set(ep_ids[:n_train_ep])
    valid_ids = set(ep_ids[n_train_ep:n_train_ep + n_valid_ep])
    test_ids  = set(ep_ids[n_train_ep + n_valid_ep:])

    train = data[np.isin(data[:, 0].astype(int), list(train_ids))]
    valid = data[np.isin(data[:, 0].astype(int), list(valid_ids))]
    test  = data[np.isin(data[:, 0].astype(int), list(test_ids))]

    sio.savemat(str(out_dir / f"{prefix}_train.mat"), {"data": train})
    sio.savemat(str(out_dir / f"{prefix}_valid.mat"), {"data": valid})
    sio.savemat(str(out_dir / f"{prefix}_test.mat"), {"data": test})

    print("Episode split:")
    print("  #episodes:", n_ep, "train/valid/test:", len(train_ids), len(valid_ids), len(test_ids))
    print("Saved:")
    print("  train:", train.shape, out_dir / f"{prefix}_train.mat")
    print("  valid:", valid.shape, out_dir / f"{prefix}_valid.mat")
    print("  test :", test.shape, out_dir / f"{prefix}_test.mat")



# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(Path.cwd() / "data"))
    ap.add_argument("--prefix", type=str, default="dataM")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_episodes", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=500)

    ap.add_argument("--x_goal", type=float, default=60.0)
    ap.add_argument("--init_px_low", type=float, default=0.0)
    ap.add_argument("--init_px_high", type=float, default=10.0)
    ap.add_argument("--init_py_low", type=float, default=0.0)
    ap.add_argument("--init_py_high", type=float, default=30.0)

    ap.add_argument("--v_low", type=float, default=0.0)
    ap.add_argument("--v_high", type=float, default=2.0)

    ap.add_argument("--terminate_dist", type=float, default=1.0)
    ap.add_argument("--keep_min_steps", type=int, default=10)

    ap.add_argument(
        "--obs",
        type=float,
        nargs=9,
        default=[35, 0, 9, 20, 16, 8, 49, 10, 7],
        help="9 numbers: ox1 oy1 R1 ox2 oy2 R2 ox3 oy3 R3",
    )

    ap.add_argument("--k1", type=float, default=1.0, help="inner gain (build h1)")
    ap.add_argument("--k2", type=float, default=1.0, help="outer gain alpha(h1)=k2*h1")
    ap.add_argument("--beta", type=float, default=10.0, help="softmin sharpness")
    ap.add_argument("--robot_radius", type=float, default=0.0, help="inflate obstacle radii")

    ap.add_argument("--omega_max", type=float, default=0.2)
    ap.add_argument("--a_max", type=float, default=0.5)

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--valid_ratio", type=float, default=0.1)

    args = ap.parse_args()

    obs = np.asarray(args.obs, dtype=np.float64).reshape(3, 3)
    obs_list = [(float(o[0]), float(o[1]), float(o[2])) for o in obs]

    # Build cbfpy controller
    cfg = UnicycleObsConfig(
        x_goal=args.x_goal,
        omega_max=args.omega_max,
        a_max=args.a_max,
        obs_list=obs_list,
        k1=args.k1,
        k2=args.k2,
        beta=args.beta,
        robot_radius=args.robot_radius,
    )
    ctrl = CBF.from_config(cfg)

    data = build_dataset(
        n_episodes=args.n_episodes,
        dt=args.dt,
        T=args.T,
        seed=args.seed,
        x_goal=args.x_goal,
        init_px_low=args.init_px_low,
        init_px_high=args.init_px_high,
        init_py_low=args.init_py_low,
        init_py_high=args.init_py_high,
        v_low=args.v_low,
        v_high=args.v_high,
        terminate_dist=args.terminate_dist,
        keep_min_steps=args.keep_min_steps,
        obs_list=obs_list,
        ctrl=ctrl,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a short demo
    demo_x0 = np.array([0.0, 15.0, 0.0, 0.0], dtype=np.float64)
    demo = rollout_episode(
        ep_id=-1, 
        dt=args.dt,
        T=args.T,
        x0=demo_x0,
        dst_y=15.0,
        x_goal=args.x_goal,
        terminate_dist=args.terminate_dist,
        keep_min_steps=1,
        obs_list=obs_list,
        ctrl=ctrl,
    )
    if demo is not None:
        sio.savemat(str(out_dir / f"{args.prefix}_demo.mat"), {"data": demo})

    split_and_save_by_episode(data, out_dir, args.prefix, args.train_ratio, args.valid_ratio, seed=0)


if __name__ == "__main__":
    main()
