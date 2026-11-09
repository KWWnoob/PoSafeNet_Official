#!/usr/bin/env python3
"""gen_data_cbfpy_pkg.py

manipulator dataset generation using cbfpy package for obstacle avoidance.
z = [theta1, omega1, theta2, omega2, dst_x, dst_y]
u = [domega1, domega2]

"""

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
from typing import Tuple    

from cbfpy import CLFCBFConfig, CLFCBF


# ============================================================
# Nominal controller (numpy + jax versions)
# ============================================================

def wrap_pi_jnp(a: Array) -> Array:
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi

def fk_elbow_ee(th1: jnp.ndarray, th2: jnp.ndarray, l1: float, l2: float):
    # MATLAB-style: theta2 is ABSOLUTE angle
    x_el = l1 * jnp.cos(th1)
    y_el = l1 * jnp.sin(th1)
    x_ee = x_el + l2 * jnp.cos(th2)
    y_ee = y_el + l2 * jnp.sin(th2)
    return jnp.array([x_el, y_el]), jnp.array([x_ee, y_ee])

def link_sample_points_jnp(q: jnp.ndarray, l1: float, l2: float, n_points: int) -> jnp.ndarray:
    """
    Sample points along BOTH links (MATLAB-style absolute angles):
      link1: base -> elbow
      link2: elbow -> end-effector
    Returns:
      points: (2*n_points, 2)
    """
    th1, th2 = q[0], q[1]
    elbow_pos, ee_pos = fk_elbow_ee(th1, th2, l1, l2)

    base_pos = jnp.array([0.0, 0.0], dtype=jnp.float64)

    # alphas in [0,1]
    alphas = jnp.linspace(0.0, 1.0, n_points, dtype=jnp.float64)  # (n_points,)

    # link1 points: base + a*(elbow-base)
    link1 = base_pos[None, :] + alphas[:, None] * (elbow_pos - base_pos)[None, :]

    # link2 points: elbow + a*(ee-elbow)
    link2 = elbow_pos[None, :] + alphas[:, None] * (ee_pos - elbow_pos)[None, :]

    return jnp.concatenate([link1, link2], axis=0)

def step_double_integrator(q: np.ndarray, dq: np.ndarray, u: np.ndarray, dt: float):
    """
    RK4 integrator for:
      q_dot  = dq
      dq_dot = u

    Inputs:
      q  (2,)
      dq (2,)
      u  (2,)
    Returns:
      q_next, dq_next
    """
    # pack state
    z0 = np.concatenate([q, dq], axis=0)  # (4,)

    def f(z):
        q_ = z[0:2]
        dq_ = z[2:4]
        dz = np.zeros_like(z)
        dz[0:2] = dq_
        dz[2:4] = u
        return dz

    k1 = f(z0)
    k2 = f(z0 + 0.5 * dt * k1)
    k3 = f(z0 + 0.5 * dt * k2)
    k4 = f(z0 + dt * k3)

    z1 = z0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    q_next = z1[0:2]
    dq_next = z1[2:4]
    return q_next, dq_next


# ============================================================
# cbfpy Config for unicycle obstacle avoidance (CBF-QP filter)
# ============================================================

class TwoLinkCLFCBFConfing(CLFCBFConfig):
    def __init__(
        self,
        *,
        l1: float,
        l2: float,
        k1: float,
        k2: float,
        obs: jnp.ndarray,
        phi_min: float,
        phi_max: float,
        p_goal: jnp.ndarray,
        
    ):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.p_goal = p_goal.astype(jnp.float64)

        self.obs = jnp.asarray(obs, dtype=jnp.float64)
        super().__init__(
            n=6,
            m=2,
        )

    def f(self, z: ArrayLike) -> Array:
        th1, w1, th2, w2, dst_x, dst_y = z
        return jnp.array([w1, 0.0, w2, 0.0, 0.0, 0.0])
    
    
    def g(self, z):
        return jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
    
    # CLF: Tip to the destination
    def V_2(self, z: ArrayLike, z_des: ArrayLike) -> Array:
        th1, w1, th2, w2, dst_x, dst_y = z

        # --- end-effector position ---
        x = self.l1 * jnp.cos(th1) + self.l2 * jnp.cos(th2)
        y = self.l1 * jnp.sin(th1) + self.l2 * jnp.sin(th2)

        # --- end-effector velocity (chain rule) ---
        dx = -self.l1 * jnp.sin(th1) * w1 - self.l2 * jnp.sin(th2) * w2
        dy =  self.l1 * jnp.cos(th1) * w1 + self.l2 * jnp.cos(th2) * w2

        # --- position error ---
        ex = x - dst_x
        ey = y - dst_y

        # --- MATLAB-style PD terms: (velocity + k * position_error) ---
        term1 = dx + 0.4 * ex
        term2 = dy + 0.45 * ey

        # CLF value (MATLAB uses term1^2 + term2^2)
        V = term1**2 + term2**2
        return V.reshape(-1)
    
    def gamma(self, V_2: ArrayLike) -> Array:
        return self.k1 * jnp.asarray(V_2)

    # CBF: Obstacle avoidance, energy limits, joint limits

    # def h_2(self, z: ArrayLike) -> Array:
    #     th1, _, th2, _, _, _ = z

    #     # ONLY tip
    #     p_tip = jnp.array([
    #         self.l1 * jnp.cos(th1) + self.l2 * jnp.cos(th2),
    #         self.l1 * jnp.sin(th1) + self.l2 * jnp.sin(th2),
    #     ], dtype=jnp.float64)

    #     def h_obs(o):
    #         ox, oy, R = o
    #         return (p_tip[0] - ox)**2 + (p_tip[1] - oy)**2 - R**2

    #     h0_body_vec = jax.vmap(h_obs)(self.obs)   # (N_obs,)

    #     # phi constraints (optional, keep if you want)
    #     phi = wrap_pi_jnp(th2 - th1)
    #     h0_phi_min = phi - self.phi_min
    #     h0_phi_max = self.phi_max - phi

    #     h_vec = jnp.concatenate([h0_body_vec, jnp.array([h0_phi_min, h0_phi_max])], axis=0)
    #     return h_vec.reshape(-1)
    
    def h_2(self, z):
        th1, _, th2, _, _, _ = z

        # elbow position
        p_el = jnp.array([self.l1*jnp.cos(th1),
                        self.l1*jnp.sin(th1)], dtype=jnp.float64)

        # tip position
        p_tip = jnp.array([self.l1*jnp.cos(th1) + self.l2*jnp.cos(th2),
                        self.l1*jnp.sin(th1) + self.l2*jnp.sin(th2)], dtype=jnp.float64)

        def h_point_obs(p, o):
            ox, oy, R = o
            return (p[0]-ox)**2 + (p[1]-oy)**2 - R**2

        # obstacle constraints for elbow and tip
        h_el  = jax.vmap(lambda o: h_point_obs(p_el,  o))(self.obs)   # (N_obs,)
        h_tip = jax.vmap(lambda o: h_point_obs(p_tip, o))(self.obs)   # (N_obs,)

        # phi constraints (你的)
        phi = wrap_pi_jnp(th2 - th1)
        h_phi_min = phi - self.phi_min
        h_phi_max = self.phi_max - phi

        # return jnp.concatenate([h_el, h_tip, jnp.array([h_phi_min, h_phi_max])], axis=0)
        return h_tip.reshape(-1)


    
    def alpha(self, h_2: ArrayLike) -> Array:
        return self.k2  * jnp.asarray(h_2)

    def H(self, z):
        # 强惩罚控制能量（类似 MATLAB 里 200 的量级）
        return jnp.diag(jnp.array([200.0, 200.0], dtype=jnp.float64))


# ============================================================
# Rollout + dataset
# ============================================================

def rollout_episode(
    *,
    ep_id: int,
    dt: float,
    T: int,
    x0: np.ndarray,                 # (4,) = [th1, w1, th2, w2]
    dst_xy: tuple[float, float],    # (dst_x, dst_y)
    l1,
    l2,
    clf_cbf: CLFCBF,
    terminate_tol: float | None = None,   # 可选：末端到目标距离阈值
    keep_min_steps: int = 0,
) -> np.ndarray:
    """
    Returns array of shape (T_ep, 10):
      [ep_id, t, th1, w1, th2, w2, dst_x, dst_y, u1, u2]
    """


    th1, w1, th2, w2 = x0.astype(np.float64)
    dst_x, dst_y = float(dst_xy[0]), float(dst_xy[1])

    data_rows = []

    # JIT the controller call. IMPORTANT:
    # controller signature is: u = clf_cbf.controller(z, z_des)
    @jax.jit
    def policy(z_jax, z_des_jax):
        return clf_cbf.controller(z_jax, z_des_jax)

    for t in range(T):
        # z: [th1,w1,th2,w2,dst_x,dst_y]
        z = jnp.array([th1, w1, th2, w2, dst_x, dst_y], dtype=jnp.float64)

        # z_des: also 6D; put goal in last two dims (so CLF can use it if needed)
        z_des = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)

        u = policy(z, z_des)
        u = jnp.asarray(u).reshape(-1)
        u_np = np.array(u, dtype=np.float64)  # (2,) = [dw1, dw2]

        data_rows.append([ep_id, t, th1, w1, th2, w2, dst_x, dst_y, u_np[0], u_np[1]])

        # integrate double-integrator with RK4 (your function)
        q = np.array([th1, th2], dtype=np.float64)
        dq = np.array([w1, w2], dtype=np.float64)
        q_next, dq_next = step_double_integrator(q, dq, u_np, dt)

        th1, th2 = float(q_next[0]), float(q_next[1])
        w1, w2   = float(dq_next[0]), float(dq_next[1])

        # optional early terminate: end-effector close to goal
        if terminate_tol is not None and (t + 1) >= keep_min_steps:
            # compute EE position in numpy (or use jax, either is fine)
            ee = np.array([
                l1*np.cos(th1) + l2*np.cos(th2),
                l1*np.sin(th1) + l2*np.sin(th2),
            ], dtype=np.float64)
            if np.linalg.norm(ee - np.array([dst_x, dst_y])) <= terminate_tol:
                break

    return np.asarray(data_rows, dtype=np.float64)

def sample_goal_in_disk(rng: np.random.Generator, cx: float, cy: float, r: float) -> tuple[float, float]:
    """
    Uniformly sample a point inside a disk centered at (cx,cy) with radius r.
    """
    ang = rng.uniform(0.0, 2.0 * np.pi)
    rad = r * np.sqrt(rng.uniform(0.0, 1.0))  # sqrt for uniform area
    x = cx + rad * np.cos(ang)
    y = cy + rad * np.sin(ang)
    return float(x), float(y)

def build_dataset(
    *,
    n_episodes: int,
    dt: float,
    T: int,
    seed: int,
    # init ranges
    th1_low: float, th1_high: float,
    th2_low: float, th2_high: float,
    w1_low: float,  w1_high: float,
    w2_low: float,  w2_high: float,
    l1:float,
    l2:float,
    # goal ranges
    goal_cx: float,
    goal_cy: float,
    goal_r: float,
    obs_list: list[tuple[float, float, float]],
    clf_cbf: CLFCBF,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    episodes = []

    for ep_id in range(n_episodes):
        th1 = rng.uniform(th1_low, th1_high)
        th2 = rng.uniform(th2_low, th2_high)
        w1  = rng.uniform(w1_low,  w1_high)
        w2  = rng.uniform(w2_low,  w2_high)

        dst_x, dst_y = sample_goal_in_disk(rng, goal_cx, goal_cy, goal_r)

        x0 = np.array([th1, w1, th2, w2], dtype=np.float64)
        ep = rollout_episode(
            ep_id=ep_id,
            dt=dt,
            T=T,
            x0=x0,
            dst_xy=(float(dst_x), float(dst_y)),
            l1=l1,
            l2=l2,
            clf_cbf=clf_cbf,
            terminate_tol=0.05,       # 先不提前终止，稳定跑通
            keep_min_steps=0,
        )
        episodes.append(ep)

    return np.concatenate(episodes, axis=0) if episodes else np.zeros((0, 10), dtype=np.float64)


def split_and_save_by_episode(
    data: np.ndarray,
    out_dir: Path,
    prefix: str,
    train_ratio: float,
    valid_ratio: float,
    seed: int = 0,
):
    # data columns:
    # [ep_id, t, th1, w1, th2, w2, dst_x, dst_y, u1, u2]
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
    sio.savemat(str(out_dir / f"{prefix}_test.mat"),  {"data": test})

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

    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--T", type=int, default=400)

    # arm
    ap.add_argument("--l1", type=float, default=3.0)
    ap.add_argument("--l2", type=float, default=3.0)

    # init ranges
    ap.add_argument("--th1_low",  type=float, default=-np.pi - 0.15)
    ap.add_argument("--th1_high", type=float, default=-np.pi + 0.15)

    ap.add_argument("--th2_low",  type=float, default=-np.pi - 0.15)
    ap.add_argument("--th2_high", type=float, default=-np.pi + 0.15)

    ap.add_argument("--w1_low", type=float, default=-0.05)
    ap.add_argument("--w1_high", type=float, default=0.05)

    ap.add_argument("--w2_low", type=float, default=-0.05)
    ap.add_argument("--w2_high", type=float, default=0.05)

    # goal ranges
    ap.add_argument("--goal_cx", type=float, default=3.5)
    ap.add_argument("--goal_cy", type=float, default=3.5)
    ap.add_argument("--goal_r",  type=float, default=0.8)

    ap.add_argument(
        "--obs",
        type=float,
        nargs="+",
        default=[0, 7.0, 4],  # two obstacles by default
        help="3*k numbers: ox1 oy1 R1 ox2 oy2 R2 ...",
    )

    ap.add_argument("--k1", type=float, default=1.0)
    ap.add_argument("--k2", type=float, default=0.6)
    ap.add_argument("--phi_min", type=float, default=-2.5)
    ap.add_argument("--phi_max", type=float, default=2.5)

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--valid_ratio", type=float, default=0.1)

    args = ap.parse_args()

    # make obs_list
    obs = np.asarray(args.obs, dtype=np.float64).reshape(-1, 3)
    obs_list = [(float(o[0]), float(o[1]), float(o[2])) for o in obs]

    # Build controller (THIS is the key fix)
    config = TwoLinkCLFCBFConfing(
        l1=args.l1,
        l2=args.l2,
        k1=args.k1,
        k2=args.k2,
        obs=jnp.asarray(obs, dtype=jnp.float64),
        phi_min=args.phi_min,
        phi_max=args.phi_max,
        p_goal=jnp.array([0.0, 0.0], dtype=jnp.float64),
    )
    clf_cbf = CLFCBF.from_config(config)

    data = build_dataset(
        n_episodes=args.n_episodes,
        dt=args.dt,
        T=args.T,
        seed=args.seed,
        th1_low=args.th1_low, th1_high=args.th1_high,
        th2_low=args.th2_low, th2_high=args.th2_high,
        w1_low=args.w1_low,   w1_high=args.w1_high,
        w2_low=args.w2_low,   w2_high=args.w2_high,
        l1 = args.l1,
        l2 = args.l2,
        goal_cx=args.goal_cx,
        goal_cy=args.goal_cy,
        goal_r=args.goal_r,
        obs_list=obs_list,
        clf_cbf=clf_cbf,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save demo
    demo_x0 = np.array([np.pi, 0.0, np.pi, 0.0], dtype=np.float64)

    rng_demo = np.random.default_rng(args.seed + 999)
    demo_goal = sample_goal_in_disk(rng_demo, args.goal_cx, args.goal_cy, args.goal_r)

    demo = rollout_episode(
        ep_id=-1,
        dt=args.dt,
        T=args.T,
        x0=demo_x0,
        dst_xy=demo_goal,
        l1=args.l1,
        l2=args.l2,
        clf_cbf=clf_cbf,
        terminate_tol=0.1,
        keep_min_steps=0,
    )
    sio.savemat(str(out_dir / f"{args.prefix}_demo.mat"), {"data": demo})

    split_and_save_by_episode(data, out_dir, args.prefix, args.train_ratio, args.valid_ratio, seed=0)


if __name__ == "__main__":
    main()

