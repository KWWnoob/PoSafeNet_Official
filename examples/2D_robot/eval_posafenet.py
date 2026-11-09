#!/usr/bin/env python3
# Closed-loop Monte-Carlo evaluation of PoSafeNet on the 2D robot navigation task.
#
#   python eval_posafenet.py                       # both variants, default checkpoints
#   python eval_posafenet.py --methods PoSafeNet_Hard --hard-ckpt path/to/model_posa19.pth
#
# ✅ ALIGNED LIKE YOUR FIRST BENCHMARK:
#   - succ/fail are ROLLOUT-level (episode/rollout success):
#       succ = number of rollouts that are FULLY SUCCESSFUL
#              (no collision AND no qp-fail through the horizon)
#       fail = number of rollouts that FAIL (collision or qp-fail or nonfinite)
#
# ✅ IMPORTANT:
#   The following metrics are computed ONLY from rollouts that are FULLY SUCCESSFUL:
#     "safety_min", "safety_mean", "safety_var",
#     "final_dst_mean", "final_dst_min", "final_dst_var",
#     "time_mean", "time_var",
#     "unc_u1", "unc_u2",
#     "ep_len_mean", "ep_len_min", "ep_len_max",
#     "steps_mean", "steps_min", "steps_max",
#     "tf_mse"  (still computed over qp-success rows only, as before)
#
# ✅ EXTRA DIAGNOSTICS (NOT used for stats):
#   - qp_calls / qp_succ_calls / qp_fail_calls (call-level accounting)

from __future__ import annotations

import csv
import io
import time
import contextlib
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import scipy.io as sio
import torch

import models  # your models.py


# ======================
# 0) Config (edit here)
# ======================
RNG = np.random.default_rng(0)

_THIS = Path(__file__).resolve()
DATA_DIR = _THIS.parent / "data_v2" / "data"

TEST_MAT  = DATA_DIR / "dataM_test.mat"
TRAIN_MAT = DATA_DIR / "dataM_train.mat"

# 3 obstacles (ox, oy, R)
OBS_LIST = [
    (35.0, 0.0, 9.0),
    (20.0, 16.0, 8.0),
    (49.0, 10.0, 7.0),
]

GOAL_X = 60.0
DT = 0.1
HORIZON_STEPS = 400

# -------- Failure handling --------
TREAT_QPTH_WARNING_AS_FAIL = True
TREAT_NONFINITE_AS_FAIL = True
FAIL_VERBOSE = False

# -------- Monte Carlo repeats --------
MC_RUNS = 100
ENABLE_INTERNAL_NOISE = True

# -------- Which methods to run --------
METHODS = [
    "PoSafeNet_Hard",
    "PoSafeNet_Mixture",
]

# default checkpoints = final epoch written by train_posafenet.py --exp 0
CKPT_MAP = {
    "PoSafeNet_Hard": str(_THIS.parent / "log/posafenet_hard/posafenet_hard00/model_posa19.pth"),
    "PoSafeNet_Mixture": str(_THIS.parent / "log/posafenet_mixture/posafenet_mixture00/model_posa19.pth"),
}
POSA_HEADS = 6

# output
OUT_DIR = _THIS.parent / "results"
OUT_CSV = OUT_DIR / f"metrics_mc_runs{MC_RUNS}_noise{int(ENABLE_INTERNAL_NOISE)}.csv"

# model arch (same as your train script)
nFeatures, nHidden1, nHidden21, nHidden22, nCls = 5, 128, 32, 32, 2

# If ABNet, set heads and pass itr
ABNET_HEADS = 5
ABNET_ITR_INFER = 0

# Device / dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64


# ======================
# 1) qpth warning capture
# ======================
_QP_WARN_SUBSTRS = [
    "qpth warning",
    "returning an inaccurate and potentially incorrect solution",
    "some residual is large",
    "your problem may be infeasible or difficult",
    "iterative refinement",
    "infeasible",
]


# ======================
# 2) helper fns
# ======================
def safety_h(px: float, py: float) -> Tuple[float, np.ndarray]:
    """Return (min_h_over_obs, h_vec[3]) where h = dist^2 - R^2."""
    hs = []
    for ox, oy, R in OBS_LIST:
        hs.append((px - ox) ** 2 + (py - oy) ** 2 - R ** 2)
    hs = np.asarray(hs, dtype=np.float64)
    return float(hs.min()), hs

def dist_to_goal(px: float, py: float, goaly: float) -> float:
    return float(np.sqrt((px - GOAL_X) ** 2 + (py - goaly) ** 2))

def unicycle_step(px, py, theta, v, omega, a, dt=DT):
    px    = px + dt * (v * np.cos(theta))
    py    = py + dt * (v * np.sin(theta))
    theta = theta + dt * omega
    v     = v + dt * a
    return px, py, theta, v

def safe_model_call(model, Xn: torch.Tensor, is_abnet: bool) -> Tuple[Optional[torch.Tensor], bool, str]:
    """
    Calls model(...) but catches:
      - exceptions
      - qpth warnings (printed or warnings.warn)
      - nonfinite outputs
    Returns:
      (u, ok, reason)
    """
    try:
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        warned = False

        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                if is_abnet:
                    u = model(Xn, 0, ABNET_ITR_INFER)
                else:
                    u = model(Xn, 0)

                for wi in w:
                    msg = str(wi.message).lower()
                    if any(s in msg for s in _QP_WARN_SUBSTRS):
                        warned = True
                        break

        printed = (buf_out.getvalue() + "\n" + buf_err.getvalue()).lower()
        if any(s in printed for s in _QP_WARN_SUBSTRS):
            warned = True

        if TREAT_QPTH_WARNING_AS_FAIL and warned:
            return None, False, "qpth_warning"

        if u is None:
            return None, False, "u_none"

        if not isinstance(u, torch.Tensor) or u.numel() != 2:
            return None, False, "bad_u_shape"

        u_np = u.detach().cpu().numpy()
        if TREAT_NONFINITE_AS_FAIL and (not np.all(np.isfinite(u_np))):
            return None, False, "nonfinite_u"

        return u, True, "ok"

    except Exception as e:
        return None, False, f"exception:{type(e).__name__}"

def _nanpad_uncertainty(ctrl_profiles: List[np.ndarray]) -> Tuple[float, float]:
    """
    ctrl_profiles: list of (T,2) arrays, variable T (ONLY from ok-rollouts)
    returns:
      unc_u1, unc_u2 (time-avg std over successful rollouts, NaN padded)
    """
    succ = len(ctrl_profiles)
    if succ <= 1:
        return float("nan"), float("nan")

    lengths = [c.shape[0] for c in ctrl_profiles]
    max_T = int(max(lengths))

    U = np.full((succ, max_T, 2), np.nan, dtype=np.float64)
    for s, c in enumerate(ctrl_profiles):
        Ti = c.shape[0]
        U[s, :Ti, :] = c

    std_t = np.nanstd(U, axis=0)      # (max_T,2)
    unc = np.nanmean(std_t, axis=0)   # (2,)
    return float(unc[0]), float(unc[1])

def _mean(x: List[float]) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.mean(np.asarray(x, dtype=np.float64)))

def _var(x: List[float]) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.var(np.asarray(x, dtype=np.float64)))

def _min(x: List[float]) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.min(np.asarray(x, dtype=np.float64)))

def _max(x: List[float]) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.max(np.asarray(x, dtype=np.float64)))


# ======================
# 3) load dataset + normalization
# ======================
train_data = sio.loadmat(str(TRAIN_MAT))["data"]
test_data  = sio.loadmat(str(TEST_MAT))["data"]

# columns: [ep_id, t, px, py, theta, v, goaly, omega, a]
train0 = np.double(train_data[:, 2:7])  # [px,py,theta,v,goaly]
mean = np.mean(train0, axis=0)
std  = np.std(train0, axis=0) + 1e-12

test0 = np.double(test_data[:, 2:7])
test_labels = np.double(test_data[:, 7:9])  # [omega,a]
test0n = (test0 - mean) / std

ep_ids = test_data[:, 0].astype(np.int64)
t_col  = test_data[:, 1].astype(np.float64)
unique_eps = np.unique(ep_ids)

def sample_episode_idx(rng: np.random.Generator) -> Tuple[int, np.ndarray]:
    ep = int(rng.choice(unique_eps))
    idx = np.where(ep_ids == ep)[0]
    idx = idx[np.argsort(t_col[idx])]
    return ep, idx


# ======================
# 4) build model
# ======================
def build_model(method: str):
    ckpt = Path(CKPT_MAP[method])
    assert ckpt.exists(), f"Missing checkpoint for {method}"

    if method == "PoSafeNet_Hard":
        m = models.PoSafeNet_Hard(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads=POSA_HEADS).to(device)
        is_ab = False
    elif method == "PoSafeNet_Mixture":
        m = models.PoSafeNet_Mixture(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads=POSA_HEADS).to(device)
        is_ab = False
    else:
        raise ValueError(method)

    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()

    # internal noise toggle
    if hasattr(m, "add_noise"):
        m.add_noise = bool(ENABLE_INTERNAL_NOISE)

    print(f"[Loaded] {method} from {ckpt} | add_noise={getattr(m, 'add_noise', 'N/A')}")
    return m, is_ab


# ======================
# 5) rollout (whole episode length)
#    + qp_calls / qp_fails are diagnostics
#    + ok=True means FULL rollout success (no collision, no qp fail)
# ======================
def rollout_closed_loop(model, is_abnet: bool, rng: np.random.Generator) -> Dict[str, Any]:
    ep, idx = sample_episode_idx(rng)

    px, py, theta, v, goaly = map(float, test0[idx[0], :])
    T_ep = len(idx)

    collide = False
    fail_reason = "ok"
    fail_step = -1

    ctrl: List[List[float]] = []
    safety_steps: List[float] = []

    qp_calls = 0
    qp_fails = 0

    t_start = time.time()
    with torch.no_grad():
        for k in range(min(T_ep, HORIZON_STEPS)):
            hmin, _ = safety_h(px, py)
            if hmin < 0.0:
                collide = True
                fail_reason = "collision"
                fail_step = k
                break

            x_raw = np.array([px, py, theta, v, goaly], dtype=np.float64)
            x_n = (x_raw - mean) / std
            Xn = torch.from_numpy(x_n).to(device=device, dtype=dtype).view(1, -1)

            qp_calls += 1
            u, ok, reason = safe_model_call(model, Xn, is_abnet)
            if not ok:
                qp_fails += 1
                fail_reason = reason
                fail_step = k
                break

            omega = float(u[0, 0].item())
            a     = float(u[0, 1].item())

            if TREAT_NONFINITE_AS_FAIL and (not np.isfinite(omega) or not np.isfinite(a)):
                qp_fails += 1
                fail_reason = "nonfinite_u"
                fail_step = k
                break

            # ✅ QP-success step => record safety for this step
            safety_steps.append(float(hmin))

            ctrl.append([omega, a])
            px, py, theta, v = unicycle_step(px, py, theta, v, omega, a, dt=DT)

    runtime = time.time() - t_start
    final_dst = dist_to_goal(px, py, goaly)

    ok_rollout = (fail_reason == "ok") and (not collide)

    # only meaningful when ok_rollout=True, but safe to compute anyway
    safety_min_rollout = float(np.min(np.asarray(safety_steps, dtype=np.float64))) if len(safety_steps) else float("nan")

    return dict(
        ok=bool(ok_rollout),
        ep_id=int(ep),
        ep_len=int(T_ep),
        steps=int(len(ctrl)),
        fail_reason=str(fail_reason),
        fail_step=int(fail_step),

        # rollout-level values (only used if ok=True)
        safety_min=float(safety_min_rollout),
        final_dst=float(final_dst),
        runtime=float(runtime),
        ctrl=np.asarray(ctrl, dtype=np.float64) if len(ctrl) else None,  # (T,2)
        safety_steps=np.asarray(safety_steps, dtype=np.float64),

        # qp-level diagnostics
        qp_calls=int(qp_calls),
        qp_fails=int(qp_fails),
    )


# ======================
# 6) teacher forcing mse (row-level)
#    IMPORTANT: compute tf_mse over QP-success rows only
# ======================
def teacher_forcing_mse_qp_success_only(model, is_abnet: bool, rng: np.random.Generator, N: int = 5000) -> float:
    """
    Returns MSE(pred, Y) computed ONLY on rows where safe_model_call succeeds.
    If too few qp-success rows exist, returns NaN.
    """
    N = min(N, len(test0n))
    idx = rng.choice(len(test0n), size=N, replace=False)

    X_np = test0n[idx]                 # normalized input
    Y_np = test_labels[idx]            # labels in same scale as model output (your assumption)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for i in range(len(idx)):
            Xn = torch.from_numpy(X_np[i]).to(device=device, dtype=dtype).view(1, -1)
            _, ok, _ = safe_model_call(model, Xn, is_abnet)
            if not ok:
                continue
            xs.append(X_np[i].copy())
            ys.append(Y_np[i].copy())

    if len(xs) == 0:
        return float("nan")

    X = torch.from_numpy(np.stack(xs, axis=0)).to(device=device, dtype=dtype)
    Y = torch.from_numpy(np.stack(ys, axis=0)).to(device=device, dtype=dtype)

    with torch.no_grad():
        if is_abnet:
            pred = model(X, 0, ABNET_ITR_INFER)
        else:
            pred = model(X, 0)

    return float(torch.mean((pred - Y) ** 2).item())


# ======================
# 7) benchmark runner
#    ✅ succ/fail = rollout-level (like your first benchmark)
#    ✅ metrics = computed ONLY from ok-rollouts (FULL QP-success rollouts)
#    ✅ qp_*_calls = diagnostics only
# ======================
def run_benchmark_one_method(method: str) -> Dict[str, Any]:
    model, is_ab = build_model(method)

    # tf_mse computed only on qp-success rows
    tf_mse = teacher_forcing_mse_qp_success_only(model, is_ab, RNG, N=5000)

    # SUCCESS-only profiles (FULL rollout success)
    safety_min_profiles: List[float] = []
    final_dst_profiles:  List[float] = []
    time_profiles:       List[float] = []
    ep_len_profiles:     List[int]   = []
    steps_profiles:      List[int]   = []
    ctrl_profiles:       List[np.ndarray] = []  # (T,2) from ok rollouts only

    rollout_fail = 0
    rollout_succ = 0
    fails: List[Tuple[str, int, int]] = []

    # qp-level diagnostics over all rollouts
    qp_calls_total = 0
    qp_fails_total = 0

    for i in range(MC_RUNS):
        out = rollout_closed_loop(model, is_ab, RNG)

        qp_calls_total += int(out.get("qp_calls", 0))
        qp_fails_total += int(out.get("qp_fails", 0))

        if not out["ok"]:
            rollout_fail += 1
            fails.append((
                str(out.get("fail_reason", "unknown")),
                int(out.get("fail_step", -1)),
                int(out.get("ep_id", -1)),
            ))
            if FAIL_VERBOSE:
                print(f"[ROLL FAIL] {method} ep={out.get('ep_id')} step={out.get('fail_step')} reason={out.get('fail_reason')}")
            continue

        # ✅ ONLY ok-rollouts contribute to these metrics
        rollout_succ += 1
        safety_min_profiles.append(float(out["safety_min"]))
        final_dst_profiles.append(float(out["final_dst"]))
        time_profiles.append(float(out["runtime"]))
        ep_len_profiles.append(int(out["ep_len"]))
        steps_profiles.append(int(out["steps"]))
        if out["ctrl"] is not None:
            ctrl_profiles.append(np.asarray(out["ctrl"], dtype=np.float64))

    # uncertainty computed only across ok-rollouts
    unc_u1, unc_u2 = _nanpad_uncertainty(ctrl_profiles)

    # qp-level diagnostics
    qp_succ_calls_total = int(qp_calls_total - qp_fails_total)

    res = dict(
        method=method,

        # ✅ LIKE your first benchmark: succ/fail aligned to SUCCESS-only stats
        succ=int(rollout_succ),
        fail=int(rollout_fail),

        # diagnostics
        qp_calls=int(qp_calls_total),
        qp_succ_calls=int(qp_succ_calls_total),
        qp_fail_calls=int(qp_fails_total),
        rollout_succ=int(rollout_succ),
        rollout_fail=int(rollout_fail),
        fails_preview=fails[:5],

        # ✅ ONLY ok-rollouts:
        safety_min=_min(safety_min_profiles),
        safety_mean=_mean(safety_min_profiles),
        safety_var=_var(safety_min_profiles),

        final_dst_mean=_mean(final_dst_profiles),
        final_dst_min=_min(final_dst_profiles),
        final_dst_var=_var(final_dst_profiles),

        time_mean=_mean(time_profiles),
        time_var=_var(time_profiles),

        unc_u1=float(unc_u1),
        unc_u2=float(unc_u2),

        ep_len_mean=_mean([float(x) for x in ep_len_profiles]),
        ep_len_min=int(_min([float(x) for x in ep_len_profiles])) if rollout_succ > 0 else -1,
        ep_len_max=int(_max([float(x) for x in ep_len_profiles])) if rollout_succ > 0 else -1,

        steps_mean=_mean([float(x) for x in steps_profiles]),
        steps_min=int(_min([float(x) for x in steps_profiles])) if rollout_succ > 0 else -1,
        steps_max=int(_max([float(x) for x in steps_profiles])) if rollout_succ > 0 else -1,

        tf_mse=float(tf_mse),
    )
    return res


# ======================
# 8) main
# ======================
def main():
    import argparse
    global MC_RUNS, METHODS, OUT_CSV
    ap = argparse.ArgumentParser(description="Closed-loop MC evaluation of PoSafeNet (2D robot).")
    ap.add_argument("--methods", nargs="+", choices=list(CKPT_MAP), default=METHODS)
    ap.add_argument("--hard-ckpt", type=str, default=CKPT_MAP["PoSafeNet_Hard"])
    ap.add_argument("--mixture-ckpt", type=str, default=CKPT_MAP["PoSafeNet_Mixture"])
    ap.add_argument("--mc-runs", type=int, default=MC_RUNS)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()
    METHODS = list(args.methods)
    CKPT_MAP["PoSafeNet_Hard"] = args.hard_ckpt
    CKPT_MAP["PoSafeNet_Mixture"] = args.mixture_ckpt
    MC_RUNS = args.mc_runs
    OUT_CSV = args.out_csv or OUT_DIR / f"metrics_mc_runs{MC_RUNS}_noise{int(ENABLE_INTERNAL_NOISE)}.csv"
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for method in METHODS:
        res = run_benchmark_one_method(method)
        results.append(res)

        print(
            f"{res['method']:10s} "
            f"succ {res['succ']:3d}/{MC_RUNS:3d}  "
            f"fail {res['fail']:3d}  "
            f"qp_succ_calls {res['qp_succ_calls']:7d}/{res['qp_calls']:7d}  "
            f"qp_fail_calls {res['qp_fail_calls']:7d}  "
            f"safety(min/mean) {res['safety_min']:+.3f}/{res['safety_mean']:+.3f}  "
            f"final_dst(mean/min) {res['final_dst_mean']:.3g}/{res['final_dst_min']:.3g}  "
            f"tf_mse(qp-succ-only) {res['tf_mse']:.2e}  "
            f"time(mean±sd) {res['time_mean']:.3g}±{np.sqrt(res['time_var']):.2g}  "
            f"unc(u1,u2)=({res['unc_u1']:.3g},{res['unc_u2']:.3g})  "
            f"fail_preview={res['fails_preview']}"
        )

    # save CSV
    fieldnames = [
        "method",

        # rollout-level (aligned with stats)
        "succ", "fail",

        # qp-call diagnostics
        "qp_calls", "qp_succ_calls", "qp_fail_calls",
        "rollout_succ", "rollout_fail",
        "fails_preview",

        # ONLY ok-rollouts:
        "safety_min", "safety_mean", "safety_var",
        "final_dst_mean", "final_dst_min", "final_dst_var",
        "time_mean", "time_var",
        "unc_u1", "unc_u2",
        "ep_len_mean", "ep_len_min", "ep_len_max",
        "steps_mean", "steps_min", "steps_max",
        "tf_mse",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            # fails_preview is a list -> write as string to keep csv valid
            row = {k: r.get(k, "") for k in fieldnames}
            row["fails_preview"] = str(row.get("fails_preview", ""))
            w.writerow(row)

    print(f"[saved] MC metrics csv -> {OUT_CSV}")


if __name__ == "__main__":
    main()
