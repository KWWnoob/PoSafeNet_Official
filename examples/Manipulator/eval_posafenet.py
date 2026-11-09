#!/usr/bin/env python3
# eval_posafenet.py (Manipulator)
#
# Benchmark logic = like your provided script, but adapted to YOUR data format:
# - data/*.mat rows: [ep_id, t, th1, w1, th2, w2, dst_x, dst_y, u1, u2]
# - each MC rollout: randomly choose ONE episode (entire episode), sort by t
# - init = first row of that episode
# - for k=0..T_ep-1:
#     normalize current state + fixed dst (from init row)
#     ctrl = model(x_r, 0)   (physical control)
#     integrate with odeint(dynamics, dt=[0,0.1])
# - record safety (tip obstacle), tip locations
# - also build test1/test2 (swap u dims) and choose lower loss vs gt (within that episode)
# - Repeat MC_RUNS times (MC) if model has internal randomness (model.add_noise=True)
#
# Extras:
# - catch qpth warnings/prints and treat as failure
# - catch exceptions and nonfinite u and treat as failure
# - compute uncertainty allowing variable episode length using NaN padding + nanstd

'''
Multi-seed paired closed-loop evaluation of PoSafeNet-Hard / PoSafeNet-Mixture.

cd examples/Manipulator
python eval_posafenet.py \
  --train-seeds 0 1 2 3 4 5 \
  --ckpt-epoch 19 \
  --mc-runs 100 \
  --bench-seed 0
'''
from __future__ import annotations
import argparse
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
import torch.nn as nn
from torch.autograd import Variable
from scipy.integrate import odeint

import models


# =========================
# 0) Config
# =========================
_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = _THIS_DIR / "data_v2" / "data"
DT_PAIR = [0.0, 0.05]

# obstacle for safety metric (tip)
OBS_X, OBS_Y, OBS_R = 0.0, 7.0, 4.0
L1, L2 = 3.0, 3.0

# failure handling
TREAT_QPTH_WARNING_AS_FAIL = True
TREAT_NONFINITE_AS_FAIL = True
FAIL_VERBOSE = False

# defaults; command-line arguments in main() can override these
MC_RUNS = 100
ENABLE_INTERNAL_NOISE = True
TRAIN_SEEDS = (0, 1, 2, 3, 4, 5)
BENCH_SEED = 0
CKPT_EPOCH = 19

METHODS = [
    "PoSafeNet_Hard",
    "PoSafeNet_Mixture",
]

MODEL_PARAM = [6, 128, 256, 128, 128, 32, 32, 2]
POSA_HEADS = 2


# Checkpoint root = output folder of train_posafenet.py
CKPT_ROOT = _THIS_DIR / "log"


def make_ckpts(exp: int, ckpt_epoch: int) -> Dict[str, Path]:
    """Checkpoint paths for one independently trained seed/experiment."""
    return {
        "PoSafeNet_Hard": CKPT_ROOT / Path(
            f"posafenet_hard/posafenet{exp:02d}/model_posa{ckpt_epoch:02d}.pth"
        ),
        "PoSafeNet_Mixture": CKPT_ROOT / Path(
            f"posafenet_mixture/posafenet{exp:02d}/model_posa{ckpt_epoch:02d}.pth"
        ),
    }


# =========================
# 1) device / dtype
# =========================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

dtype = torch.float64
torch.backends.cudnn.benchmark = True


# =========================
# 2) dynamics (EXACT yours)
# =========================
def dynamics(y, t):
    # y = [theta1, w1, theta2, w2, u1, u2]
    dxdt = y[1]
    dydt = y[4]   # u1
    dttdt = y[3]
    dvdt = y[5]   # u2
    return [dxdt, dydt, dttdt, dvdt, 0, 0]


# =========================
# 3) load dataset (your data format)
# =========================
train_data = sio.loadmat(str(DATA_DIR / "dataM_train.mat"))["data"]
valid_data = sio.loadmat(str(DATA_DIR / "dataM_valid.mat"))["data"]
test_data  = sio.loadmat(str(DATA_DIR / "dataM_test.mat"))["data"]

# columns: [ep_id, t, th1, w1, th2, w2, dst_x, dst_y, u1, u2]
ep_ids = test_data[:, 0].astype(np.int64)
t_col  = test_data[:, 1].astype(np.float64)

unique_eps = np.unique(ep_ids)
print("num episodes in test:", len(unique_eps))
print("first 10 eps:", unique_eps[:10])

PLOT_EP_ID = int(unique_eps[0])  # or pick any existing ep id
PLOTS_DIR = _THIS_DIR / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# stats of episode lengths (optional but useful)
lens = np.array([np.sum(ep_ids == e) for e in unique_eps], dtype=np.int64)
print("ep_len min/mean/max:", int(lens.min()), float(lens.mean()), int(lens.max()))

def sample_episode_indices(rng: np.random.Generator) -> np.ndarray:
    """Sample one episode and return ALL its row indices sorted by t."""
    ep = rng.choice(unique_eps)
    idx = np.where(ep_ids == ep)[0]
    idx = idx[np.argsort(t_col[idx])]
    return idx

def get_episode_indices_by_id(ep: int) -> np.ndarray:
    """Return ALL indices for episode ep sorted by t."""
    idx = np.where(ep_ids == int(ep))[0]
    if idx.size == 0:
        raise ValueError(f"episode id {ep} not found in test_data")
    idx = idx[np.argsort(t_col[idx])]
    return idx

# training normalization uses the state+dst columns (th1,w1,th2,w2,dst_x,dst_y)
train0 = np.double(train_data[:, 2:8])  # (N,6)
train_labels_unnorm = np.reshape(np.double(train_data[:, 8:10]), (len(train_data), 2))  # (N,2)

valid0 = np.double(valid_data[:, 2:8])
valid_labels_unnorm = np.reshape(np.double(valid_data[:, 8:10]), (len(valid_data), 2))

test0_all = np.double(test_data[:, 2:8])
test_labels_unnorm_all = np.reshape(np.double(test_data[:, 8:10]), (len(test_data), 2))

mean = np.mean(train0, axis=0)
std = np.std(train0, axis=0)

mean_label = np.mean(train_labels_unnorm, axis=0)
std_label = np.std(train_labels_unnorm, axis=0)

loss_fn = nn.MSELoss()


# =========================
# 4) helpers
# =========================
def normalize_state(theta1, w1, theta2, w2, dstx, dsty):
    t1 = (theta1 - mean[0]) / std[0]
    o1 = (w1     - mean[1]) / std[1]
    t2 = (theta2 - mean[2]) / std[2]
    o2 = (w2     - mean[3]) / std[3]
    dx = (dstx   - mean[4]) / std[4]
    dy = (dsty   - mean[5]) / std[5]
    return np.array([t1, o1, t2, o2, dx, dy], dtype=np.float64)

def tip_safety(theta1, theta2):
    # safe = ||tip - obs||^2 - R^2
    tx = L1 * np.cos(theta1) + L2 * np.cos(theta2)
    ty = L1 * np.sin(theta1) + L2 * np.sin(theta2)
    return (tx - OBS_X) ** 2 + (ty - OBS_Y) ** 2 - OBS_R ** 2, (tx, ty)

PHI_MIN = -np.pi * 2.2 / 3.0
PHI_MAX =  np.pi * 2.2 / 3.0

def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def phi_violation(theta1: float, theta2: float) -> Tuple[float, float, float]:
    """
    returns:
      phi (wrapped),
      vio_min = max(0, phi_min - phi),
      vio_max = max(0, phi - phi_max)
    """
    phi = wrap_pi(theta2 - theta1)
    vio_min = max(0.0, PHI_MIN - phi)
    vio_max = max(0.0, phi - PHI_MAX)
    return phi, vio_min, vio_max

def tip_to_dst(theta1: float, theta2: float, dstx: float, dsty: float) -> float:
    tx = L1 * np.cos(theta1) + L2 * np.cos(theta2)
    ty = L1 * np.sin(theta1) + L2 * np.sin(theta2)
    return float(np.sqrt((tx - dstx)**2 + (ty - dsty)**2))


# =========================
# 5) qpth warning capture
# =========================
_QP_PRINT_SUBSTRS = [
    "qpth warning",
    "returning an inaccurate and potentially incorrect solution",
    "some residual is large",
    "your problem may be infeasible or difficult",
    "iterative refinement",
    "infeasible",
]

def safe_model_call(model, x_r: torch.Tensor) -> Tuple[Optional[torch.Tensor], bool, str]:
    """
    Calls model(x_r, 0) but catches:
      - exceptions
      - qpth warnings (printed or warnings.warn)
      - nonfinite outputs
    Returns:
      (ctrl, ok, reason)
    """
    try:
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        qp_warned_by_print = False
        qp_warned_by_warn = False

        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ctrl = model(x_r, 0)

                for wi in w:
                    msg = str(wi.message).lower()
                    if any(s in msg for s in _QP_PRINT_SUBSTRS):
                        qp_warned_by_warn = True
                        break

        printed = (buf_out.getvalue() + "\n" + buf_err.getvalue()).lower()
        if any(s in printed for s in _QP_PRINT_SUBSTRS):
            qp_warned_by_print = True

        if TREAT_QPTH_WARNING_AS_FAIL and (qp_warned_by_print or qp_warned_by_warn):
            return None, False, "qpth_warning_print" if qp_warned_by_print else "qpth_warning_warn"

        if ctrl is None:
            return None, False, "ctrl_none"

        ctrl_np = ctrl.detach().cpu().numpy()
        if TREAT_NONFINITE_AS_FAIL and (not np.all(np.isfinite(ctrl_np))):
            return None, False, "nonfinite_ctrl"

        return ctrl, True, "ok"

    except Exception as e:
        return None, False, f"exception:{type(e).__name__}"


def sync_device() -> None:
    """Synchronize CUDA so wall-clock timing is comparable."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# =========================
# 6) unified single-rollout test (whole episode)
# =========================
def rollout_once(model, idx_ep: np.ndarray) -> Dict[str, Any]:
    """
    One rollout over a preselected whole episode.

    The caller supplies idx_ep so every method and every training seed is
    evaluated on the same episode schedule.
    """
    ep_id = int(ep_ids[idx_ep[0]])

    # episode data
    test0_ep = np.double(test_data[idx_ep, 2:8])          # (Tep, 6)
    gt_ep    = np.reshape(np.double(test_data[idx_ep, 8:10]), (-1, 2))  # (Tep,2)

    # init = first row of that episode
    theta1, w1, theta2, w2, dstx, dsty = map(float, test0_ep[0])

    safety_list: List[float] = []
    loc_xy_list: List[torch.Tensor] = []
    test1: List[torch.Tensor] = []
    test2: List[torch.Tensor] = []
    phi_list: List[float] = []
    phi_vio_min_list: List[float] = []
    phi_vio_max_list: List[float] = []

    sync_device()
    t0 = time.perf_counter()

    with torch.no_grad():
        for k in range(len(test0_ep)):
            # normalize current state + fixed dst (dst from init)
            z = normalize_state(theta1, w1, theta2, w2, dstx, dsty)
            x_r = Variable(torch.from_numpy(z), requires_grad=False)
            x_r = torch.reshape(x_r, (1, MODEL_PARAM[0])).to(device=device)

            # safety metric
            safe, (tx, ty) = tip_safety(theta1, theta2)
            safety_list.append(float(safe))
            loc_xy_list.append(torch.tensor([[tx, ty]], dtype=torch.float64))
            phi, vmin, vmax = phi_violation(theta1, theta2)
            phi_list.append(float(phi))
            phi_vio_min_list.append(float(vmin))
            phi_vio_max_list.append(float(vmax))

            # control
            ctrl, ok, reason = safe_model_call(model, x_r)
            if not ok:
                if FAIL_VERBOSE:
                    print(f"[FAIL] ep_id={ep_id} step={k} reason={reason}")
                return dict(ok=False, fail_reason=reason, fail_step=k, ep_id=ep_id, ep_len=len(test0_ep))

            # swap trick for matching gt dim ordering
            test1.append(torch.tensor([[ctrl[0, 0].item(), ctrl[0, 1].item()]], dtype=torch.float64))
            test2.append(torch.tensor([[ctrl[0, 1].item(), ctrl[0, 0].item()]], dtype=torch.float64))

            # integrate dynamics (EXACT yours)
            state = [theta1, w1, theta2, w2, ctrl[0, 0].item(), ctrl[0, 1].item()]
            rt = np.float32(odeint(dynamics, state, DT_PAIR))
            theta1, w1, theta2, w2 = float(rt[1][0]), float(rt[1][1]), float(rt[1][2]), float(rt[1][3])

    # stack
    test1_t = torch.cat(test1, dim=0)
    test2_t = torch.cat(test2, dim=0)
    gt = torch.from_numpy(gt_ep).to(dtype=torch.float64)

    # loss pick (EXACT yours)
    loss1 = loss_fn(test1_t, gt).cpu().item()
    loss2 = loss_fn(test2_t, gt).cpu().item()
    if loss1 <= loss2:
        test = test1_t
        loss = float(loss1)
        swap = False
    else:
        test = test2_t
        loss = float(loss2)
        swap = True

    loc_xy = torch.cat(loc_xy_list, dim=0)
    sync_device()
    t1 = time.perf_counter()

    final_tip_dst = tip_to_dst(theta1, theta2, dstx, dsty)

    return dict(
        ok=True,
        loss=loss,
        swap=swap,
        safety=np.array(safety_list, dtype=np.float64),
        safety_min=float(np.min(np.array(safety_list, dtype=np.float64))),
        ctrl=test,          # (T,2)
        loc_xy=loc_xy,      # (T,2)
        runtime=float(t1 - t0),
        ep_len=len(test0_ep),
        ep_id=ep_id,
        phi=np.array(phi_list, dtype=np.float64),
        phi_vio_min=np.array(phi_vio_min_list, dtype=np.float64),
        phi_vio_max=np.array(phi_vio_max_list, dtype=np.float64),

        phi_vio_max_step=float(np.max(np.array(phi_vio_max_list, dtype=np.float64))),
        phi_vio_min_step=float(np.max(np.array(phi_vio_min_list, dtype=np.float64))),
        # overall worst violation (either side)
        phi_vio_worst=float(max(
            np.max(np.array(phi_vio_min_list, dtype=np.float64)),
            np.max(np.array(phi_vio_max_list, dtype=np.float64))
        )),

        final_tip_dst=final_tip_dst,
    )

def rollout_episode(model, idx_ep: np.ndarray) -> Dict[str, Any]:
    ep_id = int(ep_ids[idx_ep[0]])

    test0_ep = np.double(test_data[idx_ep, 2:8])               # (T,6)
    gt_u_ep  = np.reshape(np.double(test_data[idx_ep, 8:10]), (-1, 2))  # (T,2)

    # init from first row
    theta1, w1, theta2, w2, dstx, dsty = map(float, test0_ep[0])

    safety_list: List[float] = []
    tip_xy_list: List[np.ndarray] = []
    ctrl_list: List[np.ndarray] = []          # chosen ctrl (after swap choice)
    ctrl_raw_list: List[np.ndarray] = []      # raw model output (before swap)
    swap_used = None

    test1: List[torch.Tensor] = []
    test2: List[torch.Tensor] = []

    t0 = time.time()
    with torch.no_grad():
        for k in range(len(test0_ep)):
            z = normalize_state(theta1, w1, theta2, w2, dstx, dsty)
            x_r = torch.from_numpy(z).to(device=device, dtype=torch.float64).view(1, MODEL_PARAM[0])

            safe, (tx, ty) = tip_safety(theta1, theta2)
            safety_list.append(float(safe))
            tip_xy_list.append(np.array([tx, ty], dtype=np.float64))

            ctrl, ok, reason = safe_model_call(model, x_r)
            if not ok:
                return dict(ok=False, fail_reason=reason, fail_step=k, ep_id=ep_id, ep_len=len(test0_ep))

            # record raw (u1,u2)
            u_raw = np.array([ctrl[0, 0].item(), ctrl[0, 1].item()], dtype=np.float64)
            ctrl_raw_list.append(u_raw)

            # swap trick candidates (for MSE evaluation)
            test1.append(torch.tensor([[u_raw[0], u_raw[1]]], dtype=torch.float64))
            test2.append(torch.tensor([[u_raw[1], u_raw[0]]], dtype=torch.float64))

            # integrate
            state = [theta1, w1, theta2, w2, u_raw[0], u_raw[1]]
            rt = np.float32(odeint(dynamics, state, DT_PAIR))
            theta1, w1, theta2, w2 = float(rt[1][0]), float(rt[1][1]), float(rt[1][2]), float(rt[1][3])

    # choose swap by loss vs GT u in this episode
    test1_t = torch.cat(test1, dim=0)
    test2_t = torch.cat(test2, dim=0)
    gt = torch.from_numpy(gt_u_ep).to(dtype=torch.float64)

    loss1 = loss_fn(test1_t, gt).cpu().item()
    loss2 = loss_fn(test2_t, gt).cpu().item()
    swap_used = bool(loss2 < loss1)

    ctrl_arr = np.stack(ctrl_raw_list, axis=0)  # (T,2)
    if swap_used:
        ctrl_arr = ctrl_arr[:, [1, 0]]

    t1 = time.time()
    return dict(
        ok=True,
        ep_id=ep_id,
        ep_len=len(test0_ep),
        safety=np.array(safety_list, dtype=np.float64),
        safety_min=float(np.min(np.array(safety_list, dtype=np.float64))),
        tip_xy=np.stack(tip_xy_list, axis=0),     # (T,2)
        ctrl=ctrl_arr,                            # (T,2) chosen after swap
        ctrl_raw=np.stack(ctrl_raw_list, axis=0), # (T,2)
        swap=swap_used,
        loss=float(min(loss1, loss2)),
        runtime=float(t1 - t0),
        dst=(float(dstx), float(dsty)),
    )

def expert_tip_xy_from_data(idx_ep: np.ndarray) -> Dict[str, Any]:
    ep_id = int(ep_ids[idx_ep[0]])
    test0_ep = np.double(test_data[idx_ep, 2:8])  # [th1,w1,th2,w2,dstx,dsty]
    th1 = test0_ep[:, 0]
    th2 = test0_ep[:, 2]
    tx = L1 * np.cos(th1) + L2 * np.cos(th2)
    ty = L1 * np.sin(th1) + L2 * np.sin(th2)
    safe = (tx - OBS_X) ** 2 + (ty - OBS_Y) ** 2 - OBS_R ** 2
    return dict(
        ep_id=ep_id,
        tip_xy=np.stack([tx, ty], axis=1),
        safety=safe.astype(np.float64),
        safety_min=float(np.min(safe)),
    )


# =========================
# 7) build models
# =========================
def build_model(name: str, ckpts: Dict[str, Path]):
    common = (
        MODEL_PARAM, mean, std, mean_label, std_label, device
    )

    if name == "PoSafeNet_Hard":
        model = models.PoSafeNet_Hard(
            *common, bn=False, heads=POSA_HEADS
        )

    elif name == "PoSafeNet_Mixture":
        model = models.PoSafeNet_Mixture(
            *common, bn=False, heads=POSA_HEADS
        )

    else:
        raise ValueError(f"Unknown method: {name}")

    return model.to(device), ckpts[name]


# =========================
# 8) benchmark runner
# =========================
def set_rollout_seed(seed: int) -> None:
    """Reset stochastic execution noise before one paired rollout."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def violation_stats(arrays: List[np.ndarray]) -> Dict[str, float]:
    """
    Summarize nonnegative violation magnitudes.

    Each array corresponds to one successful episode and has one entry per
    timestep. A value > 0 means that constraint is violated.
    """
    if not arrays:
        return {
            "episode_rate": float("nan"),
            "step_rate": float("nan"),
            "mean_magnitude": float("nan"),
            "mean_positive_magnitude": float("nan"),
            "max_magnitude": float("nan"),
        }

    clean = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
    flat = np.concatenate(clean)
    positive = flat[flat > 0.0]

    return {
        "episode_rate": float(np.mean([np.any(a > 0.0) for a in clean])),
        "step_rate": float(np.mean(flat > 0.0)),
        # Averaged over all timesteps; zero at nonviolating steps.
        "mean_magnitude": float(np.mean(flat)),
        # Severity conditional on an actual violation.
        "mean_positive_magnitude": (
            float(np.mean(positive)) if positive.size else 0.0
        ),
        "max_magnitude": float(np.max(flat)),
    }


def run_benchmark_one_method(
    name: str,
    train_seed: int,
    ckpts: Dict[str, Path],
    eval_ep_ids: np.ndarray,
    bench_seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model, ckpt = build_model(name, ckpts)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    num_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    if hasattr(model, "add_noise"):
        model.add_noise = bool(ENABLE_INTERNAL_NOISE)

    loss_profiles: List[float] = []
    safety_min_profiles: List[float] = []
    time_profiles: List[float] = []
    time_per_step_profiles: List[float] = []
    ctrl_profiles: List[torch.Tensor] = []
    ep_len_profiles: List[int] = []
    phi_vio_profiles: List[float] = []
    final_dst_profiles: List[float] = []

    obstacle_vio_arrays: List[np.ndarray] = []
    phi_lower_vio_arrays: List[np.ndarray] = []
    phi_upper_vio_arrays: List[np.ndarray] = []
    phi_any_vio_arrays: List[np.ndarray] = []

    fail_count = 0
    fails: List[Tuple[str, int, int]] = []
    episode_rows: List[Dict[str, Any]] = []

    for eval_run, ep_id in enumerate(eval_ep_ids):
        idx_ep = get_episode_indices_by_id(int(ep_id))

        # Paired evaluation: same episode and same stochastic seed for every
        # method and every independently trained checkpoint.
        rollout_seed = int(bench_seed * 1_000_000 + eval_run)
        set_rollout_seed(rollout_seed)

        out = rollout_once(model, idx_ep)

        base_row = {
            "train_seed": train_seed,
            "method": name,
            "eval_run": eval_run,
            "eval_seed": rollout_seed,
            "episode_id": int(ep_id),
        }

        if not out["ok"]:
            fail_count += 1
            reason = out.get("fail_reason", "unknown")
            fail_step = int(out.get("fail_step", -1))
            fails.append((reason, fail_step, int(ep_id)))
            episode_rows.append({
                **base_row,
                "ok": 0,
                "fail_reason": reason,
                "fail_step": fail_step,
                "ep_len": int(out.get("ep_len", -1)),
            })
            continue

        safety = np.asarray(out["safety"], dtype=np.float64)
        obstacle_vio = np.maximum(0.0, -safety)
        phi_lower_vio = np.asarray(out["phi_vio_min"], dtype=np.float64)
        phi_upper_vio = np.asarray(out["phi_vio_max"], dtype=np.float64)
        phi_any_vio = np.maximum(phi_lower_vio, phi_upper_vio)

        loss_profiles.append(float(out["loss"]))
        safety_min_profiles.append(float(out["safety_min"]))
        time_profiles.append(float(out["runtime"]))
        time_per_step_profiles.append(
            float(out["runtime"]) / max(int(out["ep_len"]), 1)
        )
        ctrl_profiles.append(out["ctrl"].unsqueeze(0))
        ep_len_profiles.append(int(out["ep_len"]))
        phi_vio_profiles.append(float(out["phi_vio_worst"]))
        final_dst_profiles.append(float(out["final_tip_dst"]))

        obstacle_vio_arrays.append(obstacle_vio)
        phi_lower_vio_arrays.append(phi_lower_vio)
        phi_upper_vio_arrays.append(phi_upper_vio)
        phi_any_vio_arrays.append(phi_any_vio)

        episode_rows.append({
            **base_row,
            "ok": 1,
            "fail_reason": "",
            "fail_step": -1,
            "ep_len": int(out["ep_len"]),
            "loss": float(out["loss"]),
            "runtime": float(out["runtime"]),
            "time_per_step": (
                float(out["runtime"]) / max(int(out["ep_len"]), 1)
            ),
            "safety_min": float(out["safety_min"]),
            "final_dst": float(out["final_tip_dst"]),
            "obstacle_violation_any": int(np.any(obstacle_vio > 0.0)),
            "obstacle_violation_step_rate": float(np.mean(obstacle_vio > 0.0)),
            "obstacle_violation_mean": float(np.mean(obstacle_vio)),
            "obstacle_violation_max": float(np.max(obstacle_vio)),
            "phi_lower_violation_any": int(np.any(phi_lower_vio > 0.0)),
            "phi_lower_violation_step_rate": float(np.mean(phi_lower_vio > 0.0)),
            "phi_lower_violation_mean": float(np.mean(phi_lower_vio)),
            "phi_lower_violation_max": float(np.max(phi_lower_vio)),
            "phi_upper_violation_any": int(np.any(phi_upper_vio > 0.0)),
            "phi_upper_violation_step_rate": float(np.mean(phi_upper_vio > 0.0)),
            "phi_upper_violation_mean": float(np.mean(phi_upper_vio)),
            "phi_upper_violation_max": float(np.max(phi_upper_vio)),
            "phi_any_violation_any": int(np.any(phi_any_vio > 0.0)),
            "phi_any_violation_step_rate": float(np.mean(phi_any_vio > 0.0)),
            "phi_any_violation_mean": float(np.mean(phi_any_vio)),
            "phi_any_violation_max": float(np.max(phi_any_vio)),
        })

    succ = len(loss_profiles)

    # This is rollout-level control dispersion, not the requested uncertainty
    # across independently trained seeds. Across-seed uncertainty is produced
    # later from the seed-level summaries.
    if succ > 1:
        lengths = [c.shape[1] for c in ctrl_profiles]
        max_t = int(max(lengths))
        controls = np.full((succ, max_t, 2), np.nan, dtype=np.float64)
        for idx, ctrl in enumerate(ctrl_profiles):
            ui = ctrl.squeeze(0).cpu().numpy()
            controls[idx, :ui.shape[0], :] = ui
        std_t = np.nanstd(controls, axis=0)
        dispersion = np.nanmean(std_t, axis=0)
        control_dispersion_u1 = float(dispersion[0])
        control_dispersion_u2 = float(dispersion[1])
    else:
        control_dispersion_u1 = float("nan")
        control_dispersion_u2 = float("nan")

    obstacle_stats = violation_stats(obstacle_vio_arrays)
    phi_lower_stats = violation_stats(phi_lower_vio_arrays)
    phi_upper_stats = violation_stats(phi_upper_vio_arrays)
    phi_any_stats = violation_stats(phi_any_vio_arrays)

    def mean_or_nan(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def var_or_nan(values: List[float]) -> float:
        return float(np.var(values)) if values else float("nan")

    res = {
        "train_seed": train_seed,
        "method": name,
        "checkpoint": str(ckpt),
        "num_params": num_params,
        "trainable_params": trainable_params,
        "requested_runs": len(eval_ep_ids),
        "succ": succ,
        "fail": fail_count,
        "success_rate": succ / len(eval_ep_ids) if len(eval_ep_ids) else float("nan"),
        "fails_preview": repr(fails[:5]),
        "time_mean": mean_or_nan(time_profiles),
        "time_var": var_or_nan(time_profiles),
        "time_per_step_mean": mean_or_nan(time_per_step_profiles),
        "time_per_step_var": var_or_nan(time_per_step_profiles),
        "loss_mean": mean_or_nan(loss_profiles),
        "loss_var": var_or_nan(loss_profiles),
        "safety_min": (
            float(np.min(safety_min_profiles))
            if safety_min_profiles else float("nan")
        ),
        "safety_mean": mean_or_nan(safety_min_profiles),
        "safety_var": var_or_nan(safety_min_profiles),
        "control_dispersion_u1": control_dispersion_u1,
        "control_dispersion_u2": control_dispersion_u2,
        "ep_len_mean": mean_or_nan(ep_len_profiles),
        "ep_len_min": (
            int(np.min(ep_len_profiles)) if ep_len_profiles else -1
        ),
        "ep_len_max": (
            int(np.max(ep_len_profiles)) if ep_len_profiles else -1
        ),
        "phi_vio_mean": mean_or_nan(phi_vio_profiles),
        "phi_vio_max": (
            float(np.max(phi_vio_profiles))
            if phi_vio_profiles else float("nan")
        ),
        "phi_vio_var": var_or_nan(phi_vio_profiles),
        "final_dst_mean": mean_or_nan(final_dst_profiles),
        "final_dst_min": (
            float(np.min(final_dst_profiles))
            if final_dst_profiles else float("nan")
        ),
        "final_dst_var": var_or_nan(final_dst_profiles),
    }

    for prefix, stats in (
        ("obstacle", obstacle_stats),
        ("phi_lower", phi_lower_stats),
        ("phi_upper", phi_upper_stats),
        ("phi_any", phi_any_stats),
    ):
        for key, value in stats.items():
            res[f"{prefix}_violation_{key}"] = value

    return res, episode_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_across_seed_summary(
    seed_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Long-format mean/std across independently trained seeds.

    The standard deviation uses ddof=1 and therefore represents uncertainty
    across training seeds, not uncertainty across evaluation episodes.
    """
    excluded = {
        "train_seed", "method", "checkpoint", "fails_preview",
        "requested_runs",
    }
    methods = sorted({row["method"] for row in seed_rows})
    summary: List[Dict[str, Any]] = []

    for method in methods:
        method_rows = [r for r in seed_rows if r["method"] == method]
        numeric_keys = sorted({
            key
            for row in method_rows
            for key, value in row.items()
            if key not in excluded
            and isinstance(value, (int, float, np.integer, np.floating))
        })

        for metric in numeric_keys:
            values = np.asarray(
                [row.get(metric, np.nan) for row in method_rows],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            summary.append({
                "method": method,
                "metric": metric,
                "n_train_seeds": int(values.size),
                "mean": float(np.mean(values)),
                "std": (
                    float(np.std(values, ddof=1))
                    if values.size > 1 else float("nan")
                ),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })

    return summary


#--------------------------
# 9) plotting helper
#--------------------------
import matplotlib.pyplot as plt

def plot_all_tip_xy(ep_id: int, expert: Dict[str, Any], rollouts: Dict[str, Dict[str, Any]]):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()

    # obstacle circle
    ang = np.linspace(0, 2*np.pi, 400)
    cx = OBS_X + OBS_R * np.cos(ang)
    cy = OBS_Y + OBS_R * np.sin(ang)
    ax.plot(cx, cy, linewidth=2)

    # expert
    exy = expert["tip_xy"]
    ax.plot(exy[:, 0], exy[:, 1], linewidth=3, label=f"EXPERT (min_safe={expert['safety_min']:+.3f})")

    # models
    for name, out in rollouts.items():
        if not out.get("ok", False):
            ax.plot([], [], label=f"{name} (FAIL:{out.get('fail_reason','?')})")
            continue
        xy = out["tip_xy"]
        ax.plot(xy[:, 0], xy[:, 1], linewidth=2, label=f"{name} (min_safe={out['safety_min']:+.3f}, swap={out['swap']})")

        # mark start/end
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=30)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=30)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Tip XY trajectories (no-noise) | ep_id={ep_id}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True)

    out_png = PLOTS_DIR / f"ep{ep_id}_tip_xy_all.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return out_png

def save_rollouts_npz(ep_id: int, expert: Dict[str, Any], rollouts: Dict[str, Dict[str, Any]]):
    out_npz = PLOTS_DIR / f"ep{ep_id}_rollouts_no_noise.npz"
    pack = {
        "ep_id": np.array([ep_id], dtype=np.int64),
        "expert_tip_xy": expert["tip_xy"],
        "expert_safety": expert["safety"],
    }
    for name, out in rollouts.items():
        if out.get("ok", False):
            pack[f"{name}_tip_xy"] = out["tip_xy"]
            pack[f"{name}_safety"] = out["safety"]
            pack[f"{name}_ctrl"] = out["ctrl"]
            pack[f"{name}_ctrl_raw"] = out["ctrl_raw"]
            pack[f"{name}_swap"] = np.array([int(out["swap"])], dtype=np.int64)
        else:
            pack[f"{name}_fail"] = np.array([1], dtype=np.int64)
    np.savez_compressed(out_npz, **pack)
    return out_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PoSafeNet configurations over multiple independent "
            "training seeds using a paired episode/noise schedule."
        )
    )
    parser.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=list(TRAIN_SEEDS),
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=CKPT_ROOT,
        help="checkpoint root (default: log/, as written by train_posafenet.py)",
    )
    parser.add_argument("--ckpt-epoch", type=int, default=CKPT_EPOCH)
    parser.add_argument("--mc-runs", type=int, default=MC_RUNS)
    parser.add_argument("--bench-seed", type=int, default=BENCH_SEED)
    parser.add_argument(
        "--no-internal-noise",
        action="store_true",
        help="Disable model.add_noise during benchmark rollouts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_THIS_DIR / "results" / "multiseed_benchmark",
    )
    return parser.parse_args()


def main():
    global ENABLE_INTERNAL_NOISE, CKPT_ROOT

    args = parse_args()
    CKPT_ROOT = args.ckpt_root
    if args.mc_runs <= 0:
        raise ValueError("--mc-runs must be positive")

    ENABLE_INTERNAL_NOISE = not args.no_internal_noise
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # One fixed episode schedule is reused for every method and training seed.
    schedule_rng = np.random.default_rng(args.bench_seed)
    replace = args.mc_runs > len(unique_eps)
    eval_ep_ids = schedule_rng.choice(
        unique_eps,
        size=args.mc_runs,
        replace=replace,
    ).astype(np.int64)

    schedule_rows = [
        {
            "eval_run": i,
            "episode_id": int(ep_id),
            "eval_seed": int(args.bench_seed * 1_000_000 + i),
        }
        for i, ep_id in enumerate(eval_ep_ids)
    ]
    write_csv(args.output_dir / "evaluation_schedule.csv", schedule_rows)

    print("============================================================")
    print("Multi-seed paired benchmark")
    print(f"Training seeds: {args.train_seeds}")
    print(f"Checkpoint epoch: {args.ckpt_epoch}")
    print(f"Evaluation runs per checkpoint: {args.mc_runs}")
    print(f"Benchmark seed: {args.bench_seed}")
    print(f"Internal noise: {ENABLE_INTERNAL_NOISE}")
    print(f"Unique test episodes available: {len(unique_eps)}")
    print(f"Episode sampling with replacement: {replace}")
    print(f"Output directory: {args.output_dir}")
    print("============================================================")

    seed_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    for train_seed in args.train_seeds:
        ckpts = make_ckpts(train_seed, args.ckpt_epoch)

        print("")
        print("============================================================")
        print(f"Training seed {train_seed}")
        print("============================================================")

        for name in METHODS:
            ckpt = ckpts[name]
            if not ckpt.is_file():
                print(f"[skip] seed={train_seed} {name}: {ckpt}")
                missing_rows.append({
                    "train_seed": train_seed,
                    "method": name,
                    "checkpoint": str(ckpt),
                })
                continue

            print(f"[run]  seed={train_seed} {name}: {ckpt}")
            res, rows = run_benchmark_one_method(
                name=name,
                train_seed=train_seed,
                ckpts=ckpts,
                eval_ep_ids=eval_ep_ids,
                bench_seed=args.bench_seed,
            )
            seed_rows.append(res)
            episode_rows.extend(rows)

            print(
                f"[done] seed={train_seed} {name} "
                f"succ={res['succ']}/{res['requested_runs']} "
                f"obstacle_ep_vio={res['obstacle_violation_episode_rate']:.3%} "
                f"phi_lower_ep_vio={res['phi_lower_violation_episode_rate']:.3%} "
                f"phi_upper_ep_vio={res['phi_upper_violation_episode_rate']:.3%} "
                f"final_dst={res['final_dst_mean']:.4f}"
            )

    if not seed_rows:
        raise FileNotFoundError(
            "No checkpoints were evaluated. Check checkpoint paths, "
            "--train-seeds, and --ckpt-epoch."
        )

    seed_path = args.output_dir / "seed_level_summary.csv"
    episode_path = args.output_dir / "episode_level_results.csv"
    across_path = args.output_dir / "across_seed_summary_long.csv"
    missing_path = args.output_dir / "missing_checkpoints.csv"

    write_csv(seed_path, seed_rows)
    write_csv(episode_path, episode_rows)
    write_csv(across_path, make_across_seed_summary(seed_rows))
    write_csv(missing_path, missing_rows)

    print("")
    print("============================================================")
    print("Benchmark finished")
    print(f"Seed-level summary:   {seed_path}")
    print(f"Episode-level data:   {episode_path}")
    print(f"Across-seed summary:  {across_path}")
    if missing_rows:
        print(f"Missing checkpoints:  {missing_path}")
    print("============================================================")


if __name__ == "__main__":
    main()