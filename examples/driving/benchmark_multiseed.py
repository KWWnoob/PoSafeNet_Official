#!/usr/bin/env python3
"""
benchmark_multiseed.py  (aggregates eval.py result pickles over seeds)

Reads VISTA evaluation results pickles saved by your eval script:
  all_results = [episode_results] * n_episodes
  episode_results = [step_results] * T
  step_results = {
      'ego_agent_id': str,
      'logs': {agent_id: {...}},
      'actions': ...,
      'infos': {agent_id: {...}},
      'barrier': float,   # <-- saved by your eval script
  }

Outputs:
  - rollout_summary_<stem>.csv  (one row per episode/rollout)
  - benchmark_summary.csv       (one row per pkl)

Rollout-level success metrics:
  - success: no crash, no out_of_lane, and overlap_max==0 (rot is reported but NOT in success)
  - pass: no crash

Continuous lane metrics (from d):
  Define cross-track error e_y(t) = d(t)
  - ey_abs_mean: mean_t |e_y(t)|
  - ey_var: var_t e_y(t)   (signed variance)
  - time_out_of_lane: fraction of time steps with |e_y(t)| > road_width/2
  - lane_margin_min/mean: lane margin = (road_width/2) - |d| (same as before)

MODEL barrier metrics (from step_results['barrier']):
  - barrier_min/mean: aggregated over time within rollout

Action uncertainty:
  - u1_std/u2_std: std over time within rollout (ego action first 2 dims)
  - u1_abs_mean/u2_abs_mean: mean abs over time within rollout
"""

import argparse, glob, os, pickle, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ---------- Defaults ----------
DEFAULT_ROAD_WIDTH = 4.0


def load_pkl(p: str) -> Any:
    with open(p, "rb") as f:
        return pickle.load(f)


def as_array(x):
    if x is None:
        return np.array([])
    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    if isinstance(x, (list, tuple)):
        return np.asarray(x).reshape(-1)
    return np.asarray([x]).reshape(-1)


def find_step_dict(step_like: Any) -> Dict[str, Any]:
    # step could be dict, or list containing dict
    if isinstance(step_like, dict):
        return step_like
    if isinstance(step_like, (list, tuple)):
        for it in step_like:
            if isinstance(it, dict):
                return it
    raise TypeError(f"Cannot find step dict, got {type(step_like)}")


def pick_ego_id_from_step(step: Dict[str, Any]) -> str:
    ego_id = step.get("ego_agent_id", None)
    if isinstance(ego_id, str) and ego_id:
        return ego_id

    logs = step.get("logs", {})
    if isinstance(logs, dict) and len(logs) > 0:
        return next(iter(logs.keys()))

    infos = step.get("infos", {})
    if isinstance(infos, dict) and len(infos) > 0:
        return next(iter(infos.keys()))

    return "unknown"


def get_agent_ids_from_step(step: Dict[str, Any]) -> List[str]:
    logs = step.get("logs", {})
    if isinstance(logs, dict) and len(logs) > 0:
        return list(logs.keys())
    infos = step.get("infos", {})
    if isinstance(infos, dict) and len(infos) > 0:
        return list(infos.keys())
    return []


def get_step_info(step: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    infos = step.get("infos", {})
    if isinstance(infos, dict):
        v = infos.get(agent_id, {})
        return v if isinstance(v, dict) else {}
    return {}


def get_step_log(step: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    logs = step.get("logs", {})
    if isinstance(logs, dict):
        v = logs.get(agent_id, {})
        return v if isinstance(v, dict) else {}
    return {}


def _get_float_from_log_or_step(key: str, log: Dict[str, Any], step: Dict[str, Any]) -> Optional[float]:
    if key in log:
        try:
            return float(log[key])
        except Exception:
            return None
    if key in step:
        try:
            return float(step[key])
        except Exception:
            return None
    return None


def _get_xy_from_log_or_step(log: Dict[str, Any], step: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    x = _get_float_from_log_or_step("x", log, step)
    y = _get_float_from_log_or_step("y", log, step)
    return x, y


# --------- actions parsing + u1/u2 extraction ---------
def _get_action_vec(step: Dict[str, Any], agent_id: str) -> Optional[np.ndarray]:
    """
    Try to get action vector for agent at this step.
    Supports common formats:
      - actions is dict: {agent_id: np.array([..])}
      - actions is list/ndarray: [..] (single-agent)
      - nested dict: {agent_id: {'action': [..]}} etc (best-effort)
    """
    if "actions" not in step:
        return None
    a = step.get("actions", None)
    if a is None:
        return None

    # dict format
    if isinstance(a, dict):
        v = a.get(agent_id, None)
        if v is None:
            if len(a) == 1:
                v = next(iter(a.values()))
            else:
                return None

        if isinstance(v, dict):
            for kk in ["action", "u", "ctrl", "control", "act"]:
                if kk in v:
                    v = v[kk]
                    break

        arr = as_array(v)
        return arr if arr.size else None

    # array/list format
    arr = as_array(a)
    return arr if arr.size else None


def _get_u1_u2_from_step(step: Dict[str, Any], ego_id: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract u1,u2 for ego at this step.

    Priority:
      1) step['actions'] (vector, take first 2 dims)
      2) log keys: control/a & control/omega (or omega/a), also try control/delta & control/v
      3) step top-level keys: a/omega or omega/a
    """
    # 1) from actions
    avec = _get_action_vec(step, ego_id)
    if avec is not None and avec.size >= 2:
        try:
            return float(avec[0]), float(avec[1])
        except Exception:
            pass

    # 2) from logs
    lg = get_step_log(step, ego_id)
    if isinstance(lg, dict) and len(lg) > 0:
        a = lg.get("control/a", None)
        om = lg.get("control/omega", None)
        if a is not None and om is not None:
            try:
                return float(a), float(om)
            except Exception:
                pass

        a = lg.get("a", None)
        om = lg.get("omega", None)
        if a is not None and om is not None:
            try:
                return float(a), float(om)
            except Exception:
                pass

        v = lg.get("control/v", None)
        de = lg.get("control/delta", None)
        if v is not None and de is not None:
            try:
                return float(v), float(de)
            except Exception:
                pass

    # 3) from step top-level
    a = step.get("a", None)
    om = step.get("omega", None)
    if a is not None and om is not None:
        try:
            return float(a), float(om)
        except Exception:
            pass

    return None, None


def _get_model_barrier_from_step(step: Dict[str, Any], ego_id: str) -> Optional[float]:
    """
    Read barrier saved by eval script.
    Priority:
      1) step['barrier']
      2) step['logs'][ego_id]['barrier'] (fallback if you later store it there)
    """
    if "barrier" in step:
        try:
            v = step["barrier"]
            if v is None:
                return None
            return float(v)
        except Exception:
            pass

    lg = get_step_log(step, ego_id)
    if isinstance(lg, dict) and "barrier" in lg:
        try:
            return float(lg["barrier"])
        except Exception:
            pass

    return None


def summarize_rollout(
    rollout_steps: List[Any],
    rollout_idx: int,
    half_road: float,
) -> Dict[str, Any]:
    steps = [find_step_dict(s) for s in rollout_steps]
    T = len(steps)
    ego_id = pick_ego_id_from_step(steps[0]) if T else "unknown"
    agent_ids = get_agent_ids_from_step(steps[0]) if T else []
    agent_count = len(agent_ids)

    crashed_any = False
    lane_any = False
    rot_any = False
    done_any = False
    overlap_max = 0.0
    n_passed_max = 0

    # ego trajectories if present
    s_list, d_list, v_list = [], [], []
    x_list, y_list, yaw_list, t_list = [], [], [], []

    # lane safety
    lane_margin_list = []   # half_road - |d|

    # cross-track error ey(t) = d(t)
    ey_list = []            # signed d
    ey_abs_list = []        # |d|
    outlane_bool_list = []  # 1{|d| > half_road}

    # MODEL barrier saved by eval
    model_barrier_list = []

    # action series for uncertainty
    u1_list, u2_list = [], []

    for st in steps:
        info = get_step_info(st, ego_id)
        crashed_any = crashed_any or bool(info.get("crashed", False))
        lane_any    = lane_any    or bool(info.get("out_of_lane", False))
        rot_any     = rot_any     or bool(info.get("exceed_max_rot", False))
        done_any    = done_any    or bool(info.get("done", False))

        ov = info.get("overlap", None)
        if ov is not None:
            arr = as_array(ov)
            if arr.size:
                overlap_max = max(overlap_max, float(np.max(np.abs(arr))))

        npass = info.get("n_passed", 0)
        try:
            n_passed_max = max(n_passed_max, int(npass))
        except Exception:
            pass

        log_ego = get_step_log(st, ego_id)

        # ego series
        s_ego = _get_float_from_log_or_step("s", log_ego, st)
        d_ego = _get_float_from_log_or_step("d", log_ego, st)
        v_ego = _get_float_from_log_or_step("v", log_ego, st)
        x_ego, y_ego = _get_xy_from_log_or_step(log_ego, st)

        if s_ego is not None:
            s_list.append(s_ego)
        if d_ego is not None:
            d_list.append(d_ego)
        if v_ego is not None:
            v_list.append(v_ego)

        yaw = _get_float_from_log_or_step("yaw", log_ego, st)
        tt  = _get_float_from_log_or_step("t", log_ego, st)
        if x_ego is not None:
            x_list.append(x_ego)
        if y_ego is not None:
            y_list.append(y_ego)
        if yaw is not None:
            yaw_list.append(yaw)
        if tt is not None:
            t_list.append(tt)

        # lane margin + ey metrics
        if d_ego is not None:
            absd = abs(d_ego)
            lane_margin_list.append(half_road - absd)

            ey_list.append(d_ego)
            ey_abs_list.append(absd)
            outlane_bool_list.append(1.0 if absd > half_road else 0.0)

        # actions (u1,u2)
        u1, u2 = _get_u1_u2_from_step(st, ego_id)
        if u1 is not None:
            u1_list.append(u1)
        if u2 is not None:
            u2_list.append(u2)

        # MODEL barrier from eval
        mb = _get_model_barrier_from_step(st, ego_id)
        if mb is not None:
            model_barrier_list.append(mb)

    s_arr = np.asarray(s_list) if s_list else np.asarray([])
    d_arr = np.asarray(d_list) if d_list else np.asarray([])
    v_arr = np.asarray(v_list) if v_list else np.asarray([])
    lane_arr = np.asarray(lane_margin_list) if lane_margin_list else np.asarray([])

    ey_arr = np.asarray(ey_list) if ey_list else np.asarray([])
    ey_abs_arr = np.asarray(ey_abs_list) if ey_abs_list else np.asarray([])
    outlane_arr = np.asarray(outlane_bool_list) if outlane_bool_list else np.asarray([])

    u1_arr = np.asarray(u1_list) if u1_list else np.asarray([])
    u2_arr = np.asarray(u2_list) if u2_list else np.asarray([])

    mb_arr = np.asarray(model_barrier_list, dtype=float) if model_barrier_list else np.asarray([])

    # success definition (keep as-is)
    success = int((not crashed_any) and (not lane_any) and overlap_max <= 1e-9)
    # pass definition
    passed = int(not crashed_any)

    row = {
        "rollout_idx": rollout_idx,
        "T": T,
        "ego_agent_id": ego_id,
        "agent_count": agent_count,
        "agent_ids": ",".join(agent_ids),

        "success": success,
        "pass": passed,
        "crashed_any": int(crashed_any),
        "out_of_lane_any": int(lane_any),
        "exceed_max_rot_any": int(rot_any),
        "done_any": int(done_any),
        "overlap_max": overlap_max,
        "n_passed_max": n_passed_max,
    }

    # trajectory stats
    row.update({
        "s_final": float(s_arr[-1]) if s_arr.size else np.nan,
        "s_max": float(np.max(s_arr)) if s_arr.size else np.nan,
        "d_abs_mean": float(np.mean(np.abs(d_arr))) if d_arr.size else np.nan,
        "d_abs_max": float(np.max(np.abs(d_arr))) if d_arr.size else np.nan,
        "v_mean": float(np.mean(v_arr)) if v_arr.size else np.nan,
        "v_std": float(np.std(v_arr)) if v_arr.size else np.nan,
    })

    # lane margin stats
    row.update({
        "lane_margin_min": float(np.min(lane_arr)) if lane_arr.size else np.nan,
        "lane_margin_mean": float(np.mean(lane_arr)) if lane_arr.size else np.nan,
    })

    # cross-track error metrics (ey = d)
    row.update({
        "ey_abs_mean": float(np.mean(ey_abs_arr)) if ey_abs_arr.size else np.nan,
        "ey_var": float(np.var(ey_arr)) if ey_arr.size else np.nan,  # signed variance
        "time_out_of_lane": float(np.mean(outlane_arr)) if outlane_arr.size else np.nan,
    })

    # MODEL barrier stats (from step_results['barrier'])
    row.update({
        "barrier_min": float(np.min(mb_arr)) if mb_arr.size else np.nan,
        "barrier_mean": float(np.mean(mb_arr)) if mb_arr.size else np.nan,
    })

    # action uncertainty stats
    row.update({
        "u1_mean": float(np.mean(u1_arr)) if u1_arr.size else np.nan,
        "u2_mean": float(np.mean(u2_arr)) if u2_arr.size else np.nan,
        "u1_std": float(np.std(u1_arr)) if u1_arr.size else np.nan,
        "u2_std": float(np.std(u2_arr)) if u2_arr.size else np.nan,
        "u1_abs_mean": float(np.mean(np.abs(u1_arr))) if u1_arr.size else np.nan,
        "u2_abs_mean": float(np.mean(np.abs(u2_arr))) if u2_arr.size else np.nan,
    })

    return row


def benchmark_one_pkl(pkl_path: str, road_width: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    data = load_pkl(pkl_path)

    if not isinstance(data, list):
        raise ValueError(f"Expected list at top, got {type(data)}")

    rollouts = data
    # If it's wrapped like [rollouts], unwrap (some variants dump [all_results])
    if len(rollouts) == 1 and isinstance(rollouts[0], list) and len(rollouts[0]) == 100:
        rollouts = rollouts[0]

    half_road = float(road_width) / 2.0
    rows = [summarize_rollout(rollouts[i], i, half_road=half_road) for i in range(len(rollouts))]
    df = pd.DataFrame(rows)

    N = len(df)
    summ = {
        "name": Path(pkl_path).stem,
        "file": str(pkl_path),
        "N": N,
        "road_width": float(road_width),

        "success_rate": float(df["success"].mean()) if N else np.nan,
        "pass_rate": float(df["pass"].mean()) if N else np.nan,
        "crash_rate": float(df["crashed_any"].mean()) if N else np.nan,
        "lane_rate": float(df["out_of_lane_any"].mean()) if N else np.nan,
        "rot_rate": float(df["exceed_max_rot_any"].mean()) if N else np.nan,
        "overlap_max": float(df["overlap_max"].max()) if N else np.nan,
        "n_passed_mean": float(df["n_passed_max"].mean()) if N else np.nan,

        "agent_count_mean": float(df["agent_count"].mean()) if N else np.nan,
        "T_mean": float(df["T"].mean()) if N else np.nan,

        "s_final_mean": float(np.nanmean(df["s_final"])) if N else np.nan,
        "d_abs_mean": float(np.nanmean(df["d_abs_mean"])) if N else np.nan,
        "d_abs_max": float(np.nanmax(df["d_abs_max"])) if N else np.nan,
        "v_mean": float(np.nanmean(df["v_mean"])) if N else np.nan,

        # lane margin aggregates
        "lane_margin_min_mean": float(np.nanmean(df["lane_margin_min"])) if N else np.nan,
        "lane_margin_min_min": float(np.nanmin(df["lane_margin_min"])) if N else np.nan,

        # ey aggregates
        "ey_abs_mean_mean": float(np.nanmean(df["ey_abs_mean"])) if N else np.nan,
        "ey_abs_mean_max": float(np.nanmax(df["ey_abs_mean"])) if N else np.nan,
        "ey_var_mean": float(np.nanmean(df["ey_var"])) if N else np.nan,
        "ey_var_max": float(np.nanmax(df["ey_var"])) if N else np.nan,
        "time_out_of_lane_mean": float(np.nanmean(df["time_out_of_lane"])) if N else np.nan,
        "time_out_of_lane_max": float(np.nanmax(df["time_out_of_lane"])) if N else np.nan,

        # MODEL barrier aggregates
        "barrier_min_mean": float(np.nanmean(df["barrier_min"])) if N else np.nan,
        "barrier_min_min": float(np.nanmin(df["barrier_min"])) if N else np.nan,
        "barrier_mean_mean": float(np.nanmean(df["barrier_mean"])) if N else np.nan,

        # action uncertainty aggregates
        "u1_std_mean": float(np.nanmean(df["u1_std"])) if N else np.nan,
        "u2_std_mean": float(np.nanmean(df["u2_std"])) if N else np.nan,
        "u1_abs_mean": float(np.nanmean(df["u1_abs_mean"])) if N else np.nan,
        "u2_abs_mean": float(np.nanmean(df["u2_abs_mean"])) if N else np.nan,
    }
    return df, summ



def parse_model_and_seed(name: str) -> Tuple[str, Optional[int]]:
    """
    Parse names such as:
      results_posafenet_seed_0
      posafenet_big_hard_seed_2
      results-posafenet-seed1

    Returns:
      model_name: name with the seed suffix removed
      seed: parsed integer, or None when no seed is present
    """
    match = re.search(r"(?i)(?:[_-]seed[_-]?)(\d+)(?:$|[_-])", name)
    if match is None:
        return name, None

    seed = int(match.group(1))
    model_name = name[:match.start()].rstrip("_-")
    if model_name.startswith("results_"):
        model_name = model_name[len("results_"):]
    return model_name, seed


def aggregate_across_seeds(seed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate one benchmark-summary row per seed into one row per model.

    SD is the sample standard deviation across seeds (ddof=1).
    Two tables are returned:
      1) wide numeric table with <metric>_mean and <metric>_sd columns
      2) publication-friendly table with strings formatted as "mean ± sd"
    """
    if seed_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = seed_df.copy()
    parsed = work["name"].astype(str).map(parse_model_and_seed)
    work["model"] = parsed.map(lambda x: x[0])
    work["seed"] = parsed.map(lambda x: x[1])

    missing_seed = work["seed"].isna()
    if missing_seed.any():
        bad_names = work.loc[missing_seed, "name"].tolist()
        raise ValueError(
            "Could not parse seed from these input names: "
            + ", ".join(map(str, bad_names))
            + ". Expected a suffix such as _seed_0."
        )

    work["seed"] = work["seed"].astype(int)

    # Metadata are not performance metrics and are therefore not averaged.
    excluded = {"name", "file", "model", "seed", "N", "road_width"}
    metric_cols = [
        c for c in work.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(work[c])
    ]

    numeric_rows = []
    formatted_rows = []

    for model, group in work.groupby("model", sort=True):
        group = group.sort_values("seed")
        seeds = ",".join(str(x) for x in group["seed"].tolist())

        numeric_row = {
            "model": model,
            "n_seeds": int(len(group)),
            "seeds": seeds,
            "episodes_per_seed": (
                int(group["N"].iloc[0])
                if group["N"].nunique(dropna=False) == 1
                else ",".join(str(int(x)) for x in group["N"])
            ),
            "road_width": (
                float(group["road_width"].iloc[0])
                if group["road_width"].nunique(dropna=False) == 1
                else np.nan
            ),
        }
        formatted_row = {
            "model": model,
            "n_seeds": int(len(group)),
            "seeds": seeds,
        }

        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            mean = float(values.mean()) if values.notna().any() else np.nan
            sd = float(values.std(ddof=1)) if values.notna().sum() >= 2 else np.nan

            numeric_row[f"{metric}_mean"] = mean
            numeric_row[f"{metric}_sd"] = sd
            formatted_row[metric] = (
                f"{mean:.6f} ± {sd:.6f}"
                if np.isfinite(mean) and np.isfinite(sd)
                else f"{mean:.6f} ± nan"
                if np.isfinite(mean)
                else "nan ± nan"
            )

        numeric_rows.append(numeric_row)
        formatted_rows.append(formatted_row)

    return pd.DataFrame(numeric_rows), pd.DataFrame(formatted_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default=None)
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--outdir", type=str, default="./bench_out_2d")
    ap.add_argument("--road-width", type=float, default=DEFAULT_ROAD_WIDTH)
    args = ap.parse_args()

    road_width = float(args.road_width)

    paths = []
    if args.inputs:
        paths.extend(args.inputs)
    if args.glob:
        paths.extend(glob.glob(os.path.expanduser(args.glob)))
    paths = sorted(list(dict.fromkeys(paths)))
    if not paths:
        raise SystemExit("No pkl inputs.")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for p in paths:
        df, summ = benchmark_one_pkl(p, road_width=road_width)
        summaries.append(summ)

        stem = Path(p).stem
        df.to_csv(outdir / f"rollout_summary_{stem}.csv", index=False)

        # Print compact summary
        print(
            f"{stem}: N={summ['N']} "
            f"success={summ['success_rate']:.3f} pass={summ['pass_rate']:.3f} "
            f"crash={summ['crash_rate']:.3f} lane={summ['lane_rate']:.3f} "
            f"lane_margin(min/mean)=("
            f"{summ['lane_margin_min_min']:.3f},"
            f"{summ['lane_margin_min_mean']:.3f}) "
            f"ey_abs_mean={summ['ey_abs_mean_mean']:.3f} "
            f"ey_var={summ['ey_var_mean']:.3f} "
            f"time_out_of_lane={summ['time_out_of_lane_mean']:.3f} "
            f"barrier_min(min/mean)=("
            f"{summ['barrier_min_min']:.3f},"
            f"{summ['barrier_min_mean']:.3f}) "
            f"u_std=({summ['u1_std_mean']:.3f},{summ['u2_std_mean']:.3f})"
        )

    seed_summary_df = pd.DataFrame(summaries)
    seed_summary_path = outdir / "benchmark_summary_by_seed.csv"
    seed_summary_df.to_csv(seed_summary_path, index=False)

    multiseed_df, multiseed_formatted_df = aggregate_across_seeds(seed_summary_df)

    multiseed_path = outdir / "benchmark_summary_multiseed_mean_sd.csv"
    formatted_path = outdir / "benchmark_summary_multiseed_formatted.csv"

    multiseed_df.to_csv(multiseed_path, index=False)
    multiseed_formatted_df.to_csv(formatted_path, index=False)

    print(f"[OK] Wrote per-seed results: {seed_summary_path}")
    print(f"[OK] Wrote multiseed mean/SD: {multiseed_path}")
    print(f"[OK] Wrote formatted mean ± SD: {formatted_path}")

    if not multiseed_formatted_df.empty:
        print("\n=== Multi-seed mean ± SD ===")
        display_cols = [
            c for c in [
                "model",
                "success_rate",
                "pass_rate",
                "crash_rate",
                "lane_rate",
                "s_final_mean",
                "lane_margin_min_mean",
                "ey_abs_mean_mean",
                "time_out_of_lane_mean",
                "barrier_min_mean",
            ]
            if c in multiseed_formatted_df.columns
        ]
        print(multiseed_formatted_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()


"""
Example:
python benchmark_multiseed.py \
  --glob "./eval_out/pkl_multiseed/*.pkl" \
  --outdir "./eval_out/summary_multiseed" \
  --road-width 4.0

Expected filename stems include a seed suffix, for example:
  results_posafenet_seed_0
  results_posafenet_seed_1
  results_posafenet_seed_2
"""