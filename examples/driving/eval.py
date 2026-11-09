#!/usr/bin/env python3
import os
import argparse
import importlib
import json
import pickle
import numpy as np
from skvideo.io import FFmpegWriter
import torch
import time

from eval_tools import utils
import vista


# ============================================================
# Args
# ============================================================

def parse_args():
    """
    Two-stage parsing:
    - If --make-scenario is present: only parse scenario-related args (no required model/out-dir).
    - Otherwise: parse full args (model-module required; out-dir optional like original).
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--make-scenario', action='store_true', default=False)
    pre.add_argument('--scenario-out', type=str, default='scenario.npz')
    pre.add_argument('--scenario-npz', type=str, default=None)
    pre.add_argument('--seed', type=int, default=0)
    pre.add_argument('--n-episodes', type=int, default=1)
    pre.add_argument('--trace-paths', type=str, nargs='+', default=None)
    pre.add_argument('--mesh-dir', type=str, default=None)
    pre.add_argument('--n-agents', type=int, default=3)
    pre.add_argument('--reset-mode', type=str, default='default',
                     choices=['default', 'segment_start', 'uniform'])
    pre.add_argument('--road-width', type=float, default=4.)
    pre.add_argument('--use-curvilinear-dynamics', action='store_true', default=False)
    pre.add_argument('--no-cuda', action='store_true', default=False)
    pre_ns, _ = pre.parse_known_args()

    if pre_ns.make_scenario:
        parser = argparse.ArgumentParser(description='Generate fixed scenario for VISTA eval')
        parser.add_argument('--make-scenario', action='store_true', default=False)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--n-episodes', type=int, default=1)
        parser.add_argument('--scenario-out', type=str, default='scenario.npz')

        # keep env args so your bash doesn't break
        parser.add_argument('--trace-paths', type=str, nargs='+', required=True)
        parser.add_argument('--mesh-dir', type=str, default=None)
        parser.add_argument('--n-agents', type=int, default=3)
        parser.add_argument('--reset-mode', type=str, default='default',
                            choices=['default', 'segment_start', 'uniform'])
        parser.add_argument('--road-width', type=float, default=4.)
        parser.add_argument('--use-curvilinear-dynamics', action='store_true', default=False)
        parser.add_argument('--no-cuda', action='store_true', default=False)

        # optional out-dir for saving config, not required
        parser.add_argument('--out-dir', type=str, default=None)
        return parser.parse_args()

    # ---------- full eval args (aligned to your original eval) ----------
    parser = argparse.ArgumentParser(description='Run VISTA for evaluation (scenario + traj dump supported)')

    # model
    parser.add_argument('--model-module', type=str, required=True,
                        help='Model class; should be consistent with the checkpoint')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--state-net-model-module', type=str, default=None,
                        help='Model class for state net; should be consistent with the checkpoint')
    parser.add_argument('--state-net-ckpt', type=str, default=None,
                        help='Path to checkpoint for state net model')
    parser.add_argument('--ego-init-v', type=float, default=6.,
                        help='Initial velocity used for acceleration control')
    parser.add_argument('--set-obs-d-lower-bound', type=float, default=None,
                        help='Lower bound for predicted obs_d')
    parser.add_argument('--use-reference-control', action='store_true', default=False,
                        help='Use reference control')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Not use cuda')
    parser.add_argument('--lf-cbf-threshold', type=float, default=2.,
                        help='Threshold for lane following CBF')
    parser.add_argument('--use-lf-cbf-only', action='store_true', default=False,
                        help='Use lane following CBF only')

    # general vista arguments
    parser.add_argument('--loadmat', type=str, nargs='+', default=None,
                        help='Load .pkl for env and .mat file for control')
    parser.add_argument('--trace-paths', type=str, nargs='+', required=True,
                        help='Path to the traces to use for simulation')
    parser.add_argument('--mesh-dir', type=str, default=None,
                        help='Directory of meshes for virtual agents')
    parser.add_argument('--n-agents', type=int, default=2,
                        help='Number of agents')
    parser.add_argument('--use-display', action='store_true', default=False,
                        help='Use VISTA default display')
    parser.add_argument('--road-width', type=float, default=4.,
                        help='Road width in VISTA')
    parser.add_argument('--reset-mode', type=str, default='default',
                        choices=['default', 'segment_start', 'uniform'],
                        help='Trace reset mode in VISTA')
    parser.add_argument('--use-curvilinear-dynamics', action='store_true', default=False,
                        help='Use curvilinear dynamics for vehicle dynamics')
    parser.add_argument('--max-step', type=int, default=None,
                        help='Maximal step to be ran')
    parser.add_argument('--init-dist-range', type=float, nargs='+', default=[15., 25.],
                        help='Initial distance range of obstacle (unused when scenario_npz is set)')
    parser.add_argument('--init-lat-noise-range', type=float, nargs='+', default=[1., 1.5],
                        help='Initial lateral displacement of obstacle')

    # evaluation and logging
    parser.add_argument('--n-episodes', type=int, default=1,
                        help='Number of episodes')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Directory to save output')
    parser.add_argument('--save-video', action='store_true', default=False,
                        help='Save video for every episodes')
    
    # eval-time image noise for robustness test
    parser.add_argument('--eval-noise-std', type=float, default=0.0,
                        help='Gaussian noise std added to image observations during evaluation')
    parser.add_argument('--eval-noise-clip-min', type=float, default=0.0,
                        help='Minimum value after adding eval image noise')
    parser.add_argument('--eval-noise-clip-max', type=float, default=1.0,
                        help='Maximum value after adding eval image noise')

    # reproducible scenario (NEW)
    parser.add_argument('--seed', type=int, default=0, help='Random seed for scenario generation')
    parser.add_argument('--scenario-npz', type=str, default=None,
                        help='Path to scenario npz containing fixed s positions per episode')
    parser.add_argument('--make-scenario', action='store_true', default=False,
                        help='Generate scenario npz and exit (uses --seed and --n-episodes)')
    parser.add_argument('--scenario-out', type=str, default='scenario.npz',
                        help='Output path for scenario when using --make-scenario')

    # compact trajectory dump for plotting (NEW)
    parser.add_argument('--save-traj-npz', action='store_true', default=False,
                        help='Save compact trajectory npz for plotting')
    parser.add_argument('--traj-npz-name', type=str, default=None,
                        help='Override traj npz filename (default uses model name)')

    # PoSafeNet closed-form projection diagnostics
    parser.add_argument('--collect-projection-stats', action='store_true', default=False,
                        help='Collect immediate residual statistics from PoSafeNet half-space projections')
    parser.add_argument('--projection-stats-name', type=str, default='projection_stats.json',
                        help='Filename for saved projection statistics inside --out-dir')

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def move_to_cuda(x):
    """Recursively move nested structures to cuda."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, list):
        return [move_to_cuda(v) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_cuda(v) for v in x)
    if isinstance(x, dict):
        return {k: move_to_cuda(v) for k, v in x.items()}
    raise TypeError(f"Unsupported type for cuda move: {type(x)}")


def extract_barrier_scalar(model):
    """
    Robustly extract a scalar barrier from model.barrier.
    - If barrier is vector/matrix -> return min as "most unsafe" barrier.
    - If missing -> return NaN.
    """
    if not hasattr(model, "barrier"):
        return np.nan
    b = getattr(model, "barrier")
    if b is None:
        return np.nan
    if torch.is_tensor(b):
        b = b.detach().cpu().numpy()
    b = np.asarray(b).reshape(-1)
    if b.size == 0:
        return np.nan
    return float(np.min(b))


def infer_control_mode_from_model(model):
    if not hasattr(model, "hparams") or not hasattr(model.hparams, "output_mode"):
        raise NotImplementedError("Model has no hparams.output_mode; cannot infer control_mode.")
    out = list(model.hparams.output_mode)
    s = set(out)
    if s == set(['delta', 'v']):
        return 'delta-v'
    if s == set(['omega', 'a']):
        return 'omega-a'
    if set(['delta', 'v', 'omega', 'a']).issubset(s):
        return 'omega-a'
    raise NotImplementedError(f'No corresponding control mode for {model.hparams.output_mode}')


def build_model_kwargs_and_patch_hparams(args, model):
    # model kwargs / hparams patch (same as your original)
    if args.model_module == 'barrier_net':
        model_kwargs = {'solver': 'cvxpy', 'store_intermediate_data': True}
        model.hparams.not_use_gt = True
        if args.use_reference_control:
            model.hparams.model_type = 'deri_ref'
    elif args.model_module in ['bnet_up', 'abnet', 'abnet_posafenet', 
                            'posafenet', 'posafenet_hard', 'posafenet_big_hard']:
        model_kwargs = {'solver': 'cvxpy', 'store_intermediate_data': False}
        model.hparams.not_use_gt = True
        if args.use_reference_control:
            model.hparams.model_type = 'deri_ref'
    elif args.model_module == 'old_barrier_net':
        model_kwargs = {'solver': 'cvxpy', 'store_intermediate_data': True}
        model.hparams.not_use_gt = True
        model.hparams.use_color_jitter = False
        model.hparams.use_fixed_standardize = True
    else:
        model_kwargs = dict()

    # lf cbf knobs
    model.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model.hparams.use_lf_cbf_only = args.use_lf_cbf_only
    return model_kwargs


def load_or_make_scenario(args):
    if args.make_scenario:
        out_dir = os.path.dirname(os.path.abspath(args.scenario_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        rng = np.random.RandomState(args.seed)
        s_ego, s_obs1, s_obs2 = [], [], []

        # ego in [0,10], obs1=ego+U[20,30], obs2=obs1+U[30,40]
        for _ in range(args.n_episodes):
            s0 = float(rng.uniform(0.0, 10.0))
            d1 = float(rng.uniform(20.0, 30.0))
            d2 = float(rng.uniform(30.0, 40.0))
            s1 = s0 + d1
            s2 = s1 + d2
            s_ego.append(s0)
            s_obs1.append(s1)
            s_obs2.append(s2)

        np.savez_compressed(
            args.scenario_out,
            s_ego=np.asarray(s_ego, dtype=np.float32),
            s_obs1=np.asarray(s_obs1, dtype=np.float32),
            s_obs2=np.asarray(s_obs2, dtype=np.float32),
        )
        print(f"[Scenario] saved to {args.scenario_out}")
        return None

    if args.scenario_npz is None:
        return None

    sc = np.load(args.scenario_npz)
    s_ego = sc['s_ego']
    s_obs1 = sc['s_obs1']
    s_obs2 = sc['s_obs2']
    if len(s_ego) < args.n_episodes:
        raise ValueError(f"scenario has {len(s_ego)} eps but requested {args.n_episodes}")
    return (s_ego, s_obs1, s_obs2)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # ---------- scenario make mode ----------
    sc = load_or_make_scenario(args)
    if args.make_scenario:
        # optionally dump config
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, "eval_config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
        return

    # ---------- load model ----------
    LitModel = importlib.import_module(f'.{args.model_module}', 'models').LitModel
    if args.model_module == "cnn_lstm":
        model = LitModel.load_from_checkpoint(
            args.ckpt,
            output_mode=["v", "delta"],
            loss_coef_v=0.25,
            loss_coef_delta=0.25,
            loss_coef_a=None,
            loss_coef_omega=None,
        )
    else:
        model = LitModel.load_from_checkpoint(
            args.ckpt,
            strict=False,
        )

    if hasattr(model, "hparams"):
        model.hparams.model_module = args.model_module

    args.control_mode = infer_control_mode_from_model(model)
    model_kwargs = build_model_kwargs_and_patch_hparams(args, model)

    if not args.no_cuda:
        model.cuda()
    model.eval()

    if args.collect_projection_stats:
        if not hasattr(model, 'enable_projection_stats'):
            raise AttributeError(
                'The loaded model does not provide enable_projection_stats(). '
                'Use the modified PoSafeNet model implementation.'
            )
        model.enable_projection_stats(enabled=True, reset=True)
        print('[Projection Stats] enabled')

    print(f"\n{'='*60}")
    print(f"[MODEL DEBUG]")
    print(f"  model_module:         {args.model_module}")
    print(f"  model class:          {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"  learn_mixture_weight: {getattr(model, 'learn_mixture_weight', 'N/A')}")
    if hasattr(model, 'wt'):
        print(f"  wt.requires_grad:    {model.wt.requires_grad}")
        print(f"  wt.data:             {model.wt.data.cpu()}")
        print(f"  softmax(wt):         {torch.softmax(model.wt, dim=0).data.cpu()}")
        print(f"  argmax(wt):          {torch.argmax(model.wt).item()}")
        K = model.wt.shape[0]
        print(f"  abnet_heads:         {K}")
    if hasattr(model, 'hparams'):
        print(f"  hparams.model_module: {getattr(model.hparams, 'model_module', 'N/A')}")
    print(f"  model.training:      {model.training}")
    print(f"{'='*60}\n")

    # optional state net
    model_state_net = None
    if args.state_net_model_module:
        LitModelStateNet = importlib.import_module(f'.{args.state_net_model_module}', 'models').LitModel
        model_state_net = LitModelStateNet.load_from_checkpoint(args.state_net_ckpt)
        if not args.no_cuda:
            model_state_net.cuda()
        model_state_net.eval()
        model.hparams.use_indep_state_net = True
        model.hparams.indep_state_net_output = model_state_net.hparams.output_mode

    # ---------- env setup (same as original) ----------
    if args.loadmat:
        from scipy.io import loadmat  # noqa: F401

        with open(args.loadmat[0], 'rb') as f:
            env_loaded = pickle.load(f)

        # patch trace paths
        for i in range(len(env_loaded.world.traces)):
            env_loaded.world.traces[i]._trace_path = os.path.expanduser(args.trace_paths[i])

        # HACK bounds & reset internal s
        for ag in env_loaded.world.agents:
            ag._control_mode = 'omega-a'
            ag.human_dynamics._s = 0.
            ag.ego_dynamics._s = 0.
            ag.relative_state._s = 0.
            ag._road_dynamics._s = 0.
            ag.ego_dynamics._omega_bound = [-1.0, 1.0]
            ag.ego_dynamics._a_bound = [-7.0, 7.0]
        for _a in env_loaded.world.agents:
            _a._config['use_curvilinear_dynamics'] = False

        # rebuild fresh env from args
        env = utils.get_env(args)
        ego_agent = env.world.agents[0]
        args.ego_init_v = 6

        # copy closest obstacle dynamics
        closest_dist = 9999.
        closest_agent_in_loaded_env = None
        for ag_i, ag in enumerate(env_loaded.world.agents):
            if ag_i == 0:
                continue
            dist = np.linalg.norm(ag.ego_dynamics.numpy()[:2] - ego_agent.ego_dynamics.numpy()[:2])
            if dist < closest_dist:
                closest_dist = dist
                closest_agent_in_loaded_env = ag
        if closest_agent_in_loaded_env is not None and len(env.world.agents) > 1:
            env.world.agents[1]._ego_dynamics = closest_agent_in_loaded_env.ego_dynamics.copy()
    else:
        env = utils.get_env(args)
        ego_agent = env.world.agents[0]
        ego_agent._ego_dynamics._v = args.ego_init_v

    # display/video
    display = None
    if args.use_display:
        display_config = dict(road_buffer_size=1000,)
        display = vista.Display(env.world, display_config=display_config)
        display.reset()

    dt = 1.0 / 10.0

    # ---------- output dir ----------
    all_results = None
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, 'eval_config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        all_results = []

    # ---------- global stats ----------
    crash_cnt = 0
    out_lane_cnt = 0
    n_passed_cnt = 0

    barrier_ep_mins = []
    barrier_ep_means = []
    ctime_ep_means = []

    # ---------- traj buffers for plotting (NEW) ----------
    ego_xy_eps, obs1_xy_eps, obs2_xy_eps = [], [], []
    barrier_ts_eps, lengths = [], []

    stop_eval = False

    for episode_i in range(args.n_episodes):
        print(f'Episode {episode_i:04d}')

        observations = env.reset()

        # --- place s using scenario if provided; else fall back to your random logic ---
        if sc is not None:
            s_ego, s_obs1, s_obs2 = sc
            s0 = float(s_ego[episode_i])
            env.world.agents[0]._ego_dynamics._s = s0
            if len(env.world.agents) > 1:
                env.world.agents[1]._ego_dynamics._s = float(s_obs1[episode_i])
            if len(env.world.agents) > 2:
                env.world.agents[2]._ego_dynamics._s = float(s_obs2[episode_i])
        else:
            # your original random placement
            S_MIN, S_MAX = 0.0, 10.0
            s0 = float(np.random.uniform(S_MIN, S_MAX))
            env.world.agents[0]._ego_dynamics._s = s0
            if len(env.world.agents) >= 3:
                s1 = np.random.uniform(10.0, 20.0)
                s2 = s1 + np.random.uniform(20.0, 30.0)
                env.world.agents[1]._ego_dynamics._s = s0 + s1
                env.world.agents[2]._ego_dynamics._s = s0 + s2

        if display is not None:
            display.reset()

        ego_agent = env.world.agents[0]
        ego_agent._ego_dynamics._v = args.ego_init_v

        episode_results = []
        if args.save_video:
            assert args.out_dir is not None
            video_dir = os.path.join(args.out_dir, 'video')
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f'episode_{episode_i:04d}.mp4')
            rate = f'{(1. / dt)}'
            video_writer = FFmpegWriter(
                video_path,
                inputdict={'-r': rate},
                outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-r': rate}
            )
        else:
            video_writer = None

        # init rnn states
        if hasattr(model, 'get_initial_state'):
            rnn_state = model.get_initial_state(batch_size=1)
            if not args.no_cuda:
                rnn_state = move_to_cuda(rnn_state)
        else:
            rnn_state = None

        rnn_state_state_net = None
        if model_state_net is not None and hasattr(model_state_net, 'get_initial_state'):
            rnn_state_state_net = model_state_net.get_initial_state(batch_size=1)
            if not args.no_cuda:
                rnn_state_state_net = move_to_cuda(rnn_state_state_net)

        # per-episode buffers
        barrier_i = []
        time_i = []

        ego_xy, o1_xy, o2_xy = [], [], []

        head_disagree_i = []       # <-- 新增
        head_actions_i = []        # <-- 新增

        done = False
        step_i = 0
        step_results = None

        while not done:
            try:
                model_inputs = utils.preprocess_obs(env, ego_agent, model, observations)
                # ===== eval-time image noise for robustness test =====
                if args.eval_noise_std > 0:
                    img = model_inputs[0]  # expected shape: [B, T, C, H, W]
                    noise = args.eval_noise_std * torch.randn_like(img)
                    img = img + noise
                    img = torch.clamp(
                        img,
                        min=args.eval_noise_clip_min,
                        max=args.eval_noise_clip_max,
                    )
                    model_inputs[0] = img
                # ====================================================
                if not args.no_cuda:
                    model_inputs = [v.cuda() for v in model_inputs]

                with torch.no_grad():
                    # optional state net
                    if model_state_net is not None:
                        pred_state_net, rnn_state_state_net = model_state_net(
                            model_inputs, rnn_state_state_net, **model_kwargs
                        )

                        if args.set_obs_d_lower_bound is not None and 'obs_d' in model_state_net.hparams.output_mode:
                            obs_d_idx = model_state_net.hparams.output_mode.index('obs_d')
                            # pred_state_net shape assumed [B,T,D]
                            if torch.sign(pred_state_net[:, :, obs_d_idx]) > 0:
                                pred_state_net[:, :, obs_d_idx] = torch.clamp(
                                    pred_state_net[:, :, obs_d_idx], min=args.set_obs_d_lower_bound
                                )
                            else:
                                pred_state_net[:, :, obs_d_idx] = torch.clamp(
                                    pred_state_net[:, :, obs_d_idx], max=-args.set_obs_d_lower_bound
                                )

                        if getattr(model_state_net.hparams, "drop_obs_d_offset", False):
                            if 'obs_d' in model_state_net.hparams.output_mode:
                                obs_d_idx = model_state_net.hparams.output_mode.index('obs_d')
                                pred_state_net[:, :, obs_d_idx] += torch.sign(pred_state_net[:, :, obs_d_idx]) * 5
                            if 'dd' in model_state_net.hparams.output_mode:
                                dd_idx = model_state_net.hparams.output_mode.index('dd')
                                pred_state_net[:, :, dd_idx] += torch.sign(pred_state_net[:, :, dd_idx]) * 5

                        model_inputs.append(pred_state_net)

                    # time the forward
                    start = time.time()
                    # pred, rnn_state = model(model_inputs, rnn_state, **model_kwargs)
                    pred, rnn_state = model(model_inputs, rnn_state, log_head_disagreement=True, **model_kwargs) # <-- 新增 log_head_disagreement=True
                    termi = time.time()

                    if hasattr(model, '_head_log'):
                        head_disagree_i.append(model._head_log['max_disagreement'].copy())
                        head_actions_i.append(model._head_log['head_actions'].copy())

                time_i.append(termi - start)

                pred_np = pred.cpu().numpy()[0, 0]
                actions = utils.construct_actions(env, ego_agent, model, pred_np)

                observations, rewards, dones, infos = env.step(actions, dt)

                ego_id = ego_agent.id
                env_done_now = bool(dones[ego_id])
                has_crashed_now = bool(infos[ego_id].get('crashed', False))

                if has_crashed_now:
                    print(f"[Early Stop] episode {episode_i:04d}, step {step_i}: ego crashed.")

                if env_done_now and not has_crashed_now:
                    print(f"[Ignore Env Done] episode {episode_i:04d}, step {step_i}, infos={infos[ego_id]}")

                # Only crash stops the episode.
                # Env done is logged but ignored.
                done = has_crashed_now


                # record barrier + xy for plotting
                barrier_i.append(extract_barrier_scalar(model))

                ego_xy.append([float(env.world.agents[0].ego_dynamics.x),
                               float(env.world.agents[0].ego_dynamics.y)])
                if len(env.world.agents) > 1:
                    o1_xy.append([float(env.world.agents[1].ego_dynamics.x),
                                  float(env.world.agents[1].ego_dynamics.y)])
                else:
                    o1_xy.append([np.nan, np.nan])
                if len(env.world.agents) > 2:
                    o2_xy.append([float(env.world.agents[2].ego_dynamics.x),
                                  float(env.world.agents[2].ego_dynamics.y)])
                else:
                    o2_xy.append([np.nan, np.nan])

                # save step_results (for results.pkl)
                if args.out_dir:
                    logs = {}
                    for _a in env.world.agents:
                        if _a.id == ego_agent.id:
                            logs[_a.id] = utils.extract_logs(env, _a, model, pred_np)
                        else:
                            logs[_a.id] = utils.extract_logs(env, _a)
                    step_results = {
                        'ego_agent_id': ego_agent.id,
                        'logs': logs,
                        'actions': actions,
                        'infos': infos,
                        'barrier': extract_barrier_scalar(model),
                    }
                    episode_results.append(step_results)
                else:
                    step_results = {'infos': infos}  # minimal for counters

                # render
                if display is not None:
                    img = display.render()
                    img = utils.add_descriptions(env, ego_agent, model, pred_np, img)
                    if video_writer is not None:
                        video_writer.writeFrame(img)

                step_i += 1

                if args.max_step is not None and step_i >= args.max_step:
                    break

            except KeyboardInterrupt:
                stop_eval = True
                break

        if video_writer is not None:
            video_writer.close()

        # episode-level counts
        if step_results is not None:
            ego_id = ego_agent.id
            try:
                if args.out_dir and len(episode_results) > 0:
                    has_crashed = np.any([v['infos'][ego_id]['crashed'] for v in episode_results])
                    if has_crashed:
                        crash_cnt += 1
                    if episode_results[-1]['infos'][ego_id].get('out_of_lane', False):
                        out_lane_cnt += 1
                    if episode_results[-1]['infos'][ego_id].get('n_passed', 0):
                        n_passed_cnt += 1
                else:
                    if step_results['infos'][ego_id].get('crashed', False):
                        crash_cnt += 1
                    if step_results['infos'][ego_id].get('out_of_lane', False):
                        out_lane_cnt += 1
                    if step_results['infos'][ego_id].get('n_passed', 0):
                        n_passed_cnt += 1
            except Exception:
                pass

        # barrier stats
        barrier_arr = np.asarray(barrier_i, dtype=float)
        if np.all(np.isnan(barrier_arr)) or barrier_arr.size == 0:
            ep_bmin = np.nan
            ep_bmean = np.nan
        else:
            ep_bmin = float(np.nanmin(barrier_arr))
            ep_bmean = float(np.nanmean(barrier_arr))

        barrier_ep_mins.append(ep_bmin)
        barrier_ep_means.append(ep_bmean)

        if len(time_i) > 0:
            ctime_ep_means.append(float(np.mean(time_i)))

        print(f"[Barrier] episode_min={ep_bmin:.6f}, episode_mean={ep_bmean:.6f}")
        print(f"steps={step_i}")

        # store results.pkl
        if args.out_dir:
            all_results.append(episode_results)

        # store traj buffers
        ego_xy_eps.append(ego_xy)
        obs1_xy_eps.append(o1_xy)
        obs2_xy_eps.append(o2_xy)
        barrier_ts_eps.append(barrier_i)
        lengths.append(len(ego_xy))

        # ========== 新增：每个 episode 存 head disagreement ==========
        if args.out_dir and len(head_disagree_i) > 0:
            np.savez(
                os.path.join(args.out_dir, f'head_disagreement_ep{episode_i:04d}.npz'),
                max_disagreement=np.array(head_disagree_i),   # (T, B)
                head_actions=np.array(head_actions_i),         # (T, K, B, 2)
                barrier=np.array(barrier_i),                   # (T,)
            )
            print(f"[Head Log] saved ep{episode_i} disagree shape={np.array(head_disagree_i).shape}")
        # ==============================================================
        if stop_eval:
            break

    # global summary
    barrier_ep_mins = np.asarray(barrier_ep_mins, dtype=float)
    barrier_ep_means = np.asarray(barrier_ep_means, dtype=float)

    print('crashed:', crash_cnt)
    print('out_of_lane:', out_lane_cnt)
    print('n_passed:', n_passed_cnt)

    if len(ctime_ep_means) > 0:
        ctime_ep_means = np.asarray(ctime_ep_means, dtype=float)
        print('computation time ave:', np.mean(ctime_ep_means), 'std:', np.std(ctime_ep_means))

    print('safety (min over episodes):', np.nanmin(barrier_ep_mins))
    print('conservativeness (ep-mean) ave:', np.nanmean(barrier_ep_means), 'std:', np.nanstd(barrier_ep_means))

    # save PoSafeNet projection diagnostics
    if args.collect_projection_stats:
        if not hasattr(model, 'get_projection_stats_summary'):
            raise AttributeError(
                'The loaded model does not provide get_projection_stats_summary().'
            )

        projection_stats = model.get_projection_stats_summary()
        print('\n=== PoSafeNet projection statistics ===')
        print(json.dumps(projection_stats, indent=2))

        stats_dir = args.out_dir if args.out_dir else '.'
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, args.projection_stats_name)
        with open(stats_path, 'w') as f:
            json.dump(projection_stats, f, indent=2)
        print(f'[Projection Stats] saved to {stats_path}')

    # save results.pkl
    if args.out_dir:
        results_path = os.path.join(args.out_dir, 'results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)

    # save traj npz (NEW)
    if args.save_traj_npz:
        Tmax = int(max(lengths)) if len(lengths) > 0 else 0
        E = int(len(lengths))

        def pad_list_of_xy(list_eps, shape_last):
            out = np.full((E, Tmax, shape_last), np.nan, dtype=np.float32)
            for i, a in enumerate(list_eps):
                if len(a) == 0:
                    continue
                out[i, :len(a), :] = np.asarray(a, dtype=np.float32)
            return out

        def pad_list_of_scalar(list_eps):
            out = np.full((E, Tmax), np.nan, dtype=np.float32)
            for i, a in enumerate(list_eps):
                if len(a) == 0:
                    continue
                out[i, :len(a)] = np.asarray(a, dtype=np.float32)
            return out

        ego_xy = pad_list_of_xy(ego_xy_eps, 2)
        obs1_xy = pad_list_of_xy(obs1_xy_eps, 2)
        obs2_xy = pad_list_of_xy(obs2_xy_eps, 2)
        barrier_ts = pad_list_of_scalar(barrier_ts_eps)

        name = args.traj_npz_name
        if name is None:
            tag = os.path.splitext(os.path.basename(args.ckpt))[0] if args.ckpt else "nocheckpoint"
            name = f"traj_{args.model_module}_{tag}.npz"

        path = os.path.join(args.out_dir if args.out_dir else ".", name)

        # include scenario s if provided
        scenario_payload = {}
        if sc is not None:
            s_ego, s_obs1, s_obs2 = sc
            scenario_payload.update(dict(
                scenario_s_ego=np.asarray(s_ego[:E], dtype=np.float32),
                scenario_s_obs1=np.asarray(s_obs1[:E], dtype=np.float32),
                scenario_s_obs2=np.asarray(s_obs2[:E], dtype=np.float32),
            ))

        np.savez_compressed(
            path,
            ego_xy=ego_xy,
            obs1_xy=obs1_xy,
            obs2_xy=obs2_xy,
            barrier_ts=barrier_ts,
            lengths=np.asarray(lengths, dtype=np.int32),
            dt=np.float32(dt),
            road_width=np.float32(args.road_width),
            control_mode=str(args.control_mode),
            ego_init_v=np.float32(args.ego_init_v),
            **scenario_payload,
        )
        print(f"[Saved traj] {path}")


if __name__ == '__main__':
    main()