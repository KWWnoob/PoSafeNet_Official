# PoSafeNet

[arXiv](https://arxiv.org/abs/2601.22356)

**PoSafeNet: Safe Learning with Poset-Structured Neural Nets**

Kiwan Wong, Wei Xiao, Daniela Rus

Implementation of our PoSafeNet paper (NeurIPS 2026).

PoSafeNet is a multi-head safe-control network. Each head maps a nominal
control to a safe one by **sequential closed-form projection onto the CBF
half-spaces in a head-specific priority order** (a total order of the safety
constraints). The heads are combined by learned weights:

- **PoSafeNet (Mixture)**: softmax-weighted mixture of the heads.
- **PoSafeNet (Hard)**: one head is selected (Gumbel-softmax during training,
  argmax of the learned weights at inference).

This code builds on [ABNet](https://github.com/Weixy21/ABNet): the example
tasks and the VISTA driving pipeline come from ABNet / BarrierNet.

# Installation

```
conda env create -f environment.yml
conda activate posafenet
```

`environment.yml` pins the versions used to run all commands below (Python 3.9,
PyTorch 2.8.0 + CUDA 12.8). If cu128 does not match your machine, install a
suitable PyTorch build. Note that `pytorch-lightning==1.5.8` is required by the
driving example.

## Install VISTA (driving example only)
```
cd examples/driving/vista
pip install -e .
```
This is the modified copy of [VISTA](https://github.com/vista-simulator/vista)
used by BarrierNet / ABNet; use it instead of a pip-released VISTA. The driving
example also needs a system `ffmpeg` binary.

# Examples
Run the following in the terminal/environment before training or testing:
```
export PYTHONNOUSERSITE=True
export PYOPENGL_PLATFORM=egl
```

## 2D obstacle avoidance
The dataset is included (`data_v2/data/dataM_{train,valid,test}.mat`, rows
`[ep_id, t, px, py, theta, v, goal_y, omega, a]`).

### Train a model:
```
cd examples/2D_robot
python train_posafenet.py --model hard  --exp 0
python train_posafenet.py --model mixture --exp 0
```
20 epochs, 6 heads; checkpoints are written to
`log/posafenet_<model>/posafenet_<model>00/model_posa{00..19}.pth`.

### Test a model:
```
python eval_posafenet.py
```
Closed-loop Monte-Carlo evaluation (100 rollouts, internal noise on) of the
epoch-19 checkpoints of both variants; results in
`results/metrics_mc_runs100_noise1.csv`. Use `--hard-ckpt`, `--mixture-ckpt`,
`--methods` and `--mc-runs` to change the defaults.

## Manipulation
The dataset is included (`data_v2/data/dataM_{train,valid,test}.mat`, rows
`[ep_id, t, th1, w1, th2, w2, dst_x, dst_y, u1, u2]`).

### Train a model:
```
cd examples/Manipulator
python train_posafenet.py --model posafenet_hard    --seed 0 --skip-test
python train_posafenet.py --model posafenet_mixture --seed 0 --skip-test
```
20 epochs, 2 heads; checkpoints go to `log/posafenet_{hard,mixture}/posafenetNN/`.
To train both variants for the 6 seeds used in the evaluation (GPU via `GPU_ID`, default 0):
```
bash train_all_seeds.sh
```

### Test a model:
```
python eval_posafenet.py --train-seeds 0 1 2 3 4 5 --ckpt-epoch 19 --mc-runs 100 --bench-seed 0
```
Paired closed-loop evaluation: every model and seed sees the same episodes and
the same noise seeds. Results are in `results/multiseed_benchmark/`.

## Vision-based end-to-end autonomous driving
Please refer to the [BarrierNet repo](https://github.com/Weixy21/BarrierNet) for
the dataset and the VISTA trace / car mesh.

**Training data** (default location `examples/driving/data`, or set
`POSAFENET_DRIVING_DATA`):
`images/`, `state/`, `obstacle/`, `{train,val,test}_split.csv`. The split CSVs store
absolute paths, so rewrite them once after placing the data (backups are kept as `*.csv.bak`):
```
python examples/driving/scripts/relocate_dataset.py --data-src <path/to/data>
```

**VISTA data for testing** (default `$HOME/data/vista`, or set `VISTA_DATA_ROOT`):
```
$VISTA_DATA_ROOT/traces/20220113-131922_lexus_devens_outerloop_reverse/
$VISTA_DATA_ROOT/carpack01/
```

### Train a model:
```
cd examples/driving
SEED=0 bash ./scripts/train_posafenet.sh             # Mixture, 2 heads
SEED=0 bash ./scripts/train_posafenet_hard.sh        # Hard,    2 heads
SEED=0 bash ./scripts/train_posafenet_big.sh         # Mixture, 4 heads
SEED=0 bash ./scripts/train_posafenet_big_hard.sh    # Hard,    4 heads
bash ./scripts/train_all.sh                          # all 4 variants, seeds 0, 1, 2
```
Checkpoints go to `results/<variant>_seed_<k>/version_*/checkpoints/` (10 epochs,
batch 48, one checkpoint every half epoch, GPU 0; extra arguments are passed to
`run.py`, e.g. `--gpus "1,"`).

### Test a model:
```
bash ./scripts/eval_multiseed.sh
python benchmark_multiseed.py --glob "./eval_out/pkl_multiseed/*.pkl" --outdir ./eval_out/summary_multiseed
```
`eval_multiseed.sh` runs `eval.py` for the 4 variants × 3 seeds on the fixed
scenario `scenarios/fixed_scenario_seed0.npz` (10 episodes, 100 steps, 3 cars,
CPU). It evaluates the checkpoint `epoch=0-step=1948.ckpt` of each run.
`benchmark_multiseed.py` aggregates the results into
`benchmark_summary_by_seed.csv` and `benchmark_summary_multiseed_mean_sd.csv`.
You can narrow a run with the environment variables `EXPERIMENTS_FILTER` (regex,
e.g. `'^posafenet_big$'`), `SEEDS_OVERRIDE` (e.g. `"0"`), `N_EPISODES` and
`CKPT_NAME`.

Notes:
- The driving closed-loop evaluation is not bit-reproducible: repeated runs with
  the same checkpoint and scenario give slightly different actions, also with
  single-threaded CPU execution, most likely because of the rendering. Compare
  aggregated metrics over episodes and seeds.
- `models/posafenet_big_hard.py` scales the CBF class-K parameters by `3.1`
  (`3` in the other variants). This is the value used in our evaluation.

# Acknowledgement
The example pipelines are adapted from
[ABNet](https://github.com/Weixy21/ABNet); the driving setup and data come from
[BarrierNet](https://github.com/Weixy21/BarrierNet) and
[VISTA](https://github.com/vista-simulator/vista).

# License
This project is released under the [MIT License](LICENSE). It includes a
modified copy of VISTA and code adapted from BarrierNet, both MIT-licensed; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

# Reference
If you find this helpful, please cite our work:
```
@article{wong2026posafenet,
  title   = {PoSafeNet: Safe Learning with Poset-Structured Neural Nets},
  author  = {Wong, Kiwan and Xiao, Wei and Rus, Daniela},
  journal = {arXiv preprint arXiv:2601.22356},
  year    = {2026}
}
```
