#!/usr/bin/env python3
"""
Train one PoSafeNet model (Hard or Mixture) on the manipulator task for one random seed.

Examples
--------
python train_posafenet.py --model posafenet_hard    --seed 0 --skip-test
python train_posafenet.py --model posafenet_mixture --seed 5 --epochs 20 --skip-test

The experiment number defaults to the seed:
    seed 0 -> exp00
    seed 1 -> exp01
    ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from my_classes import Dataset_list as Dataset
import models

_THIS_DIR = Path(__file__).resolve().parent


MODEL_CHOICES = (
    "posafenet_hard",
    "posafenet_mixture",
)

MODEL_PARAM = [6, 128, 256, 128, 128, 32, 32, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Manipulator model for one random seed."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Model configuration to train.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--exp",
        type=int,
        default=None,
        help="Experiment number used in the output folder. Defaults to --seed.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rate-after-5", type=float, default=1e-4)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_THIS_DIR / "data_v2" / "data",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=_THIS_DIR / "log",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save a checkpoint every N epochs.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the final open-loop test-set evaluation.",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Allow nondeterministic CUDA algorithms for potentially higher speed.",
    )
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Some qpth/CUDA operations may not have deterministic implementations.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_data(data_dir: Path) -> Dict[str, np.ndarray]:
    required = {
        "train": data_dir / "dataM_train.mat",
        "valid": data_dir / "dataM_valid.mat",
        "test": data_dir / "dataM_test.mat",
    }
    for split, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} data: {path.resolve()}")

    train_data = sio.loadmat(required["train"])["data"]
    valid_data = sio.loadmat(required["valid"])["data"]
    test_data = sio.loadmat(required["test"])["data"]

    # Row format:
    # [ep_id, t, th1, w1, th2, w2, dst_x, dst_y, u1, u2]
    train_x_phys = np.asarray(train_data[:, 2:8], dtype=np.float64)
    valid_x_phys = np.asarray(valid_data[:, 2:8], dtype=np.float64)
    test_x_phys = np.asarray(test_data[:, 2:8], dtype=np.float64)

    train_y_phys = np.asarray(train_data[:, 8:10], dtype=np.float64)
    valid_y_phys = np.asarray(valid_data[:, 8:10], dtype=np.float64)
    test_y_phys = np.asarray(test_data[:, 8:10], dtype=np.float64)

    mean = np.mean(train_x_phys, axis=0)
    std = np.maximum(np.std(train_x_phys, axis=0), 1e-6)

    mean_label = np.mean(train_y_phys, axis=0)
    std_label = np.maximum(np.std(train_y_phys, axis=0), 1e-6)

    return {
        "train_x": (train_x_phys - mean) / std,
        "valid_x": (valid_x_phys - mean) / std,
        "test_x": (test_x_phys - mean) / std,
        "train_y": (train_y_phys - mean_label) / std_label,
        "valid_y": (valid_y_phys - mean_label) / std_label,
        "test_y": (test_y_phys - mean_label) / std_label,
        "test_y_phys": test_y_phys,
        "mean": mean,
        "std": std,
        "mean_label": mean_label,
        "std_label": std_label,
    }


def build_model(
    model_name: str,
    mean: np.ndarray,
    std: np.ndarray,
    mean_label: np.ndarray,
    std_label: np.ndarray,
    device: torch.device,
) -> nn.Module:
    common = (MODEL_PARAM, mean, std, mean_label, std_label, device)

    if model_name == "posafenet_hard":
        return models.PoSafeNet_Hard(
            *common,
            bn=False,
            heads=2,
        ).to(device)

    if model_name == "posafenet_mixture":
        return models.PoSafeNet_Mixture(
            *common,
            bn=False,
            heads=2,
        ).to(device)

    raise ValueError(f"Unknown model: {model_name}")


def checkpoint_path(
    log_root: Path,
    model_name: str,
    exp: int,
    epoch: int,
) -> Path:
    exp2 = f"{exp:02d}"
    ep2 = f"{epoch:02d}"

    mapping = {
        "posafenet_hard": (
            log_root
            / "posafenet_hard"
            / f"posafenet{exp2}"
            / f"model_posa{ep2}.pth"
        ),
        "posafenet_mixture": (
            log_root
            / "posafenet_mixture"
            / f"posafenet{exp2}"
            / f"model_posa{ep2}.pth"
        ),
    }
    return mapping[model_name]


def run_epoch(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (x, y) in enumerate(dataloader):
            x = x.to(device=device, dtype=torch.float64, non_blocking=True)
            y = y.to(device=device, dtype=torch.float64, non_blocking=True)

            # sgn=1 returns normalized control, matching normalized labels.
            pred = model(x, 1)
            loss = loss_fn(pred, y)

            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            if training and batch_index % 25 == 0:
                print(
                    f"  batch={batch_index:04d} "
                    f"loss={loss.item():.8f} "
                    f"samples={total_examples}/{len(dataloader.dataset)}",
                    flush=True,
                )

    if total_examples == 0:
        raise RuntimeError("The dataloader is empty.")

    return total_loss / total_examples


def evaluate_test_open_loop(
    model: nn.Module,
    test_x: np.ndarray,
    test_y_phys: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    """
    Open-loop test on dataset rows.

    Current models.py implementations are called with model.eval()
    and sgn=0, which returns physical controls.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(test_x), batch_size):
            end = min(start + batch_size, len(test_x))
            x = torch.from_numpy(test_x[start:end]).to(
                device=device,
                dtype=torch.float64,
            )
            u_phys = model(x, 0)
            predictions.append(u_phys.detach().cpu().numpy())

    pred_phys = np.concatenate(predictions, axis=0)
    error = pred_phys - test_y_phys

    return {
        "test_mse_physical": float(np.mean(error**2)),
        "test_mae_physical": float(np.mean(np.abs(error))),
        "mean_abs_prediction": float(np.mean(np.abs(pred_phys))),
        "mean_abs_label": float(np.mean(np.abs(test_y_phys))),
    }


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")

    exp = args.seed if args.exp is None else args.exp
    deterministic = not args.no_deterministic
    set_seed(args.seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print(f"Model:          {args.model}")
    print(f"Seed:           {args.seed}")
    print(f"Experiment:     {exp:02d}")
    print(f"Epochs:         {args.epochs}")
    print(f"Device:         {device}")
    print(f"Deterministic:  {deterministic}")
    print(f"Skip test:      {args.skip_test}")
    print(f"Data directory: {args.data_dir.resolve()}")
    print("=" * 72, flush=True)

    data = load_data(args.data_dir)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "worker_init_fn": seed_worker,
        "generator": train_generator,
        "pin_memory": device.type == "cuda",
    }
    valid_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "worker_init_fn": seed_worker,
        "pin_memory": device.type == "cuda",
    }

    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        valid_loader_kwargs["persistent_workers"] = True

    train_loader = torch.utils.data.DataLoader(
        Dataset(data["train_x"], data["train_y"]),
        shuffle=True,
        **train_loader_kwargs,
    )
    valid_loader = torch.utils.data.DataLoader(
        Dataset(data["valid_x"], data["valid_y"]),
        shuffle=False,
        **valid_loader_kwargs,
    )

    model = build_model(
        args.model,
        data["mean"],
        data["std"],
        data["mean_label"],
        data["std_label"],
        device,
    )
    print(model, flush=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    train_losses = []
    valid_losses = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_start = time.perf_counter()

    for epoch in range(args.epochs):
        if epoch == 5:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate_after_5,
            )

        epoch_start = time.perf_counter()

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}",
            flush=True,
        )

        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            device,
            optimizer=optimizer,
        )
        valid_loss = run_epoch(
            valid_loader,
            model,
            loss_fn,
            device,
            optimizer=None,
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_start

        print(
            f"Epoch summary: train={train_loss:.8f} "
            f"valid={valid_loss:.8f} "
            f"time={epoch_seconds:.3f}s",
            flush=True,
        )

        should_save = (
            (epoch + 1) % args.save_every == 0
            or epoch == args.epochs - 1
        )
        if should_save:
            ckpt = checkpoint_path(
                args.log_root,
                args.model,
                exp,
                epoch,
            )
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            print(f"[checkpoint] {ckpt}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - total_start

    final_ckpt = checkpoint_path(
        args.log_root,
        args.model,
        exp,
        args.epochs - 1,
    )
    run_dir = final_ckpt.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        run_dir / "training_history.npz",
        train_loss=np.asarray(train_losses, dtype=np.float64),
        valid_loss=np.asarray(valid_losses, dtype=np.float64),
    )

    if args.skip_test:
        print("[test] skipped", flush=True)
        test_metrics: Dict[str, float] = {}
    else:
        test_metrics = evaluate_test_open_loop(
            model,
            data["test_x"],
            data["test_y_phys"],
            device,
            batch_size=args.batch_size,
        )

    summary = {
        "model": args.model,
        "seed": args.seed,
        "exp": exp,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "deterministic": deterministic,
        "skip_test": args.skip_test,
        "total_train_valid_seconds": total_seconds,
        "average_seconds_per_epoch": total_seconds / args.epochs,
        "final_train_loss": train_losses[-1],
        "final_valid_loss": valid_losses[-1],
        "final_checkpoint": str(final_ckpt),
        **test_metrics,
    }

    with open(run_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining done.")
    print(f"Final checkpoint: {final_ckpt}")
    print(f"Total train+valid time: {total_seconds:.3f}s")
    print(f"Average per epoch: {total_seconds / args.epochs:.3f}s")

    if not args.skip_test:
        print(f"Physical test MSE: {test_metrics['test_mse_physical']:.8f}")

    print(f"History: {run_dir / 'training_history.npz'}")
    print(f"Summary: {run_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
