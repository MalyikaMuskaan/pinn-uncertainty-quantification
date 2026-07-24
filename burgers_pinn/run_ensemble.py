"""
run_ensemble.py
---------------
Main script for Deep Ensemble UQ on the Burgers' PINN.

Run with:
    python run_ensemble.py

What this script does
---------------------
1.  Trains N_MEMBERS = 10 independent BurgersPINN models.
    Each member uses a different random seed, which diversifies:
      - Xavier weight initialisation inside BurgersPINN.__init__
      - The collocation-point sampling sequence during training
    Models are saved to  outputs/ensemble/model_{i}.pt

2.  Loads all 10 checkpoints and runs the ensemble forward pass to compute
    pointwise mean and standard deviation (uncertainty) over the full (x,t)
    domain.

3.  Computes calibration metrics:
      - Expected Calibration Error (ECE)
      - 90% prediction interval coverage probability

4.  Generates four plots saved to outputs/ensemble/:
      ensemble_mean_heatmap.png      — mean prediction field
      ensemble_std_heatmap.png       — uncertainty (std) field
      ensemble_time_slices.png       — mean +/- 2*std band vs FD reference
      ensemble_calibration.png       — reliability diagram

Epoch count for ensemble members
---------------------------------
Training 10 models at 15 000 epochs each takes ~10x the single-model wall
time. On CPU with 10 000 collocation points that is roughly 10-15 hours.
We use 5 000 epochs per member here — empirically sufficient for the ensemble
spread to stabilise — while still giving good individual accuracy.
You can override this with the N_EPOCHS entry in CONFIG below.
"""

import os
import sys
import time
import torch
import numpy as np

from train import train
from ensemble import load_ensemble, ensemble_predict, calibration_metrics
from ensemble_plot import (
    plot_ensemble_mean_heatmap,
    plot_ensemble_std_heatmap,
    plot_ensemble_time_slices,
    plot_calibration,
)
from data import make_evaluation_grid


# ================================================================== #
#  Configuration                                                       #
# ================================================================== #
CONFIG = {
    # --- Ensemble ---
    "n_members":    10,          # number of independent models
    "seeds":        list(range(10)),   # seeds 0-9; extend for reproducibility

    # --- Per-member training (same as main.py but fewer epochs) ---
    "n_col":        10_000,
    "n_ic":         200,
    "n_bc":         200,
    "n_hidden":     4,
    "n_neurons":    50,
    "n_epochs":     5_000,       # reduced from 15 000 for feasibility
    "lr":           1e-3,
    "lambda_pde":   1.0,
    "lambda_ic":    10.0,
    "lambda_bc":    10.0,
    "print_every":  1000,        # less verbose per-member output

    # --- Paths ---
    "checkpoint_dir": "outputs/ensemble",
    "plot_dir":       "outputs/ensemble",

    # --- Device ---
    "device": "auto",
}


# ================================================================== #
#  Helpers                                                             #
# ================================================================== #

def resolve_device(cfg: dict) -> torch.device:
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def checkpoint_path(cfg: dict, i: int) -> str:
    return os.path.join(cfg["checkpoint_dir"], f"model_{i}.pt")


def all_checkpoints_exist(cfg: dict) -> bool:
    return all(
        os.path.isfile(checkpoint_path(cfg, i))
        for i in range(cfg["n_members"])
    )


# ================================================================== #
#  Step 1 — Train the ensemble                                         #
# ================================================================== #

def train_ensemble(cfg: dict, device: torch.device) -> None:
    """
    Train N_MEMBERS independent models, each with a unique random seed.

    The seed is set on both numpy and torch before calling train(), so
    that weight initialisation and collocation sampling differ across members.

    Already-saved checkpoints are skipped (allows resuming after interruption).
    """
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    n = cfg["n_members"]

    print("=" * 65)
    print(f"  ENSEMBLE TRAINING  ({n} members x {cfg['n_epochs']} epochs)")
    print("=" * 65)

    t0_total = time.time()

    for i in range(n):
        ckpt = checkpoint_path(cfg, i)
        if os.path.isfile(ckpt):
            print(f"\n[ensemble] Member {i}: checkpoint exists, skipping.")
            continue

        seed = cfg["seeds"][i]
        print(f"\n[ensemble] ---- Member {i}  (seed={seed}) ----")
        t0 = time.time()

        # Seed everything before building + training this member
        torch.manual_seed(seed)
        np.random.seed(seed)

        train(
            n_col=cfg["n_col"],
            n_ic=cfg["n_ic"],
            n_bc=cfg["n_bc"],
            n_hidden=cfg["n_hidden"],
            n_neurons=cfg["n_neurons"],
            n_epochs=cfg["n_epochs"],
            lr=cfg["lr"],
            print_every=cfg["print_every"],
            save_path=ckpt,
            device_str=cfg["device"],
            lambda_pde=cfg["lambda_pde"],
            lambda_ic=cfg["lambda_ic"],
            lambda_bc=cfg["lambda_bc"],
        )

        elapsed = time.time() - t0
        print(f"[ensemble] Member {i} done in {elapsed/60:.1f} min.")

    total_elapsed = time.time() - t0_total
    print(f"\n[ensemble] All {n} members trained in {total_elapsed/60:.1f} min total.")


# ================================================================== #
#  Step 2 — Evaluate ensemble & plot                                   #
# ================================================================== #

def evaluate_and_plot(cfg: dict, device: torch.device) -> None:
    """
    Load all checkpoints, run ensemble inference, compute calibration
    metrics, and save all four plots to the plot directory.
    """
    # -- Load models --
    models = load_ensemble(
        checkpoint_dir=cfg["checkpoint_dir"],
        n_members=cfg["n_members"],
        n_hidden=cfg["n_hidden"],
        n_neurons=cfg["n_neurons"],
        device=device,
    )

    # -- Build evaluation grid --
    n_x, n_t = 256, 100
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(n_x, n_t, device)

    # -- Forward pass: mean + std --
    print("[ensemble] Running ensemble forward pass...")
    u_mean, u_std, u_all = ensemble_predict(
        models, x_flat, t_flat, grid_shape=(n_t, n_x)
    )
    print(f"[ensemble] mean range: [{u_mean.min():.3f}, {u_mean.max():.3f}]")
    print(f"[ensemble] std  range: [{u_std.min():.4e}, {u_std.max():.4e}]")

    # -- Calibration metrics --
    metrics = calibration_metrics(u_mean, u_std, x_grid, t_grid)

    # -- Plots --
    save_dir = cfg["plot_dir"]
    os.makedirs(save_dir, exist_ok=True)

    plot_ensemble_mean_heatmap(
        x_grid, t_grid, u_mean,
        save_path=os.path.join(save_dir, "ensemble_mean_heatmap.png"),
    )
    plot_ensemble_std_heatmap(
        x_grid, t_grid, u_std,
        save_path=os.path.join(save_dir, "ensemble_std_heatmap.png"),
    )
    plot_ensemble_time_slices(
        x_grid, t_grid, u_mean, u_std,
        time_slices=[0.25, 0.50, 0.75],
        save_path=os.path.join(save_dir, "ensemble_time_slices.png"),
    )
    plot_calibration(
        metrics["confidence_levels"],
        metrics["empirical_coverage"],
        metrics["ece"],
        metrics["coverage_90"],
        save_path=os.path.join(save_dir, "ensemble_calibration.png"),
    )

    # -- Summary --
    print()
    print("=" * 65)
    print("  CALIBRATION SUMMARY")
    print("=" * 65)
    print(f"  Expected Calibration Error (ECE) : {metrics['ece']:.4f}")
    print(f"  90% coverage probability         : {metrics['coverage_90']:.4f}  "
          f"(ideal = 0.9000)")
    print()
    print("  SAVED FILES")
    print("  " + "-" * 50)
    expected = [
        "ensemble_mean_heatmap.png",
        "ensemble_std_heatmap.png",
        "ensemble_time_slices.png",
        "ensemble_calibration.png",
    ]
    for fname in expected:
        fpath = os.path.join(save_dir, fname)
        exists = os.path.isfile(fpath)
        status = "OK" if exists else "MISSING"
        size_str = f"  ({os.path.getsize(fpath)/1024:.1f} KB)" if exists else ""
        print(f"  [{status}]  {fpath}{size_str}")
    print("=" * 65)


# ================================================================== #
#  Entry point                                                         #
# ================================================================== #

def main():
    cfg    = CONFIG
    device = resolve_device(cfg)
    print(f"[run_ensemble] Device: {device}")

    train_ensemble(cfg, device)
    evaluate_and_plot(cfg, device)

    print("\n[run_ensemble] Done. All outputs in outputs/ensemble/")


if __name__ == "__main__":
    main()
