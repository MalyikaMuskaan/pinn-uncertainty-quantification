"""
run_ensemble.py  —  ocean_pinn
--------------------------------
Main script for Deep Ensemble UQ on the advection-diffusion PINN.

Run with:
    python run_ensemble.py

Steps
-----
1. Train N_MEMBERS = 10 independent OceanPINN models (different seeds).
   Checkpoints saved to  outputs/ensemble/model_{i}.pt

2. Load all 10 models, run ensemble forward pass to compute
   pointwise mean and std of c(x, t).

3. Compute ECE and 90%-coverage probability vs FD reference.

4. Save four plots to outputs/ensemble/:
     ensemble_mean_heatmap.png
     ensemble_std_heatmap.png
     ensemble_time_slices.png
     ensemble_calibration.png
"""

import os
import time
import numpy as np
import torch

from train import train
from ensemble import load_ensemble, ensemble_predict, calibration_metrics
from ensemble_plot import (
    plot_ensemble_mean_heatmap,
    plot_ensemble_std_heatmap,
    plot_ensemble_time_slices,
    plot_calibration,
)
from data import make_evaluation_grid
from plot import fd_reference

CONFIG = {
    "n_members":   10,
    "seeds":       list(range(10)),
    "n_col":       10_000,
    "n_ic":        200,
    "n_bc":        200,
    "n_hidden":    4,
    "n_neurons":   50,
    "n_epochs":    5_000,
    "lr":          1e-3,
    "lambda_pde":  1.0,
    "lambda_ic":   10.0,
    "lambda_bc":   10.0,
    "print_every": 1000,
    "checkpoint_dir": "outputs/ensemble",
    "device":      "auto",
}


def resolve_device(cfg):
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def ckpt_path(cfg, i):
    return os.path.join(cfg["checkpoint_dir"], f"model_{i}.pt")


def train_ensemble(cfg, device):
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    n = cfg["n_members"]
    print("=" * 65)
    print(f"  ENSEMBLE TRAINING  ({n} members x {cfg['n_epochs']} epochs)")
    print("=" * 65)
    t0_total = time.time()

    for i in range(n):
        ckpt = ckpt_path(cfg, i)
        if os.path.isfile(ckpt):
            print(f"[ensemble] Member {i}: checkpoint exists, skipping.")
            continue

        seed = cfg["seeds"][i]
        print(f"\n[ensemble] ---- Member {i}  (seed={seed}) ----")
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()

        train(
            n_col=cfg["n_col"], n_ic=cfg["n_ic"], n_bc=cfg["n_bc"],
            n_hidden=cfg["n_hidden"], n_neurons=cfg["n_neurons"],
            n_epochs=cfg["n_epochs"], lr=cfg["lr"],
            lambda_pde=cfg["lambda_pde"],
            lambda_ic=cfg["lambda_ic"],
            lambda_bc=cfg["lambda_bc"],
            print_every=cfg["print_every"],
            save_path=ckpt,
            device_str=cfg["device"],
        )
        print(f"[ensemble] Member {i} done in {(time.time()-t0)/60:.1f} min.")

    total_elapsed = time.time() - t0_total
    print(f"\n[ensemble] All {n} members trained in {total_elapsed/60:.1f} min total.")


def evaluate_and_plot(cfg, device):
    models = load_ensemble(
        cfg["checkpoint_dir"], cfg["n_members"],
        cfg["n_hidden"], cfg["n_neurons"], device,
    )

    n_x, n_t = 256, 100
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(n_x, n_t, device)

    print("[ensemble] Running ensemble forward pass...")
    c_mean, c_std, c_all = ensemble_predict(
        models, x_flat, t_flat, grid_shape=(n_t, n_x)
    )
    print(f"[ensemble] mean range: [{c_mean.min():.3f}, {c_mean.max():.3f}]")
    print(f"[ensemble] std  range: [{c_std.min():.4e}, {c_std.max():.4e}]")

    metrics = calibration_metrics(c_mean, c_std, x_grid, t_grid)

    save_dir = cfg["checkpoint_dir"]
    plot_ensemble_mean_heatmap(
        x_grid, t_grid, c_mean,
        save_path=os.path.join(save_dir, "ensemble_mean_heatmap.png"),
    )
    plot_ensemble_std_heatmap(
        x_grid, t_grid, c_std,
        save_path=os.path.join(save_dir, "ensemble_std_heatmap.png"),
    )
    plot_ensemble_time_slices(
        x_grid, t_grid, c_mean, c_std,
        time_slices=[1.0, 2.5, 4.0],
        save_path=os.path.join(save_dir, "ensemble_time_slices.png"),
    )
    plot_calibration(
        metrics["confidence_levels"],
        metrics["empirical_coverage"],
        metrics["ece"],
        metrics["coverage_90"],
        save_path=os.path.join(save_dir, "ensemble_calibration.png"),
    )

    # MSE
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]
    c_ref  = fd_reference(x_vals, t_vals)
    mse    = float(np.mean((c_mean - c_ref) ** 2))

    print()
    print("=" * 65)
    print("  ENSEMBLE RESULTS SUMMARY")
    print("=" * 65)
    print(f"  MSE vs FD reference      : {mse:.4e}")
    print(f"  ECE                      : {metrics['ece']:.4f}")
    print(f"  90% coverage probability : {metrics['coverage_90']:.4f}")

    expected = [
        "ensemble_mean_heatmap.png",
        "ensemble_std_heatmap.png",
        "ensemble_time_slices.png",
        "ensemble_calibration.png",
    ]
    print()
    print("  SAVED FILES")
    print("  " + "-" * 50)
    for fname in expected:
        fpath  = os.path.join(save_dir, fname)
        exists = os.path.isfile(fpath)
        status = "OK" if exists else "MISSING"
        size_s = f"  ({os.path.getsize(fpath)/1024:.1f} KB)" if exists else ""
        print(f"  [{status}]  {fpath}{size_s}")
    print("=" * 65)


def main():
    cfg    = CONFIG
    device = resolve_device(cfg)
    print(f"[run_ensemble] Device: {device}")
    train_ensemble(cfg, device)
    evaluate_and_plot(cfg, device)
    print("\n[run_ensemble] Done. Outputs in outputs/ensemble/")


if __name__ == "__main__":
    main()
