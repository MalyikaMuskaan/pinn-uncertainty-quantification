"""
run_bayesian.py
---------------
Entry point for the Bayesian PINN UQ pipeline.

Run with:
    python run_bayesian.py

Steps
-----
1. Train a BayesianBurgersPINN with mean-field VI for N_EPOCHS epochs.
   Checkpoint saved to  outputs/bayesian/bayesian_pinn.pt

2. Load the checkpoint and draw N_MC_SAMPLES Monte Carlo weight samples to
   estimate the posterior predictive mean and uncertainty.

3. Compute calibration metrics (ECE, 90% coverage) against the FD reference.

4. Save four plots to outputs/bayesian/:
     bayesian_mean_heatmap.png
     bayesian_std_heatmap.png
     bayesian_time_slices.png
     bayesian_calibration.png
     bayesian_loss_history.png

5. Save a metrics JSON file for later comparison by compare_methods.py.
"""

import os
import json
import time
import torch
import numpy as np

import math
from bayesian_train import train_bayesian
from bayesian_predict import load_bayesian_model, evaluate_bayesian

NU = 0.01 / math.pi
from bayesian_plot import (
    plot_bayesian_mean_heatmap,
    plot_bayesian_std_heatmap,
    plot_bayesian_time_slices,
    plot_bayesian_calibration,
    plot_bayesian_loss_history,
)


# ================================================================== #
#  Configuration                                                       #
# ================================================================== #
CONFIG = {
    # --- Architecture (same as main.py for fair comparison) ---
    "n_hidden":  4,
    "n_neurons": 50,

    # --- Training ---
    "n_col":        10_000,
    "n_ic":         200,
    "n_bc":         200,
    "n_epochs":     5_000,       # same as ensemble members for fairness
    "lr":           1e-3,
    "kl_weight":    None,        # None → auto: 1/n_col
    "n_mc_samples": 1,           # MC samples per epoch during training
    "lambda_pde":   1.0,
    "lambda_ic":    10.0,
    "lambda_bc":    10.0,
    "print_every":  500,

    # --- Inference ---
    "n_mc_eval":    200,         # MC samples at evaluation time

    # --- Paths ---
    "checkpoint": "outputs/bayesian/bayesian_pinn.pt",
    "output_dir": "outputs/bayesian",

    # --- Device ---
    "device": "auto",
}


def resolve_device(cfg: dict) -> torch.device:
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def main():
    cfg    = CONFIG
    device = resolve_device(cfg)
    print(f"[run_bayesian] Device: {device}")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---------------------------------------------------------------- #
    #  Step 1 — Train                                                   #
    # ---------------------------------------------------------------- #
    ckpt = cfg["checkpoint"]
    if os.path.isfile(ckpt):
        print(f"[run_bayesian] Checkpoint already exists at '{ckpt}', "
              f"skipping training.")
        # Still need to measure a rough train_time from the file mtime
        train_time = 0.0
    else:
        model, h_pde, h_ic, h_bc, h_kl, train_time = train_bayesian(
            n_col=cfg["n_col"],
            n_ic=cfg["n_ic"],
            n_bc=cfg["n_bc"],
            n_hidden=cfg["n_hidden"],
            n_neurons=cfg["n_neurons"],
            n_epochs=cfg["n_epochs"],
            lr=cfg["lr"],
            kl_weight=cfg["kl_weight"],
            n_mc_samples=cfg["n_mc_samples"],
            lambda_pde=cfg["lambda_pde"],
            lambda_ic=cfg["lambda_ic"],
            lambda_bc=cfg["lambda_bc"],
            print_every=cfg["print_every"],
            save_path=ckpt,
            device_str=cfg["device"],
        )
        # Save loss histories for the plot
        plot_bayesian_loss_history(
            h_pde, h_ic, h_bc, h_kl,
            print_every=cfg["print_every"],
            save_path=os.path.join(cfg["output_dir"], "bayesian_loss_history.png"),
        )

    # ---------------------------------------------------------------- #
    #  Step 2 — Load + MC evaluation                                    #
    # ---------------------------------------------------------------- #
    model = load_bayesian_model(ckpt, cfg["n_hidden"], cfg["n_neurons"], device)

    t_inf_start = time.time()
    x_grid, t_grid, u_mean, u_std, u_all, metrics = evaluate_bayesian(
        model, device,
        n_x=256, n_t=100,
        n_samples=cfg["n_mc_eval"],
    )
    inference_time = time.time() - t_inf_start
    print(f"[run_bayesian] Inference time ({cfg['n_mc_eval']} samples): "
          f"{inference_time:.1f}s")

    # ---------------------------------------------------------------- #
    #  Step 3 — Plots                                                   #
    # ---------------------------------------------------------------- #
    d = cfg["output_dir"]
    plot_bayesian_mean_heatmap(
        x_grid, t_grid, u_mean,
        save_path=os.path.join(d, "bayesian_mean_heatmap.png"),
    )
    plot_bayesian_std_heatmap(
        x_grid, t_grid, u_std,
        save_path=os.path.join(d, "bayesian_std_heatmap.png"),
    )
    plot_bayesian_time_slices(
        x_grid, t_grid, u_mean, u_std,
        time_slices=[0.25, 0.50, 0.75],
        save_path=os.path.join(d, "bayesian_time_slices.png"),
    )
    plot_bayesian_calibration(
        metrics["confidence_levels"],
        metrics["empirical_coverage"],
        metrics["ece"],
        metrics["coverage_90"],
        save_path=os.path.join(d, "bayesian_calibration.png"),
    )

    # ---------------------------------------------------------------- #
    #  Step 4 — Persist metrics for compare_methods.py                 #
    # ---------------------------------------------------------------- #
    # Compute MSE against FD reference
    from plot import _fd_reference
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]
    u_ref  = _fd_reference(x_vals, t_vals, NU)
    mse    = float(np.mean((u_mean - u_ref) ** 2))

    metrics_out = {
        "method":         "Bayesian PINN (VI)",
        "mse":            mse,
        "ece":            metrics["ece"],
        "coverage_90":    metrics["coverage_90"],
        "train_time_s":   train_time,
        "inference_time_s": inference_time,
        "n_epochs":       cfg["n_epochs"],
        "n_mc_eval":      cfg["n_mc_eval"],
    }

    metrics_path = os.path.join(d, "bayesian_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    print(f"[run_bayesian] Metrics saved to '{metrics_path}'")

    # ---------------------------------------------------------------- #
    #  Step 5 — Summary                                                 #
    # ---------------------------------------------------------------- #
    print()
    print("=" * 65)
    print("  BAYESIAN PINN — RESULTS SUMMARY")
    print("=" * 65)
    print(f"  MSE vs FD reference          : {mse:.4e}")
    print(f"  ECE                          : {metrics['ece']:.4f}")
    print(f"  90% coverage probability     : {metrics['coverage_90']:.4f}")
    print(f"  Training time                : {train_time/60:.1f} min")
    print(f"  Inference time ({cfg['n_mc_eval']} samples) : {inference_time:.1f}s")

    expected = [
        "bayesian_mean_heatmap.png",
        "bayesian_std_heatmap.png",
        "bayesian_time_slices.png",
        "bayesian_calibration.png",
        "bayesian_loss_history.png",
        "bayesian_metrics.json",
    ]
    print()
    print("  SAVED FILES")
    print("  " + "-" * 50)
    all_ok = True
    for fname in expected:
        fpath  = os.path.join(d, fname)
        exists = os.path.isfile(fpath)
        status = "OK" if exists else "MISSING"
        size_s = f"  ({os.path.getsize(fpath)/1024:.1f} KB)" if exists else ""
        print(f"  [{status}]  {fpath}{size_s}")
        if not exists:
            all_ok = False
    print("=" * 65)
    if all_ok:
        print("  All Bayesian outputs confirmed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
