"""
run_dropout.py
--------------
Entry point for the MC Dropout PINN UQ pipeline.

Run with:
    python run_dropout.py

Steps
-----
1. Train a DropoutBurgersPINN (p=0.05, 10 000 epochs).
   Checkpoint saved to  outputs/dropout/dropout_pinn.pt

2. Load the checkpoint, enable MC Dropout (model.train() mode),
   and draw N_MC=100 stochastic forward passes on a dense grid.

3. Compute calibration metrics (ECE, 90% coverage) vs FD reference.

4. Save four plots to outputs/dropout/:
     dropout_mean_heatmap.png
     dropout_std_heatmap.png
     dropout_time_slices.png
     dropout_calibration.png
     dropout_loss_history.png

5. Save a metrics JSON for compare_methods.py.

Note on expected uncertainty magnitude
----------------------------------------
With p=0.05 and 50 neurons per layer, each forward pass drops ~2-3 neurons.
The resulting stochastic variation is genuinely small — std is typically
O(1e-3) to O(1e-2), i.e. 5-50x smaller than the Deep Ensemble's std.
This is faithfully reported and discussed in COMPARISON.md.
"""

import os
import json
import time
import math
import torch
import numpy as np

from dropout_train import train_dropout
from dropout_predict import load_dropout_model, evaluate_dropout
from dropout_plot import (
    plot_dropout_mean_heatmap,
    plot_dropout_std_heatmap,
    plot_dropout_time_slices,
    plot_dropout_calibration,
    plot_dropout_loss_history,
)
from plot import _fd_reference

NU = 0.01 / math.pi

CONFIG = {
    "n_hidden":     4,
    "n_neurons":    50,
    "dropout_rate": 0.05,
    "n_col":        10_000,
    "n_ic":         200,
    "n_bc":         200,
    "n_epochs":     10_000,
    "lr":           1e-3,
    "lambda_pde":   1.0,
    "lambda_ic":    10.0,
    "lambda_bc":    10.0,
    "print_every":  500,
    "n_mc_eval":    100,
    "checkpoint":   "outputs/dropout/dropout_pinn.pt",
    "output_dir":   "outputs/dropout",
    "device":       "auto",
}


def resolve_device(cfg):
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def main():
    cfg    = CONFIG
    device = resolve_device(cfg)
    print(f"[run_dropout] Device: {device}")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---------------------------------------------------------------- #
    #  Step 1 — Train                                                   #
    # ---------------------------------------------------------------- #
    ckpt = cfg["checkpoint"]
    if os.path.isfile(ckpt):
        print(f"[run_dropout] Checkpoint exists at '{ckpt}', skipping training.")
        train_time = 0.0
        h_pde = h_ic = h_bc = []
    else:
        model, h_pde, h_ic, h_bc, train_time = train_dropout(
            n_col=cfg["n_col"], n_ic=cfg["n_ic"], n_bc=cfg["n_bc"],
            n_hidden=cfg["n_hidden"], n_neurons=cfg["n_neurons"],
            dropout_rate=cfg["dropout_rate"],
            n_epochs=cfg["n_epochs"], lr=cfg["lr"],
            lambda_pde=cfg["lambda_pde"],
            lambda_ic=cfg["lambda_ic"],
            lambda_bc=cfg["lambda_bc"],
            print_every=cfg["print_every"],
            save_path=ckpt,
            device_str=cfg["device"],
        )
        plot_dropout_loss_history(
            h_pde, h_ic, h_bc,
            print_every=cfg["print_every"],
            save_path=os.path.join(cfg["output_dir"], "dropout_loss_history.png"),
        )

    # ---------------------------------------------------------------- #
    #  Step 2 — MC Dropout evaluation                                   #
    # ---------------------------------------------------------------- #
    model = load_dropout_model(
        ckpt,
        n_hidden=cfg["n_hidden"],
        n_neurons=cfg["n_neurons"],
        dropout_rate=cfg["dropout_rate"],
        device=device,
    )

    t_inf_start = time.time()
    x_grid, t_grid, u_mean, u_std, u_all, metrics = evaluate_dropout(
        model, device,
        n_x=256, n_t=100,
        n_samples=cfg["n_mc_eval"],
    )
    inference_time = time.time() - t_inf_start
    print(f"[run_dropout] Inference ({cfg['n_mc_eval']} passes): "
          f"{inference_time:.1f}s")

    # ---------------------------------------------------------------- #
    #  Step 3 — Plots                                                   #
    # ---------------------------------------------------------------- #
    d = cfg["output_dir"]
    plot_dropout_mean_heatmap(
        x_grid, t_grid, u_mean,
        save_path=os.path.join(d, "dropout_mean_heatmap.png"),
    )
    plot_dropout_std_heatmap(
        x_grid, t_grid, u_std,
        save_path=os.path.join(d, "dropout_std_heatmap.png"),
    )
    plot_dropout_time_slices(
        x_grid, t_grid, u_mean, u_std,
        time_slices=[0.25, 0.50, 0.75],
        save_path=os.path.join(d, "dropout_time_slices.png"),
    )
    plot_dropout_calibration(
        metrics["confidence_levels"],
        metrics["empirical_coverage"],
        metrics["ece"],
        metrics["coverage_90"],
        save_path=os.path.join(d, "dropout_calibration.png"),
    )

    # ---------------------------------------------------------------- #
    #  Step 4 — Persist metrics                                         #
    # ---------------------------------------------------------------- #
    from data import X_MIN, X_MAX
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]
    u_ref  = _fd_reference(x_vals, t_vals, NU)
    mse    = float(np.mean((u_mean - u_ref) ** 2))

    metrics_out = {
        "method":           "MC Dropout",
        "mse":              mse,
        "ece":              metrics["ece"],
        "coverage_90":      metrics["coverage_90"],
        "train_time_s":     train_time,
        "inference_time_s": inference_time,
        "n_epochs":         cfg["n_epochs"],
        "n_mc_eval":        cfg["n_mc_eval"],
        "dropout_rate":     cfg["dropout_rate"],
        "std_max":          metrics["std_max"],
        "std_mean":         metrics["std_mean"],
    }
    mpath = os.path.join(d, "dropout_metrics.json")
    with open(mpath, "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    print(f"[run_dropout] Metrics saved to '{mpath}'")

    # ---------------------------------------------------------------- #
    #  Step 5 — Summary                                                 #
    # ---------------------------------------------------------------- #
    print()
    print("=" * 65)
    print("  MC DROPOUT PINN — RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Dropout rate                 : p={cfg['dropout_rate']}")
    print(f"  MC passes at inference       : {cfg['n_mc_eval']}")
    print(f"  MSE vs FD reference          : {mse:.4e}")
    print(f"  ECE                          : {metrics['ece']:.4f}")
    print(f"  90% coverage probability     : {metrics['coverage_90']:.4f}")
    print(f"  Std max / mean               : "
          f"{metrics['std_max']:.3e} / {metrics['std_mean']:.3e}")
    if metrics["std_max"] < 5e-3:
        print(f"  ** Near-zero uncertainty detected — expected for p="
              f"{cfg['dropout_rate']} with {cfg['n_neurons']}-neuron layers.")
        print(f"     See COMPARISON.md for interpretation.")
    print(f"  Training time                : {train_time/60:.1f} min")
    print(f"  Inference time               : {inference_time:.1f}s")

    expected = [
        "dropout_mean_heatmap.png", "dropout_std_heatmap.png",
        "dropout_time_slices.png",  "dropout_calibration.png",
        "dropout_loss_history.png", "dropout_metrics.json",
    ]
    print()
    print("  SAVED FILES")
    print("  " + "-" * 50)
    for fname in expected:
        fpath  = os.path.join(d, fname)
        exists = os.path.isfile(fpath)
        status = "OK" if exists else "MISSING"
        size_s = f"  ({os.path.getsize(fpath)/1024:.1f} KB)" if exists else ""
        print(f"  [{status}]  {fpath}{size_s}")
    print("=" * 65)


if __name__ == "__main__":
    main()
