"""
main.py
-------
Entry point for the 2-D Darcy flow PINN experiment.

What it does
------------
1. Trains the DarcyPINN using the Adam → L-BFGS two-phase schedule
   (via train.train).
2. Generates three figures:
     outputs/solution_comparison.png  — PINN vs exact vs error heatmap
     outputs/loss_history.png         — training loss curves
     outputs/pde_residual_map.png     — PDE residual diagnostic
3. Computes and prints final accuracy metrics (MSE, Rel-L2 error).

Usage
-----
    python main.py                          # default settings
    python main.py --n_epochs 5000          # custom epoch count
    python main.py --device cpu             # force CPU (for testing)
    python main.py --skip_residual_plot     # skip the autograd-heavy residual map

All outputs are written to the outputs/ subdirectory.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from train import train
from plot  import plot_solution, plot_loss_history, plot_pde_residual


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main(
    # Training hyperparameters (can also be overridden via CLI)
    n_hidden:           int   = 5,
    n_neurons:          int   = 64,
    n_col:              int   = 10_000,
    n_per_edge:         int   = 200,
    n_epochs:           int   = 5_000,
    adam_epochs:        int   = 3_000,
    lr:                 float = 1e-3,
    lr_lbfgs:           float = 1.0,
    print_every:        int   = 500,
    lambda_pde:         float = 1.0,
    lambda_bc:          float = 10.0,
    device_str:         str   = "auto",
    save_path:          str   = "outputs/darcy_pinn.pt",
    out_dir:            str   = "outputs",
    skip_residual_plot: bool  = False,
) -> dict:
    """
    Run the full Darcy PINN experiment.

    Returns the training result dict (same as train.train returns) plus
    added 'mse' and 'rel_l2' keys from the final evaluation.
    """

    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------- #
    #  Device                                                            #
    # ---------------------------------------------------------------- #
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[main] Device: {device}")

    # ---------------------------------------------------------------- #
    #  Train                                                             #
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 65)
    print("  Darcy 2-D PINN  —  Training")
    print("=" * 65)

    results = train(
        n_hidden      = n_hidden,
        n_neurons     = n_neurons,
        n_col         = n_col,
        n_per_edge    = n_per_edge,
        n_epochs      = n_epochs,
        adam_epochs   = adam_epochs,
        lr            = lr,
        lr_lbfgs      = lr_lbfgs,
        print_every   = print_every,
        lambda_pde    = lambda_pde,
        lambda_bc     = lambda_bc,
        save_path     = save_path,
        device_str    = device_str,
        verbose       = True,
    )

    model      = results["model"]
    train_time = results["train_time"]

    print(f"\n[main] Training time: {train_time:.1f}s  ({train_time/60:.1f} min)")

    # ---------------------------------------------------------------- #
    #  Final evaluation — MSE and Relative L2 vs exact solution         #
    # ---------------------------------------------------------------- #
    from data import make_eval_grid, exact_solution_np

    model.eval()
    x_flat, y_flat, X, Y = make_eval_grid(n=256, device=device)

    with torch.no_grad():
        u_pred = model(x_flat, y_flat).cpu().numpy().reshape(256, 256)

    u_exact = exact_solution_np(X, Y)
    mse     = float(np.mean((u_pred - u_exact) ** 2))
    rel_l2  = float(np.linalg.norm(u_pred - u_exact) /
                    (np.linalg.norm(u_exact) + 1e-12))

    print("\n" + "=" * 65)
    print(f"  Final Metrics (256×256 eval grid vs analytical solution)")
    print(f"    MSE         : {mse:.4e}")
    print(f"    Rel-L2 err  : {rel_l2:.4e}  ({rel_l2:.2%})")
    print("=" * 65)

    # ---------------------------------------------------------------- #
    #  Save metrics JSON                                                 #
    # ---------------------------------------------------------------- #
    metrics = {
        "mse":        mse,
        "rel_l2":     rel_l2,
        "train_time": train_time,
        "n_epochs":   n_epochs,
        "adam_epochs": adam_epochs,
        "n_col":      n_col,
        "n_per_edge": n_per_edge,
        "n_hidden":   n_hidden,
        "n_neurons":  n_neurons,
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[main] Metrics saved to '{metrics_path}'")

    # ---------------------------------------------------------------- #
    #  Plots                                                             #
    # ---------------------------------------------------------------- #
    print("\n[main] Generating plots...")

    plot_solution(model, device, n_grid=256, save_dir=out_dir)
    plot_loss_history(
        results["loss_history"],
        save_dir    = out_dir,
        adam_epochs = adam_epochs,
        print_every = print_every,
    )

    if not skip_residual_plot:
        plot_pde_residual(model, device, n_grid=128, save_dir=out_dir)

    print("\n[main] Done.")
    results["mse"]    = mse
    results["rel_l2"] = rel_l2
    return results


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the 2-D Darcy flow PINN"
    )
    parser.add_argument("--n_hidden",           type=int,   default=5)
    parser.add_argument("--n_neurons",          type=int,   default=64)
    parser.add_argument("--n_col",              type=int,   default=10_000)
    parser.add_argument("--n_per_edge",         type=int,   default=200)
    parser.add_argument("--n_epochs",           type=int,   default=5_000)
    parser.add_argument("--adam_epochs",        type=int,   default=3_000)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--lr_lbfgs",           type=float, default=1.0)
    parser.add_argument("--print_every",        type=int,   default=500)
    parser.add_argument("--lambda_pde",         type=float, default=1.0)
    parser.add_argument("--lambda_bc",          type=float, default=10.0)
    parser.add_argument("--device",             type=str,   default="auto",
                        dest="device_str")
    parser.add_argument("--out_dir",            type=str,   default="outputs")
    parser.add_argument("--skip_residual_plot", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
