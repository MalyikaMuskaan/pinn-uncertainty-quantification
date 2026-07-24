"""
main.py  —  ocean_pinn
-----------------------
Entry point for the single-model advection-diffusion PINN.

Run with:
    python main.py

Steps
-----
1. Train OceanPINN for N_EPOCHS epochs.
2. Generate heatmap, time-slice comparison, and loss history plots.
3. Print final loss summary and confirm saved files.
"""

import os
import torch
from train import train
from plot import plot_solution

CONFIG = {
    "n_hidden":  4,
    "n_neurons": 50,
    "n_col":     10_000,
    "n_ic":      200,
    "n_bc":      200,
    "n_epochs":  15_000,
    "lr":        1e-3,
    "lambda_pde": 1.0,
    "lambda_ic":  10.0,
    "lambda_bc":  10.0,
    "print_every": 500,
    "save_path":  "outputs/ocean_pinn.pt",
    "device":     "auto",
}


def main():
    cfg = CONFIG
    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])

    print("=" * 65)
    print("  Ocean PINN -- 1D Advection-Diffusion Equation")
    print("  dc/dt + v*dc/dx = D*d2c/dx2   (v=1.0, D=0.05)")
    print("  x in [0,10] km,  t in [0,5]")
    print("  IC: Gaussian pulse at x=2")
    print("=" * 65)

    model, h_pde, h_ic, h_bc = train(
        n_col=cfg["n_col"], n_ic=cfg["n_ic"], n_bc=cfg["n_bc"],
        n_hidden=cfg["n_hidden"], n_neurons=cfg["n_neurons"],
        n_epochs=cfg["n_epochs"], lr=cfg["lr"],
        lambda_pde=cfg["lambda_pde"],
        lambda_ic=cfg["lambda_ic"],
        lambda_bc=cfg["lambda_bc"],
        print_every=cfg["print_every"],
        save_path=cfg["save_path"],
        device_str=cfg["device"],
    )

    plot_solution(
        model=model, device=device,
        history_pde=h_pde, history_ic=h_ic, history_bc=h_bc,
        save_dir="outputs/",
        print_every=cfg["print_every"],
    )

    final_pde = h_pde[-1]
    final_ic  = h_ic[-1]
    final_bc  = h_bc[-1]
    final_tot = (cfg["lambda_pde"] * final_pde
                 + cfg["lambda_ic"] * final_ic
                 + cfg["lambda_bc"] * final_bc)

    print()
    print("=" * 65)
    print("  FINAL LOSS SUMMARY")
    print("=" * 65)
    print(f"  PDE residual loss  : {final_pde:.6e}")
    print(f"  Initial cond. loss : {final_ic:.6e}")
    print(f"  Boundary cond. loss: {final_bc:.6e}")
    print(f"  Weighted total     : {final_tot:.6e}")

    expected = {
        "outputs/ocean_pinn.pt":    "model weights",
        "outputs/heatmap.png":      "full c(x,t) field",
        "outputs/time_slices.png":  "PINN vs FD reference",
        "outputs/loss_history.png": "training loss curves",
    }
    print()
    print("  SAVED FILES")
    print("  " + "-" * 50)
    for path, desc in expected.items():
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        size_s = f"  ({os.path.getsize(path)/1024:.1f} KB)" if exists else ""
        print(f"  [{status}]  {path}{size_s}")
        print(f"         {desc}")
    print("=" * 65)


if __name__ == "__main__":
    main()
