"""
main.py
-------
Entry point for the Burgers' PINN project.

Run with:
    python main.py

All hyperparameters are centralised in the CONFIG dict at the top of this
file so they are easy to tweak for experimentation.

What this script does
---------------------
1. Trains the PINN on the 1D viscous Burgers' equation.
2. Saves the trained model weights to  outputs/burgers_pinn.pt
3. Generates three plots in outputs/:
     heatmap.png      — full (x, t) solution field
     time_slices.png  — PINN vs FD reference at t=0.25, 0.5, 0.75
     loss_history.png — training loss curves (log scale)
4. Prints a final summary of loss values and confirms saved files.
"""

import os
import torch
from train import train
from plot import plot_solution


# ================================================================== #
#  Hyperparameter configuration                                        #
#  Modify these values to experiment with the network or training.    #
# ================================================================== #
CONFIG = {
    # --- Network architecture ---
    "n_hidden":  4,       # number of hidden layers  (≥ 4 as required)
    "n_neurons": 50,      # neurons per hidden layer

    # --- Training data sizes ---
    "n_col": 10_000,      # collocation (PDE residual) points per epoch
    "n_ic":  200,         # initial condition points
    "n_bc":  200,         # boundary condition points (per wall → 400 total)

    # --- Optimisation ---
    "n_epochs":    15_000,   # total training epochs
    "lr":          1e-3,     # initial Adam learning rate
    "lambda_pde":  1.0,      # weight for PDE residual loss
    "lambda_ic":   10.0,     # weight for initial condition loss
    "lambda_bc":   10.0,     # weight for boundary condition loss

    # --- Logging & saving ---
    "print_every": 500,
    "save_path":   "outputs/burgers_pinn.pt",

    # --- Device: "auto" → uses GPU if available, else CPU ---
    "device": "auto",
}


def main():
    print("=" * 65)
    print("  Physics-Informed Neural Network -- 1D Burgers Equation")
    print("  nu = 0.01/pi,  x in [-1,1],  t in [0,1]")
    print("  u(x,0) = -sin(pi*x),   u(+-1,t) = 0")
    print("=" * 65)

    # ---------------------------------------------------------------- #
    #  Step 1 -- Train                                                  #
    # ---------------------------------------------------------------- #
    model, history_pde, history_ic, history_bc = train(
        n_col=CONFIG["n_col"],
        n_ic=CONFIG["n_ic"],
        n_bc=CONFIG["n_bc"],
        n_hidden=CONFIG["n_hidden"],
        n_neurons=CONFIG["n_neurons"],
        n_epochs=CONFIG["n_epochs"],
        lr=CONFIG["lr"],
        print_every=CONFIG["print_every"],
        save_path=CONFIG["save_path"],
        device_str=CONFIG["device"],
        lambda_pde=CONFIG["lambda_pde"],
        lambda_ic=CONFIG["lambda_ic"],
        lambda_bc=CONFIG["lambda_bc"],
    )

    # ---------------------------------------------------------------- #
    #  Step 2 -- Resolve device for plotting                            #
    # ---------------------------------------------------------------- #
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])

    # ---------------------------------------------------------------- #
    #  Step 3 -- Generate all plots                                     #
    # ---------------------------------------------------------------- #
    save_dir = "outputs/"
    plot_solution(
        model=model,
        device=device,
        history_pde=history_pde,
        history_ic=history_ic,
        history_bc=history_bc,
        save_dir=save_dir,
        print_every=CONFIG["print_every"],
    )

    # ---------------------------------------------------------------- #
    #  Step 4 -- Final loss summary                                     #
    # ---------------------------------------------------------------- #
    final_pde = history_pde[-1]
    final_ic  = history_ic[-1]
    final_bc  = history_bc[-1]
    final_tot = (CONFIG["lambda_pde"] * final_pde
                 + CONFIG["lambda_ic"] * final_ic
                 + CONFIG["lambda_bc"] * final_bc)

    print()
    print("=" * 65)
    print("  FINAL LOSS SUMMARY  (last recorded checkpoint)")
    print("=" * 65)
    print(f"  PDE residual loss  : {final_pde:.6e}")
    print(f"  Initial cond. loss : {final_ic:.6e}")
    print(f"  Boundary cond. loss: {final_bc:.6e}")
    print(f"  Weighted total     : {final_tot:.6e}")
    print(f"    (weights: PDE={CONFIG['lambda_pde']}, IC={CONFIG['lambda_ic']}, BC={CONFIG['lambda_bc']})")

    # ---------------------------------------------------------------- #
    #  Step 5 -- Confirm saved files                                    #
    # ---------------------------------------------------------------- #
    expected_files = {
        "outputs/burgers_pinn.pt":   "trained model weights",
        "outputs/heatmap.png":       "full (x,t) solution field",
        "outputs/time_slices.png":   "PINN vs FD reference at t=0.25, 0.50, 0.75",
        "outputs/loss_history.png":  "training loss curves (log scale)",
    }

    print()
    print("  SAVED FILES")
    print("  " + "-" * 50)
    all_present = True
    for path, description in expected_files.items():
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        size_str = ""
        if exists:
            size_kb = os.path.getsize(path) / 1024
            size_str = f"  ({size_kb:.1f} KB)"
        print(f"  [{status}]  {path}{size_str}")
        print(f"         {description}")
        if not exists:
            all_present = False

    print("=" * 65)
    if all_present:
        print("  All files confirmed present in outputs/")
    else:
        print("  WARNING: one or more expected files are missing.")
    print("=" * 65)


if __name__ == "__main__":
    main()
