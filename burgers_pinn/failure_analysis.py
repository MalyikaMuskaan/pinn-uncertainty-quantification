"""
failure_analysis.py
-------------------
Phase 6A — Failure Case Analysis

Demonstrates where the standard Burgers' PINN breaks down by sweeping the
kinematic viscosity ν from the baseline value down to progressively smaller
values.  Smaller ν → sharper shock → harder problem for PINNs.

Viscosity sweep
---------------
ν = 0.01/π  (baseline, ~3.18e-3)     — established working case
ν = 0.005/π (~1.59e-3)               — 2× sharper shock
ν = 0.002/π (~6.37e-4)               — 5× sharper shock
ν = 0.001/π (~3.18e-4)               — 10× sharper shock

For each ν the script:
  1. Trains a fresh vanilla PINN (same 4×50 tanh architecture and training
     schedule as the baseline, only ν is changed).
  2. Evaluates relative L2 error against the Crank-Nicolson FD reference.
  3. Saves the trained checkpoint and per-ν metrics.

After the sweep, two summary figures are generated:
  failure_error_vs_nu.png
      Rel-L2 error vs ν on a log-log scale.  The upward trend toward low ν
      quantifies PINN accuracy degradation as the shock sharpens.

  failure_heatmap_comparison.png
      Side-by-side heatmaps (PINN prediction / FD reference / pointwise error)
      at the hardest (smallest) ν, visually showing where the PINN fails.

Why PINNs fail at low ν
-----------------------
See validation_notes/NOTES.md §1 for the full explanation.  The short version:
the shock becomes a near-discontinuity with spatial gradient ~1/ν; the PINN
must represent this via smooth tanh-activated functions, which are biased
towards low-frequency components (spectral bias).  The PDE residual near the
shock also becomes stiff, with gradients that are difficult for first-order
optimisers to follow without extremely dense collocation near the shock front.

Outputs (all under outputs/failure_analysis/)
---------------------------------------------
  model_nu_{i}.pt                  trained checkpoint for each ν
  metrics.json                     all per-ν metrics + summary
  failure_error_vs_nu.png          log-log error vs ν plot
  failure_heatmap_comparison.png   worst-case PINN vs FD heatmaps

Usage
-----
    cd burgers_pinn
    python failure_analysis.py

    # Colab one-liner:
    # %cd burgers_pinn && python failure_analysis.py
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---- Local imports from burgers_pinn/ --------------------------------
from model import BurgersPINN
from data  import (
    sample_collocation_points,
    sample_initial_condition_points,
    sample_boundary_condition_points,
    make_evaluation_grid,
    X_MIN, X_MAX, T_MIN, T_MAX,
)
from plot import _fd_reference    # Crank-Nicolson solver — accepts any nu


# ================================================================== #
#  Configuration                                                       #
# ================================================================== #

# Viscosities to sweep (decreasing → sharper shock)
NU_VALUES = [
    0.01  / np.pi,   # baseline
    0.005 / np.pi,   # 2× sharper
    0.002 / np.pi,   # 5× sharper
    0.001 / np.pi,   # 10× sharper — expected failure
]

CONFIG = {
    # Architecture (identical to baseline)
    "n_hidden":   4,
    "n_neurons":  50,
    # Training schedule — same as run_ensemble.py members
    "n_col":      10_000,
    "n_ic":       200,
    "n_bc":       200,
    "n_epochs":   5_000,
    "lr":         1e-3,
    "lambda_pde": 1.0,
    "lambda_ic":  10.0,
    "lambda_bc":  10.0,
    "print_every": 1_000,
    # Evaluation grid
    "n_x": 256,
    "n_t": 100,
    # Output
    "out_dir": "outputs/failure_analysis",
    "device":  "auto",
}


# ================================================================== #
#  Per-ν training with nu passed as a parameter                        #
# ================================================================== #

def train_for_nu(
    nu:        float,
    cfg:       dict,
    device:    torch.device,
    save_path: str,
    seed:      int = 0,
) -> tuple[BurgersPINN, list, list, list]:
    """
    Train a vanilla BurgersPINN for a specific viscosity nu.

    The training loop is reproduced here (rather than calling train.train)
    because train.py uses a module-level NU constant that cannot be overridden
    at call time.  The logic is identical to train.train; only the PDE
    residual uses the local `nu` parameter.

    Returns
    -------
    model      : trained BurgersPINN
    hist_pde   : list of PDE loss values (every print_every epochs)
    hist_ic    : list of IC  loss values
    hist_bc    : list of BC  loss values
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BurgersPINN(
        n_hidden=cfg["n_hidden"], n_neurons=cfg["n_neurons"]
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2000
    )

    # Fixed IC / BC data
    x_ic, t_ic, u_ic = sample_initial_condition_points(cfg["n_ic"], device)
    x_bc, t_bc, u_bc = sample_boundary_condition_points(cfg["n_bc"], device)

    lp, li, lb = cfg["lambda_pde"], cfg["lambda_ic"], cfg["lambda_bc"]
    hist_pde, hist_ic, hist_bc = [], [], []

    for epoch in range(1, cfg["n_epochs"] + 1):
        model.train()
        x_col, t_col = sample_collocation_points(cfg["n_col"], device)

        # -- PDE residual (inline, with local nu) --
        u      = model(x_col, t_col)
        ones   = torch.ones_like(u)
        u_x    = torch.autograd.grad(u, x_col, grad_outputs=ones,
                                     create_graph=True, retain_graph=True)[0]
        u_t    = torch.autograd.grad(u, t_col, grad_outputs=ones,
                                     create_graph=True, retain_graph=True)[0]
        u_xx   = torch.autograd.grad(u_x, x_col,
                                     grad_outputs=torch.ones_like(u_x),
                                     create_graph=True, retain_graph=True)[0]
        f      = u_t + u * u_x - nu * u_xx
        l_pde  = torch.mean(f ** 2)

        # -- IC / BC losses --
        l_ic   = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
        l_bc   = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)

        total  = lp * l_pde + li * l_ic + lb * l_bc

        optimiser.zero_grad()
        total.backward()
        optimiser.step()
        scheduler.step(total.detach())

        if epoch % cfg["print_every"] == 0 or epoch == 1:
            hist_pde.append(l_pde.item())
            hist_ic.append(l_ic.item())
            hist_bc.append(l_bc.item())
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"  ep {epoch:>6d}  pde={l_pde.item():.3e}  "
                  f"ic={l_ic.item():.3e}  bc={l_bc.item():.3e}  "
                  f"lr={lr_now:.1e}  nu={nu:.4e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return model, hist_pde, hist_ic, hist_bc


# ================================================================== #
#  Relative L2 error vs FD reference                                   #
# ================================================================== #

def compute_rel_l2(
    model:  BurgersPINN,
    nu:     float,
    device: torch.device,
    n_x:    int = 256,
    n_t:    int = 100,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the PINN and compute relative L2 error vs Crank-Nicolson.

    Returns
    -------
    rel_l2   : scalar relative L2 error
    x_grid   : (n_t, n_x) meshgrid x
    t_grid   : (n_t, n_x) meshgrid t
    u_pred   : (n_t, n_x) PINN prediction
    u_ref    : (n_t, n_x) FD reference
    """
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(n_x, n_t, device)

    model.eval()
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(n_t, n_x)

    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]
    u_ref  = _fd_reference(x_vals, t_vals, nu)

    rel_l2 = float(
        np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    )
    return rel_l2, x_grid, t_grid, u_pred, u_ref


# ================================================================== #
#  Plots                                                               #
# ================================================================== #

def plot_error_vs_nu(
    nu_values: list,
    rel_l2s:   list,
    out_dir:   str,
) -> None:
    """
    Log-log plot of relative L2 error vs viscosity ν.

    A negative slope (error rising as ν decreases) directly demonstrates
    PINN accuracy degradation as the shock sharpens.
    """
    nus  = np.array(nu_values)
    errs = np.array(rel_l2s)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(nus, errs,
              color="#3b82d4", linewidth=2.2,
              marker="o", markersize=9, markerfacecolor="white",
              markeredgewidth=2.0, label="Rel-L2 error")

    # Annotate each point with its ν label
    for nu, err in zip(nus, errs):
        ax.annotate(
            f"ν={nu:.2e}\n{err:.1%}",
            xy=(nu, err),
            xytext=(8, 4), textcoords="offset points",
            fontsize=8, color="#3a3f47",
        )

    # Reference slope lines for visual comparison
    nu_ref = np.array([nus.min(), nus.max()])
    mid_err = np.exp(np.log(errs).mean())
    mid_nu  = np.exp(np.log(nus).mean())
    for slope, ls, label in [(-1, "--", "slope −1"), (-2, ":", "slope −2")]:
        ax.loglog(nu_ref,
                  mid_err * (nu_ref / mid_nu) ** slope,
                  color="gray", lw=1.0, ls=ls, label=label)

    ax.set_xlabel("Kinematic viscosity  ν  (log scale)", fontsize=12)
    ax.set_ylabel("Relative L2 error  (log scale)", fontsize=12)
    ax.set_title(
        "PINN accuracy degradation vs viscosity\n"
        "Burgers' equation  u_t + u·u_x = ν·u_xx",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()   # low ν (hard) on the right

    plt.tight_layout()
    path = os.path.join(out_dir, "failure_error_vs_nu.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[failure] Saved '{path}'")


def plot_heatmap_comparison(
    x_grid:  np.ndarray,
    t_grid:  np.ndarray,
    u_pred:  np.ndarray,
    u_ref:   np.ndarray,
    nu:      float,
    rel_l2:  float,
    out_dir: str,
) -> None:
    """
    Three-panel heatmap: PINN prediction / FD reference / pointwise error.

    The error panel is the diagnostic: concentrated error at the shock
    location (x ≈ 0, late t) is the visual signature of PINN failure.
    """
    error = np.abs(u_pred - u_ref)

    vmin = min(u_pred.min(), u_ref.min())
    vmax = max(u_pred.max(), u_ref.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        f"PINN failure at ν = {nu:.4e}  "
        f"(Rel-L2 = {rel_l2:.1%})  —  "
        "Burgers' equation shock region",
        fontsize=12,
    )

    panels = [
        (u_pred, f"PINN prediction  (ν={nu:.2e})", "RdBu_r", vmin, vmax),
        (u_ref,  "FD reference  (Crank-Nicolson)",   "RdBu_r", vmin, vmax),
        (error,  "Pointwise error  |PINN − FD|",      "hot_r",  0.0,  error.max()),
    ]

    for ax, (data, title, cmap, lo, hi) in zip(axes, panels):
        cf = ax.contourf(
            x_grid, t_grid, data,
            levels=128,
            cmap=cmap,
            vmin=lo, vmax=hi,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(cf, cax=cax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("t", fontsize=10)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(T_MIN, T_MAX)

    plt.tight_layout()
    path = os.path.join(out_dir, "failure_heatmap_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[failure] Saved '{path}'")


# ================================================================== #
#  Main                                                                #
# ================================================================== #

def main() -> dict:
    cfg    = CONFIG
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    print(f"[failure] Device: {device}")

    results = []   # list of dicts, one per nu

    # ---------------------------------------------------------------- #
    #  Sweep over nu values                                             #
    # ---------------------------------------------------------------- #
    for i, nu in enumerate(NU_VALUES):
        ckpt_path = os.path.join(out_dir, f"model_nu_{i}.pt")

        print()
        print("=" * 65)
        print(f"  ν sweep [{i+1}/{len(NU_VALUES)}]  ν = {nu:.4e}  "
              f"(0.01/π = {0.01/np.pi:.4e}  ratio = {(0.01/np.pi)/nu:.1f}×)")
        print("=" * 65)

        # -- Train (skip if checkpoint already exists) --
        if os.path.isfile(ckpt_path):
            print(f"[failure] Checkpoint exists at '{ckpt_path}', loading...")
            model = BurgersPINN(
                n_hidden=cfg["n_hidden"], n_neurons=cfg["n_neurons"]
            ).to(device)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device)
            )
        else:
            t0 = time.time()
            model, _, _, _ = train_for_nu(
                nu=nu, cfg=cfg, device=device,
                save_path=ckpt_path, seed=i,
            )
            print(f"[failure] Training done in {(time.time()-t0)/60:.1f} min")

        # -- Evaluate --
        print(f"[failure] Computing FD reference for ν={nu:.4e} ...")
        rel_l2, x_grid, t_grid, u_pred, u_ref = compute_rel_l2(
            model, nu, device, cfg["n_x"], cfg["n_t"]
        )
        print(f"[failure] ν={nu:.4e}  Rel-L2 = {rel_l2:.4e}  ({rel_l2:.2%})")

        results.append({
            "nu":      nu,
            "rel_l2":  rel_l2,
            "ckpt":    ckpt_path,
            # Keep arrays only for the last (hardest) nu to save memory
            "x_grid":  x_grid if i == len(NU_VALUES) - 1 else None,
            "t_grid":  t_grid if i == len(NU_VALUES) - 1 else None,
            "u_pred":  u_pred if i == len(NU_VALUES) - 1 else None,
            "u_ref":   u_ref  if i == len(NU_VALUES) - 1 else None,
        })

    # ---------------------------------------------------------------- #
    #  Save metrics JSON                                                 #
    # ---------------------------------------------------------------- #
    metrics_out = [
        {"nu": r["nu"], "rel_l2": r["rel_l2"]}
        for r in results
    ]
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[failure] Metrics saved to '{metrics_path}'")

    # ---------------------------------------------------------------- #
    #  Plots                                                             #
    # ---------------------------------------------------------------- #
    nu_list  = [r["nu"]     for r in results]
    err_list = [r["rel_l2"] for r in results]

    plot_error_vs_nu(nu_list, err_list, out_dir)

    # Heatmap comparison at the hardest (last) nu
    worst = results[-1]
    plot_heatmap_comparison(
        worst["x_grid"], worst["t_grid"],
        worst["u_pred"], worst["u_ref"],
        nu=worst["nu"], rel_l2=worst["rel_l2"],
        out_dir=out_dir,
    )

    # ---------------------------------------------------------------- #
    #  Summary table                                                     #
    # ---------------------------------------------------------------- #
    print()
    print("=" * 65)
    print("  FAILURE ANALYSIS — SUMMARY")
    print("  " + "-" * 50)
    print(f"  {'ν':>12}   {'Rel-L2':>10}   {'Ratio vs baseline':>20}")
    print("  " + "-" * 50)
    base_err = results[0]["rel_l2"]
    for r in results:
        print(f"  {r['nu']:>12.4e}   {r['rel_l2']:>10.4e}   "
              f"  ×{r['rel_l2']/base_err:>6.1f}")
    print("=" * 65)

    return {"results": metrics_out, "out_dir": out_dir}


if __name__ == "__main__":
    main()
