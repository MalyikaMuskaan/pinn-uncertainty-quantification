"""
plot.py
-------
Visualisation for the 2-D Darcy flow PINN.

Figures produced
----------------
1. solution_comparison.png
   Three side-by-side panels:
     (a) PINN predicted u(x,y)
     (b) Analytical exact u*(x,y) = sin(πx)sin(πy)
     (c) Pointwise absolute error |u_pred - u_exact|
   All three share the same spatial axes and are displayed as filled
   contour / imshow plots with a colour bar.

2. loss_history.png
   Log-scale training curves for the total loss, PDE residual loss, and
   boundary condition loss vs. epoch index (one point per logged epoch).

Usage
-----
    from plot import plot_solution, plot_loss_history
    plot_solution(model, device, save_dir="outputs")
    plot_loss_history(loss_history, save_dir="outputs")

Both functions save PNG files and close the figure to avoid memory leaks.
"""

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Colab / headless)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model import DarcyPINN
from data  import make_eval_grid, exact_solution_np


# ------------------------------------------------------------------ #
#  Solution comparison                                                 #
# ------------------------------------------------------------------ #

def plot_solution(
    model:    DarcyPINN,
    device:   torch.device,
    n_grid:   int = 256,
    save_dir: str = "outputs",
    fname:    str = "solution_comparison.png",
) -> str:
    """
    Plot PINN prediction, exact solution, and pointwise error as a
    three-panel heatmap figure.

    Parameters
    ----------
    model    : trained DarcyPINN
    device   : torch device
    n_grid   : resolution of the evaluation grid (n_grid × n_grid)
    save_dir : directory to save the PNG
    fname    : output filename

    Returns
    -------
    path : full path to the saved figure
    """
    model.eval()

    # ---- Evaluate model on the grid ------------------------------------
    x_flat, y_flat, X, Y = make_eval_grid(n_grid, device=device)

    with torch.no_grad():
        u_pred_flat = model(x_flat, y_flat).cpu().numpy()   # (n_grid², 1)

    u_pred = u_pred_flat.reshape(n_grid, n_grid)            # (n_grid, n_grid)

    # ---- Exact solution ------------------------------------------------
    u_exact = exact_solution_np(X, Y)                       # (n_grid, n_grid)

    # ---- Pointwise error -----------------------------------------------
    error = np.abs(u_pred - u_exact)

    # ---- Compute summary metrics (printed, not plotted) ----------------
    mse    = float(np.mean((u_pred - u_exact) ** 2))
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) /
                   (np.linalg.norm(u_exact) + 1e-12))
    print(f"[plot] MSE={mse:.4e}   Rel-L2={rel_l2:.4e}")

    # ---- Figure --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "Darcy Flow PINN  —  2-D Steady-State\n"
        r"$-\nabla\cdot(k\,\nabla u)=f$"
        f"     MSE={mse:.2e}   Rel-L2={rel_l2:.2%}",
        fontsize=12,
    )

    vmin = min(u_pred.min(), u_exact.min())
    vmax = max(u_pred.max(), u_exact.max())

    panels = [
        (u_pred,  "PINN prediction $\\hat{u}$",     "viridis",  vmin, vmax),
        (u_exact, "Exact $u^*=\\sin(\\pi x)\\sin(\\pi y)$", "viridis", vmin, vmax),
        (error,   "Pointwise error $|\\hat{u}-u^*|$", "hot_r",  0.0,  error.max()),
    ]

    for ax, (data, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(
            data,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            aspect="equal",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Colour bar attached to each panel
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved '{path}'")
    return path


# ------------------------------------------------------------------ #
#  Loss history                                                        #
# ------------------------------------------------------------------ #

def plot_loss_history(
    loss_history: dict,
    save_dir:     str = "outputs",
    fname:        str = "loss_history.png",
    adam_epochs:  int = 3_000,
    print_every:  int = 500,
) -> str:
    """
    Plot training loss curves on a log scale.

    Parameters
    ----------
    loss_history : dict with keys 'pde', 'bc', 'total'
                   each a list of loss values (one per logged epoch)
    save_dir     : directory to save the PNG
    fname        : output filename
    adam_epochs  : used to draw a vertical line separating Adam/L-BFGS phases
    print_every  : epoch interval at which losses were logged (for x-axis ticks)

    Returns
    -------
    path : full path to the saved figure
    """
    pde_vals   = loss_history.get("pde",   [])
    bc_vals    = loss_history.get("bc",    [])
    total_vals = loss_history.get("total", [])

    n_pts = len(total_vals)
    # Reconstruct epoch indices from print_every
    # First logged point is epoch 1, then every print_every thereafter
    epochs = [1] + [print_every * i for i in range(1, n_pts)]
    epochs = epochs[:n_pts]

    fig, ax = plt.subplots(figsize=(8, 4))

    if total_vals:
        ax.semilogy(epochs, total_vals, label="Total",       color="black",   lw=1.8)
    if pde_vals:
        ax.semilogy(epochs, pde_vals,   label="PDE residual",color="#3b82d4", lw=1.4, ls="--")
    if bc_vals:
        ax.semilogy(epochs, bc_vals,    label="BC",          color="#f87171", lw=1.4, ls="-.")

    # Phase separator
    if adam_epochs < (epochs[-1] if epochs else 0):
        ax.axvline(adam_epochs, color="gray", lw=1.0, ls=":", label=f"Adam→L-BFGS (ep {adam_epochs})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Darcy PINN — Training Loss History")
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", alpha=0.1)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved '{path}'")
    return path


# ------------------------------------------------------------------ #
#  Residual map (optional diagnostic)                                  #
# ------------------------------------------------------------------ #

def plot_pde_residual(
    model:    DarcyPINN,
    device:   torch.device,
    n_grid:   int = 128,
    save_dir: str = "outputs",
    fname:    str = "pde_residual_map.png",
) -> str:
    """
    Evaluate and plot the PDE residual R(x,y) over the domain.

    This is a diagnostic tool — a well-trained PINN should show |R| ≈ 0
    everywhere in the interior.

    Parameters
    ----------
    model    : trained DarcyPINN
    device   : torch device
    n_grid   : resolution (n_grid × n_grid) — use smaller than solution plot
               to keep autograd tractable (128 gives 128²=16384 forward passes)
    save_dir : directory to save the PNG
    fname    : output filename

    Returns
    -------
    path : full path to the saved figure
    """
    from train import pde_residual as compute_residual

    model.eval()
    lin = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin)

    x_flat = torch.tensor(X.reshape(-1, 1), device=device, requires_grad=True)
    y_flat = torch.tensor(Y.reshape(-1, 1), device=device, requires_grad=True)

    R = compute_residual(model, x_flat, y_flat)
    R_np = R.detach().cpu().numpy().reshape(n_grid, n_grid)
    R_abs = np.abs(R_np)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(
        R_abs,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="hot_r",
        aspect="equal",
    )
    ax.set_title(r"PDE Residual Map  $|R(x,y)|$", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved '{path}'  (max|R|={R_abs.max():.3e})")
    return path
