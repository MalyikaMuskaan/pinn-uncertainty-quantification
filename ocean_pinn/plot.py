"""
plot.py  —  ocean_pinn
-----------------------
Visualisation utilities for the advection-diffusion PINN.

Plots produced:
  1. Heatmap of c(x, t) over the full domain
  2. Time-slice comparison: PINN vs Crank-Nicolson FD reference
  3. Training loss history (log scale)

Reference solution
------------------
The advection-diffusion equation has an exact analytical solution for an
infinite domain with Gaussian IC:

  c_exact(x, t) = 1/sqrt(1 + 4*D*t/w) * exp( -(x - x0 - v*t)^2 / (w + 4*D*t) )

where w = IC_WIDTH = 0.5.  However, our domain is finite ([0,10]) with zero
Dirichlet BCs, so the exact solution includes reflections.  We therefore use
the same Crank-Nicolson FD solver approach as burgers_pinn/plot.py to get
an accurate numerical reference that respects the BCs.

The infinite-domain analytical solution is also implemented here for sanity
checking at early times when boundary effects are negligible.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

from model import OceanPINN
from data import make_evaluation_grid, X_MIN, X_MAX, T_MIN, T_MAX, V, D, IC_CENTER, IC_WIDTH


# ------------------------------------------------------------------ #
#  Reference solution — Crank-Nicolson FD solver                      #
# ------------------------------------------------------------------ #

def fd_reference(
    x: np.ndarray,
    t: np.ndarray,
    v: float = V,
    diff: float = D,
    nx_fd: int = 512,
    nt_fd: int = 4000,
) -> np.ndarray:
    """
    Crank-Nicolson implicit FD solution for:
        c_t + v*c_x = D*c_xx,   x in [0,10],  t in [0,5]
        c(x,0) = exp(-(x-2)^2/0.5),  c(0,t) = c(10,t) = 0

    The advection term is treated explicitly (upwind difference for stability),
    and the diffusion term is treated implicitly (CN).

    Parameters
    ----------
    x      : 1-D query x values
    t      : 1-D query t values
    v      : advection velocity
    diff   : diffusion coefficient
    nx_fd  : interior spatial grid points
    nt_fd  : number of time steps

    Returns
    -------
    c_ref : (len(t), len(x)) reference solution
    """
    from scipy.linalg import solve_banded

    dx = (X_MAX - X_MIN) / (nx_fd + 1)
    dt = T_MAX / nt_fd

    x_fd = np.linspace(X_MIN + dx, X_MAX - dx, nx_fd)

    # Initial condition
    c = np.exp(-((x_fd - IC_CENTER) ** 2) / IC_WIDTH).astype(np.float64)

    # CN diffusion coefficients
    r  = diff * dt / (2.0 * dx ** 2)
    # Upwind advection coefficient (explicit, one-sided)
    # For v > 0 use backward (upwind) difference: c_x ≈ (c_i - c_{i-1})/dx
    adv = v * dt / dx

    # Build banded matrix for the diffusion (implicit) part
    main  = np.ones(nx_fd) * (1.0 + 2.0 * r)
    upper = np.ones(nx_fd - 1) * (-r)
    lower = np.ones(nx_fd - 1) * (-r)

    ab = np.zeros((3, nx_fd))
    ab[0, 1:]  = upper
    ab[1, :]   = main
    ab[2, :-1] = lower

    # Identify which time indices to store
    t_fd = np.linspace(0.0, T_MAX, nt_fd + 1)
    needed = set()
    for tv in t:
        needed.add(int(np.argmin(np.abs(t_fd - tv))))
    needed.add(0)
    store = {0: c.copy()}

    for step in range(1, nt_fd + 1):
        # --- Explicit upwind advection ---
        adv_term = np.zeros(nx_fd)
        # interior (upwind for v > 0: use left neighbour)
        adv_term[1:] = adv * (c[1:] - c[:-1])
        # left boundary cell: c_{-1} = 0 (BC)
        adv_term[0]  = adv * (c[0] - 0.0)

        # --- RHS: explicit diffusion half-step + advection ---
        rhs = np.zeros(nx_fd)
        rhs[1:-1] = (r * c[:-2]
                     + (1.0 - 2.0 * r) * c[1:-1]
                     + r * c[2:]) - adv_term[1:-1]
        rhs[0]  = ((1.0 - 2.0 * r) * c[0]
                   + r * c[1]) - adv_term[0]
        rhs[-1] = (r * c[-2]
                   + (1.0 - 2.0 * r) * c[-1]) - adv_term[-1]

        c = solve_banded((1, 1), ab, rhs)
        c = np.clip(c, 0.0, None)   # concentration cannot be negative

        if step in needed:
            store[step] = c.copy()

    # Interpolate onto requested (t, x) grid
    c_ref = np.zeros((len(t), len(x)))
    for i, tv in enumerate(t):
        idx  = int(np.argmin(np.abs(t_fd - tv)))
        c_fd = store.get(idx, store[max(store.keys())])
        c_ref[i, :] = np.interp(x, x_fd, c_fd, left=0.0, right=0.0)

    return c_ref


# ------------------------------------------------------------------ #
#  Analytical solution (infinite domain — valid for t << 1 or small D) #
# ------------------------------------------------------------------ #

def analytical_reference(
    x: np.ndarray,
    t: np.ndarray,
    v: float = V,
    diff: float = D,
) -> np.ndarray:
    """
    Exact Gaussian solution for the infinite-domain advection-diffusion problem.
    Valid when the pulse hasn't yet felt the boundaries.

    c(x, t) = sqrt(w/(w + 4*D*t)) * exp( -(x - x0 - v*t)^2 / (w + 4*D*t) )
    """
    c_out = np.zeros((len(t), len(x)))
    w = IC_WIDTH
    for i, ti in enumerate(t):
        denom = w + 4.0 * diff * ti
        amp   = np.sqrt(w / denom)
        c_out[i, :] = amp * np.exp(-((x - IC_CENTER - v * ti) ** 2) / denom)
    return c_out


# ------------------------------------------------------------------ #
#  Model evaluation on dense grid                                      #
# ------------------------------------------------------------------ #

def evaluate_model(
    model: OceanPINN,
    device: torch.device,
    n_x: int = 256,
    n_t: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on a dense grid; return x_grid, t_grid, c_pred."""
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(n_x, n_t, device)
    model.eval()
    with torch.no_grad():
        c_flat = model(x_flat, t_flat)
    c_pred = c_flat.cpu().numpy().reshape(n_t, n_x)
    return x_grid, t_grid, c_pred


# ------------------------------------------------------------------ #
#  Plot 1 — heatmap                                                    #
# ------------------------------------------------------------------ #

def plot_heatmap(
    x_grid:  np.ndarray,
    t_grid:  np.ndarray,
    c_pred:  np.ndarray,
    title:   str = "PINN solution — advection-diffusion  c(x, t)",
    save_path: str = "outputs/heatmap.png",
) -> None:
    """Filled-contour plot of c(x,t) over the full domain."""
    fig, ax = plt.subplots(figsize=(9, 5))

    vmax = max(abs(c_pred.max()), 1e-6)
    levels = np.linspace(0, vmax, 128)
    cf = ax.contourf(x_grid, t_grid, np.clip(c_pred, 0, None),
                     levels=levels, cmap="YlOrRd")
    cbar = fig.colorbar(cf, ax=ax, label="c(x, t)  [concentration]")
    cbar.ax.tick_params(labelsize=9)
    ax.contour(x_grid, t_grid, c_pred,
               levels=8, colors="k", linewidths=0.3, alpha=0.4)

    ax.set_xlabel("x  [km]", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(T_MIN, T_MAX)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Heatmap saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot 2 — time slices vs reference                                   #
# ------------------------------------------------------------------ #

def plot_time_slices(
    x_grid:     np.ndarray,
    t_grid:     np.ndarray,
    c_pred:     np.ndarray,
    time_slices: list[float] = [1.0, 2.5, 4.0],
    save_path:  str = "outputs/time_slices.png",
) -> None:
    """PINN prediction vs Crank-Nicolson FD reference at fixed t values."""
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]

    print("[plot] Computing FD reference for time-slice plot...")
    c_ref_full = fd_reference(x_vals, t_vals)   # (n_t, n_x)

    n_slices = len(time_slices)
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 4), sharey=True)
    if n_slices == 1:
        axes = [axes]

    for ax, t_star in zip(axes, time_slices):
        idx      = int(np.argmin(np.abs(t_vals - t_star)))
        t_actual = t_vals[idx]

        ax.plot(x_vals, c_ref_full[idx, :],
                color="#333333", lw=2.0, label="FD reference")
        ax.plot(x_vals, c_pred[idx, :],
                color="#2563eb", lw=1.8, ls="--", label="PINN prediction")

        ax.set_title(f"t = {t_actual:.2f}", fontsize=11)
        ax.set_xlabel("x  [km]", fontsize=11)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(bottom=-0.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("c(x, t)", fontsize=11)
    fig.suptitle("PINN vs FD reference — time slices", fontsize=13, y=1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Time-slice comparison saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot 3 — loss history                                               #
# ------------------------------------------------------------------ #

def plot_loss_history(
    history_pde: list[float],
    history_ic:  list[float],
    history_bc:  list[float],
    print_every: int = 500,
    save_path:   str = "outputs/loss_history.png",
) -> None:
    """Log-scale training curves for all three loss terms."""
    epochs = [i * print_every for i in range(1, len(history_pde) + 1)]
    if epochs:
        epochs[0] = 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(epochs, history_pde, label="PDE residual",  color="#3b82d4", lw=1.8)
    ax.semilogy(epochs, history_ic,  label="Initial cond.", color="#e05c2a", lw=1.8)
    ax.semilogy(epochs, history_bc,  label="Boundary cond.",color="#2ca02c", lw=1.8)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.set_title("Training loss history", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Loss history saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Convenience wrapper                                                 #
# ------------------------------------------------------------------ #

def plot_solution(
    model:       OceanPINN,
    device:      torch.device,
    history_pde: list[float] | None = None,
    history_ic:  list[float] | None = None,
    history_bc:  list[float] | None = None,
    save_dir:    str = "outputs/",
    print_every: int = 500,
) -> None:
    """Generate heatmap, time-slice comparison, and optional loss history."""
    x_grid, t_grid, c_pred = evaluate_model(model, device)

    plot_heatmap(
        x_grid, t_grid, c_pred,
        save_path=os.path.join(save_dir, "heatmap.png"),
    )
    plot_time_slices(
        x_grid, t_grid, c_pred,
        time_slices=[1.0, 2.5, 4.0],
        save_path=os.path.join(save_dir, "time_slices.png"),
    )
    if history_pde is not None:
        plot_loss_history(
            history_pde, history_ic, history_bc,
            print_every=print_every,
            save_path=os.path.join(save_dir, "loss_history.png"),
        )
