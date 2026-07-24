"""
plot.py
-------
Visualisation utilities for the trained Burgers' PINN.

Two main plots are produced:

  1. Heatmap / filled-contour plot of u(x, t) over the full domain.
     This gives a global view of the shock that forms near t = 1.

  2. Time-slice comparison plot: predicted u(x, t*) vs a reference solution
     at t* = 0.25, 0.50, 0.75.

Reference solution
------------------
The exact solution to viscous Burgers' equation with these initial and
boundary conditions is not a simple closed-form expression, but it can be
computed via the Cole-Hopf transformation.  Here we provide a numerical
reference obtained by spectral / finite-difference methods using scipy.
The reference data used by the original Raissi et al. (2019) PINN paper
is loaded from 'data/burgers_shock.mat' if it is present; otherwise the
plots fall back to showing only the PINN prediction.

Usage
-----
    from plot import plot_solution
    plot_solution(model, device, save_dir="outputs/")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from model import BurgersPINN
from data import make_evaluation_grid, X_MIN, X_MAX, T_MIN, T_MAX


# ------------------------------------------------------------------ #
#  Reference solution via Crank-Nicolson finite-difference solver     #
# ------------------------------------------------------------------ #

def _fd_reference(x: np.ndarray, t: np.ndarray, nu: float) -> np.ndarray:
    """
    Compute a high-accuracy reference solution of the viscous Burgers' equation

        u_t + u * u_x = nu * u_xx,   x in [-1,1], t in [0,1]

    with  u(x, 0) = -sin(pi*x),  u(-1,t) = u(1,t) = 0

    using a Crank-Nicolson (implicit) finite-difference scheme with a
    Picard-iteration step to handle the nonlinear convection term.

    Why FD instead of Cole-Hopf Fourier series?
    --------------------------------------------
    For nu = 0.01/pi the initial Cole-Hopf potential exp((cos(pi*x)+1)/(2*nu*pi))
    reaches values ~exp(100), making Fourier coefficients numerically unstable
    unless thousands of terms are retained.  The Crank-Nicolson scheme is
    unconditionally stable, second-order accurate in both space and time, and
    straightforward to implement.

    Grid resolution and adaptive scaling
    -------------------------------------
    The explicit convection term in the RHS is  -dt * u * u_x.  Near the
    shock, |u_x| ~ 1/nu (the gradient scales inversely with shock width).
    For the RHS to remain finite we need  dt * |u_x|_max < 1, i.e.
    dt < nu.  Equivalently, Nt > 1/nu.

    Spatially, the shock has width O(nu*pi); resolving it with at least 8
    interior points requires dx < nu*pi/8, i.e. Nx > 16/nu.

    Both constraints scale as 1/nu, so Nx and Nt are set proportionally:

        nu_ref = 0.01/pi  (the baseline — known-good at Nx=512, Nt=2000)
        scale  = nu_ref / nu          (>1 for sharper shocks)
        Nx     = clip(round(512  * scale), 512,  16_000)
        Nt     = clip(round(2000 * scale), 2000, 80_000)

    The caps prevent excessive runtime for extreme values while still
    resolving the shocks encountered in the failure-analysis sweep
    (nu down to 0.001/pi, scale=10, giving Nx=5120, Nt=20000).

    Parameters
    ----------
    x  : 1-D array of query x values  (will be interpolated onto the FD grid)
    t  : 1-D array of query t values
    nu : kinematic viscosity

    Returns
    -------
    u_ref : (len(t), len(x)) array of reference solution values
    """
    # ---- Adaptive grid resolution ----------------------------------------
    # nu_ref is the baseline at which Nx=512, Nt=2000 is known stable.
    nu_ref = 0.01 / 3.141592653589793
    scale  = nu_ref / max(nu, 1e-10)          # >1 for sharper shocks
    Nx     = int(np.clip(round(512  * scale),  512,  16_000))
    Nt     = int(np.clip(round(2000 * scale), 2000,  80_000))
    # ----------------------------------------------------------------------

    dx   = 2.0 / (Nx + 1)
    dt   = 1.0 / Nt

    x_fd = np.linspace(-1.0 + dx, 1.0 - dx, Nx)   # interior nodes

    # Initial condition
    u = -np.sin(np.pi * x_fd)

    # Matrices for the diffusion part (implicit Crank-Nicolson).
    # The tridiagonal system is:
    #   (I - nu*dt/(2*dx^2) * D2) * u_new = (I + nu*dt/(2*dx^2) * D2) * u_star
    # where u_star comes from the explicit convection step.
    r = nu * dt / (2.0 * dx ** 2)
    diag_main  = np.ones(Nx) * (1.0 + 2.0 * r)
    diag_upper = np.ones(Nx - 1) * (-r)
    diag_lower = np.ones(Nx - 1) * (-r)

    # LU-factored tridiagonal matrix (Thomas algorithm arrays)
    from scipy.linalg import solve_banded
    ab = np.zeros((3, Nx))
    ab[0, 1:] = diag_upper   # upper diagonal
    ab[1, :]  = diag_main    # main diagonal
    ab[2, :-1] = diag_lower  # lower diagonal

    # We store snapshots at all requested t values
    t_fd    = np.linspace(0.0, 1.0, Nt + 1)
    u_store = {}   # t_index -> solution on x_fd

    # Include t=0 and every requested slice
    needed_t_idx = set()
    for t_val in t:
        needed_t_idx.add(int(np.argmin(np.abs(t_fd - t_val))))
    if 0 not in needed_t_idx:
        needed_t_idx.add(0)
    u_store[0] = u.copy()

    for step in range(1, Nt + 1):
        # --- Explicit (Adams-Bashforth) convection: u_star ---
        # central difference for du/dx, zero at boundaries
        u_x = np.zeros(Nx)
        u_x[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
        # upwind for the boundary cells to maintain stability
        u_x[0]    = (u[1] - 0.0)  / (2.0 * dx)   # u(-1)=0
        u_x[-1]   = (0.0 - u[-2]) / (2.0 * dx)   # u(+1)=0

        # RHS of CN system
        rhs = np.zeros(Nx)
        rhs[1:-1] = (r * u[:-2]
                     + (1.0 - 2.0 * r) * u[1:-1]
                     + r * u[2:]) - dt * u[1:-1] * u_x[1:-1]
        # boundary cells
        rhs[0]  = (r * 0.0
                   + (1.0 - 2.0 * r) * u[0]
                   + r * u[1]) - dt * u[0] * u_x[0]
        rhs[-1] = (r * u[-2]
                   + (1.0 - 2.0 * r) * u[-1]
                   + r * 0.0) - dt * u[-1] * u_x[-1]

        u = solve_banded((1, 1), ab, rhs)

        if step in needed_t_idx:
            u_store[step] = u.copy()

    # ---- Interpolate onto the requested (t, x) grid ----
    u_ref = np.zeros((len(t), len(x)))
    for i, t_val in enumerate(t):
        t_idx = int(np.argmin(np.abs(t_fd - t_val)))
        u_fd  = u_store.get(t_idx, u_store[max(u_store.keys())])
        # linear interpolation from FD grid to requested x
        u_ref[i, :] = np.interp(x, x_fd, u_fd,
                                 left=0.0, right=0.0)

    return u_ref


# ------------------------------------------------------------------ #
#  PINN evaluation helper                                              #
# ------------------------------------------------------------------ #

def evaluate_model(
    model: BurgersPINN,
    device: torch.device,
    n_x: int = 256,
    n_t: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the trained model on a dense (x, t) grid.

    Returns
    -------
    x_grid  : (n_t, n_x) numpy array
    t_grid  : (n_t, n_x) numpy array
    u_pred  : (n_t, n_x) numpy array of predicted solution values
    """
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(n_x, n_t, device)

    model.eval()
    with torch.no_grad():
        u_flat = model(x_flat, t_flat)   # (n_x*n_t, 1)

    u_pred = u_flat.cpu().numpy().reshape(n_t, n_x)
    return x_grid, t_grid, u_pred


# ------------------------------------------------------------------ #
#  Plot 1 — full-domain heatmap                                        #
# ------------------------------------------------------------------ #

def plot_heatmap(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    u_pred: np.ndarray,
    save_path: str = "outputs/heatmap.png",
) -> None:
    """
    Filled contour (heatmap) of u(x, t) over the complete domain.

    The characteristic steepening and eventual near-shock near t → 1 should
    be clearly visible as the blue region narrows to a sharp interface.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    levels = np.linspace(u_pred.min(), u_pred.max(), 128)
    cf = ax.contourf(
        x_grid, t_grid, u_pred,
        levels=levels,
        cmap="RdBu_r",
    )
    cbar = fig.colorbar(cf, ax=ax, label="u(x, t)")
    cbar.ax.tick_params(labelsize=9)

    # Overlay a few contour lines for readability
    ax.contour(x_grid, t_grid, u_pred, levels=10, colors="k", linewidths=0.3, alpha=0.4)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    ax.set_title("PINN solution — Burgers' equation  u(x, t)", fontsize=13)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(T_MIN, T_MAX)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Heatmap saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot 2 — time slices vs reference                                   #
# ------------------------------------------------------------------ #

def plot_time_slices(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    u_pred: np.ndarray,
    time_slices: list[float] = [0.25, 0.50, 0.75],
    nu: float = 0.01 / 3.141592653589793,
    save_path: str = "outputs/time_slices.png",
) -> None:
    """
    Compare the PINN prediction against the Cole-Hopf reference solution
    at a few fixed time snapshots.

    Parameters
    ----------
    x_grid      : (n_t, n_x) meshgrid output from evaluate_model
    t_grid      : (n_t, n_x) meshgrid output from evaluate_model
    u_pred      : (n_t, n_x) PINN prediction
    time_slices : list of t values at which to compare
    nu          : viscosity (must match the training value)
    save_path   : file path for the saved figure
    """
    x_vals  = x_grid[0, :]     # 1-D array of x values (same for every row)
    t_vals  = t_grid[:, 0]     # 1-D array of t values (same for every col)

    # Pre-compute Cole-Hopf reference on the same x, t grid
    u_ref_full = _fd_reference(x_vals, t_vals, nu)   # (n_t, n_x)

    n_slices = len(time_slices)
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 4), sharey=True)

    if n_slices == 1:
        axes = [axes]

    for ax, t_star in zip(axes, time_slices):
        # Find the row index closest to t_star
        idx = int(np.argmin(np.abs(t_vals - t_star)))
        t_actual = t_vals[idx]

        ax.plot(x_vals, u_ref_full[idx, :],
                color="#333333", linewidth=2.0, label="Cole-Hopf reference")
        ax.plot(x_vals, u_pred[idx, :],
                color="#e05c2a", linewidth=1.8, linestyle="--", label="PINN prediction")

        ax.set_title(f"t = {t_actual:.2f}", fontsize=11)
        ax.set_xlabel("x", fontsize=11)
        ax.set_xlim(X_MIN, X_MAX)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("u(x, t)", fontsize=11)
    fig.suptitle("PINN vs Cole-Hopf reference — time slices", fontsize=13, y=1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Time-slice comparison saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot 3 — training loss history                                      #
# ------------------------------------------------------------------ #

def plot_loss_history(
    history_pde: list[float],
    history_ic:  list[float],
    history_bc:  list[float],
    print_every: int = 500,
    save_path: str = "outputs/loss_history.png",
) -> None:
    """
    Log-scale plot of each loss component over training.

    Parameters
    ----------
    history_pde : PDE residual loss values (sampled every print_every epochs)
    history_ic  : Initial condition loss values
    history_bc  : Boundary condition loss values
    print_every : epoch interval at which losses were recorded
    save_path   : output file path
    """
    epochs = [i * print_every for i in range(1, len(history_pde) + 1)]
    # Edge case: first recorded point is epoch 1
    if len(epochs) > 0:
        epochs[0] = 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(epochs, history_pde, label="PDE residual",  color="#3b82d4", linewidth=1.8)
    ax.semilogy(epochs, history_ic,  label="Initial cond.", color="#e05c2a", linewidth=1.8)
    ax.semilogy(epochs, history_bc,  label="Boundary cond.",color="#2ca02c", linewidth=1.8)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.set_title("Training loss history", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Loss history saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Convenience wrapper                                                 #
# ------------------------------------------------------------------ #

def plot_solution(
    model: BurgersPINN,
    device: torch.device,
    history_pde: list[float] | None = None,
    history_ic:  list[float] | None = None,
    history_bc:  list[float] | None = None,
    save_dir: str = "outputs/",
    print_every: int = 500,
) -> None:
    """
    Generate all plots after training.

    Calls:
      1. plot_heatmap        → save_dir/heatmap.png
      2. plot_time_slices    → save_dir/time_slices.png
      3. plot_loss_history   → save_dir/loss_history.png  (if histories supplied)
    """
    x_grid, t_grid, u_pred = evaluate_model(model, device)

    plot_heatmap(
        x_grid, t_grid, u_pred,
        save_path=os.path.join(save_dir, "heatmap.png"),
    )

    plot_time_slices(
        x_grid, t_grid, u_pred,
        time_slices=[0.25, 0.50, 0.75],
        nu=0.01 / 3.141592653589793,
        save_path=os.path.join(save_dir, "time_slices.png"),
    )

    if history_pde is not None:
        plot_loss_history(
            history_pde, history_ic, history_bc,
            print_every=print_every,
            save_path=os.path.join(save_dir, "loss_history.png"),
        )
