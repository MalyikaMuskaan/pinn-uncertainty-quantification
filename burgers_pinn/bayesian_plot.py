"""
bayesian_plot.py
----------------
Plotting utilities for the Bayesian PINN UQ results.

All four functions are styled identically to ensemble_plot.py so that
the outputs can be placed side-by-side for a fair visual comparison.

Functions
---------
plot_bayesian_mean_heatmap   — ensemble-mean-style heatmap of MC mean
plot_bayesian_std_heatmap    — log-scale heatmap of MC std
plot_bayesian_time_slices    — mean +/- 2*std shaded band vs FD reference
plot_bayesian_calibration    — reliability diagram
plot_bayesian_loss_history   — PDE / IC / BC / KL loss curves during training
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from plot import _fd_reference
from data import X_MIN, X_MAX, T_MIN, T_MAX

NU = 0.01 / 3.141592653589793


# ------------------------------------------------------------------ #
#  A — mean heatmap                                                    #
# ------------------------------------------------------------------ #

def plot_bayesian_mean_heatmap(
    x_grid:   np.ndarray,
    t_grid:   np.ndarray,
    u_mean:   np.ndarray,
    save_path: str = "outputs/bayesian/bayesian_mean_heatmap.png",
) -> None:
    """Filled-contour of the MC mean prediction (same style as ensemble)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    levels = np.linspace(u_mean.min(), u_mean.max(), 128)
    cf = ax.contourf(x_grid, t_grid, u_mean, levels=levels, cmap="RdBu_r")
    cbar = fig.colorbar(cf, ax=ax, label="u(x, t)  [MC mean]")
    cbar.ax.tick_params(labelsize=9)
    ax.contour(x_grid, t_grid, u_mean,
               levels=10, colors="k", linewidths=0.3, alpha=0.4)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    ax.set_title("Bayesian PINN (VI) mean prediction  u(x, t)", fontsize=13)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(T_MIN, T_MAX)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[bayesian_plot] Mean heatmap saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  B — uncertainty (std) heatmap                                       #
# ------------------------------------------------------------------ #

def plot_bayesian_std_heatmap(
    x_grid:   np.ndarray,
    t_grid:   np.ndarray,
    u_std:    np.ndarray,
    save_path: str = "outputs/bayesian/bayesian_std_heatmap.png",
) -> None:
    """Log-scale heatmap of MC std (uncertainty). Same style as ensemble."""
    fig, ax = plt.subplots(figsize=(8, 5))

    std_plot = np.clip(u_std.copy(), 1e-6, None)
    vmin = float(std_plot.min())
    vmax = float(std_plot.max())
    if vmax <= vmin * 1.001:
        vmax = vmin * 10.0

    norm   = LogNorm(vmin=vmin, vmax=vmax)
    levels = np.unique(np.logspace(np.log10(vmin), np.log10(vmax), 64))

    cf = ax.contourf(x_grid, t_grid, std_plot,
                     levels=levels, norm=norm, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax, label="std(u)  [log scale]")
    cbar.ax.tick_params(labelsize=9)
    cbar.locator = mticker.LogLocator()
    cbar.update_ticks()

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    ax.set_title("Bayesian PINN (VI) uncertainty  std(u(x, t))", fontsize=13)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(T_MIN, T_MAX)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[bayesian_plot] Std heatmap saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  C — time slices with shaded uncertainty bands                       #
# ------------------------------------------------------------------ #

def plot_bayesian_time_slices(
    x_grid:      np.ndarray,
    t_grid:      np.ndarray,
    u_mean:      np.ndarray,
    u_std:       np.ndarray,
    time_slices: list[float] = [0.25, 0.50, 0.75],
    n_sigma:     float = 2.0,
    save_path:   str = "outputs/bayesian/bayesian_time_slices.png",
) -> None:
    """Mean +/- n_sigma*std band vs FD reference at fixed time slices."""
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]

    print("[bayesian_plot] Computing FD reference for time-slice plot...")
    u_ref_full = _fd_reference(x_vals, t_vals, NU)

    n_slices = len(time_slices)
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 4), sharey=True)
    if n_slices == 1:
        axes = [axes]

    for ax, t_star in zip(axes, time_slices):
        idx      = int(np.argmin(np.abs(t_vals - t_star)))
        t_actual = t_vals[idx]

        mean_s = u_mean[idx, :]
        std_s  = u_std[idx, :]
        ref_s  = u_ref_full[idx, :]

        ax.fill_between(x_vals, mean_s - n_sigma * std_s,
                                mean_s + n_sigma * std_s,
                        color="#7c5cd8", alpha=0.20,
                        label=f"mean +/- {n_sigma:.0f}*std")
        ax.plot(x_vals, ref_s, color="#1f2328", linewidth=2.0,
                label="FD reference")
        ax.plot(x_vals, mean_s, color="#7c5cd8", linewidth=1.8,
                linestyle="--", label="Bayes mean")

        ax.set_title(f"t = {t_actual:.2f}", fontsize=11)
        ax.set_xlabel("x", fontsize=11)
        ax.set_xlim(X_MIN, X_MAX)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("u(x, t)", fontsize=11)
    fig.suptitle(
        f"Bayesian PINN mean +/- {n_sigma:.0f}*std  vs  FD reference",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[bayesian_plot] Time-slice plot saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  D — calibration diagram                                             #
# ------------------------------------------------------------------ #

def plot_bayesian_calibration(
    confidence_levels:  np.ndarray,
    empirical_coverage: np.ndarray,
    ece:                float,
    coverage_90:        float,
    save_path: str = "outputs/bayesian/bayesian_calibration.png",
) -> None:
    """Reliability diagram (same style as ensemble_plot.plot_calibration)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot([0, 1], [0, 1], color="#57606a", linewidth=1.2,
            linestyle="--", label="Perfect calibration")
    ax.fill_between(confidence_levels, confidence_levels, empirical_coverage,
                    alpha=0.15, color="#7c5cd8")
    ax.plot(confidence_levels, empirical_coverage,
            color="#7c5cd8", linewidth=2.2, marker="o", markersize=4,
            label="Bayesian PINN calibration")

    ax.text(
        0.04, 0.88,
        f"ECE = {ece:.4f}\n90% coverage = {coverage_90:.4f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f8fa",
                  edgecolor="#e5e7eb", linewidth=0.8),
    )

    ax.set_xlabel("Nominal confidence level", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Calibration diagram — Bayesian PINN (VI)", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[bayesian_plot] Calibration plot saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  E — training loss history                                           #
# ------------------------------------------------------------------ #

def plot_bayesian_loss_history(
    history_pde: list[float],
    history_ic:  list[float],
    history_bc:  list[float],
    history_kl:  list[float],
    print_every: int = 500,
    save_path:   str = "outputs/bayesian/bayesian_loss_history.png",
) -> None:
    """Log-scale training curves including the KL divergence term."""
    epochs = [i * print_every for i in range(1, len(history_pde) + 1)]
    if epochs:
        epochs[0] = 1

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(epochs, history_pde, label="PDE residual",   color="#3b82d4", lw=1.8)
    ax.semilogy(epochs, history_ic,  label="Initial cond.",  color="#e05c2a", lw=1.8)
    ax.semilogy(epochs, history_bc,  label="Boundary cond.", color="#2ca02c", lw=1.8)
    ax.semilogy(epochs, history_kl,  label="KL divergence",  color="#7c5cd8",
                lw=1.8, linestyle="--")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.set_title("Bayesian PINN — training loss history", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[bayesian_plot] Loss history saved to '{save_path}'")
