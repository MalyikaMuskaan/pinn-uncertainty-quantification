"""
ensemble_plot.py  —  ocean_pinn
--------------------------------
Ensemble UQ plots for the advection-diffusion PINN.

Styled identically to burgers_pinn/ensemble_plot.py for direct comparison.
Uses YlOrRd for concentration (matching single-model heatmap) and viridis
(log-scale) for uncertainty.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm

from plot import fd_reference
from data import X_MIN, X_MAX, T_MIN, T_MAX


# ------------------------------------------------------------------ #
#  A — ensemble mean heatmap                                           #
# ------------------------------------------------------------------ #

def plot_ensemble_mean_heatmap(
    x_grid:  np.ndarray,
    t_grid:  np.ndarray,
    c_mean:  np.ndarray,
    save_path: str = "outputs/ensemble/ensemble_mean_heatmap.png",
) -> None:
    """Filled-contour of the ensemble mean c(x,t)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    vmax   = max(abs(c_mean.max()), 1e-6)
    levels = np.linspace(0, vmax, 128)
    cf = ax.contourf(x_grid, t_grid, np.clip(c_mean, 0, None),
                     levels=levels, cmap="YlOrRd")
    cbar = fig.colorbar(cf, ax=ax, label="c(x,t)  [ensemble mean]")
    cbar.ax.tick_params(labelsize=9)
    ax.contour(x_grid, t_grid, c_mean,
               levels=8, colors="k", linewidths=0.3, alpha=0.4)
    ax.set_xlabel("x  [km]", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    ax.set_title("Ensemble mean — advection-diffusion  c(x, t)", fontsize=13)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(T_MIN, T_MAX)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[ensemble_plot] Mean heatmap saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  B — uncertainty (std) heatmap                                       #
# ------------------------------------------------------------------ #

def plot_ensemble_std_heatmap(
    x_grid:  np.ndarray,
    t_grid:  np.ndarray,
    c_std:   np.ndarray,
    save_path: str = "outputs/ensemble/ensemble_std_heatmap.png",
) -> None:
    """Log-scale uncertainty heatmap."""
    fig, ax = plt.subplots(figsize=(9, 5))
    std_plot = np.clip(c_std, 1e-6, None)
    vmin = float(std_plot.min())
    vmax = float(std_plot.max())
    if vmax <= vmin * 1.001:
        vmax = vmin * 10.0

    norm   = LogNorm(vmin=vmin, vmax=vmax)
    levels = np.unique(np.logspace(np.log10(vmin), np.log10(vmax), 64))
    cf = ax.contourf(x_grid, t_grid, std_plot,
                     levels=levels, norm=norm, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax, label="std(c)  [log scale]")
    cbar.ax.tick_params(labelsize=9)
    cbar.locator = mticker.LogLocator()
    cbar.update_ticks()

    ax.set_xlabel("x  [km]", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    ax.set_title("Ensemble uncertainty — std(c(x, t))", fontsize=13)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(T_MIN, T_MAX)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[ensemble_plot] Std heatmap saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  C — time slices with uncertainty bands                              #
# ------------------------------------------------------------------ #

def plot_ensemble_time_slices(
    x_grid:      np.ndarray,
    t_grid:      np.ndarray,
    c_mean:      np.ndarray,
    c_std:       np.ndarray,
    time_slices: list[float] = [1.0, 2.5, 4.0],
    n_sigma:     float = 2.0,
    save_path:   str = "outputs/ensemble/ensemble_time_slices.png",
) -> None:
    """Mean ± n_sigma*std band vs FD reference at fixed time slices."""
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]

    print("[ensemble_plot] Computing FD reference for time-slice plot...")
    c_ref_full = fd_reference(x_vals, t_vals)

    n_slices = len(time_slices)
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 4), sharey=True)
    if n_slices == 1:
        axes = [axes]

    for ax, t_star in zip(axes, time_slices):
        idx      = int(np.argmin(np.abs(t_vals - t_star)))
        t_actual = t_vals[idx]

        mean_s = c_mean[idx, :]
        std_s  = c_std[idx, :]
        ref_s  = c_ref_full[idx, :]

        ax.fill_between(x_vals,
                        mean_s - n_sigma * std_s,
                        mean_s + n_sigma * std_s,
                        color="#2563eb", alpha=0.20,
                        label=f"mean +/- {n_sigma:.0f}*std")
        ax.plot(x_vals, ref_s,  color="#1f2328", lw=2.0, label="FD reference")
        ax.plot(x_vals, mean_s, color="#2563eb", lw=1.8, ls="--",
                label="Ensemble mean")

        ax.set_title(f"t = {t_actual:.2f}", fontsize=11)
        ax.set_xlabel("x  [km]", fontsize=11)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(bottom=-0.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("c(x, t)", fontsize=11)
    fig.suptitle(
        f"Ensemble mean +/- {n_sigma:.0f}*std  vs  FD reference",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ensemble_plot] Time-slice plot saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  D — calibration diagram                                             #
# ------------------------------------------------------------------ #

def plot_calibration(
    confidence_levels:  np.ndarray,
    empirical_coverage: np.ndarray,
    ece:                float,
    coverage_90:        float,
    save_path: str = "outputs/ensemble/ensemble_calibration.png",
) -> None:
    """Reliability diagram (same style as burgers_pinn)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="#57606a", lw=1.2, ls="--",
            label="Perfect calibration")
    ax.fill_between(confidence_levels, confidence_levels, empirical_coverage,
                    alpha=0.15, color="#3b82d4")
    ax.plot(confidence_levels, empirical_coverage,
            color="#3b82d4", lw=2.2, marker="o", markersize=4,
            label="Ensemble calibration")
    ax.text(
        0.04, 0.88,
        f"ECE = {ece:.4f}\n90% coverage = {coverage_90:.4f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f8fa",
                  edgecolor="#e5e7eb", lw=0.8),
    )
    ax.set_xlabel("Nominal confidence level", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Calibration diagram — Deep Ensemble", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[ensemble_plot] Calibration plot saved to '{save_path}'")
