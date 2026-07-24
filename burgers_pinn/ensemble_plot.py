"""
ensemble_plot.py
----------------
Plotting utilities specific to the Deep Ensemble UQ results.

Functions
---------
plot_ensemble_mean_heatmap   — filled-contour of the ensemble mean u(x,t)
plot_ensemble_std_heatmap    — filled-contour of ensemble std (uncertainty)
plot_ensemble_time_slices    — mean +/- 2*std shaded band vs FD reference
plot_calibration             — reliability / calibration diagram

These are kept in a separate file from plot.py so that plot.py (used by the
single-model main.py workflow) is not cluttered with ensemble-specific code.
The FD reference solver is imported from plot.py and reused directly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from plot import _fd_reference
from data import X_MIN, X_MAX, T_MIN, T_MAX

NU = 0.01 / 3.141592653589793


# ------------------------------------------------------------------ #
#  Plot A — ensemble mean heatmap                                      #
# ------------------------------------------------------------------ #

def plot_ensemble_mean_heatmap(
    x_grid:  np.ndarray,
    t_grid:  np.ndarray,
    u_mean:  np.ndarray,
    save_path: str = "outputs/ensemble/ensemble_mean_heatmap.png",
) -> None:
    """
    Filled-contour plot of the ensemble mean prediction over the full domain.

    Identical style to the single-model heatmap in plot.py so results are
    directly visually comparable.

    Parameters
    ----------
    x_grid  : (n_t, n_x) meshgrid x
    t_grid  : (n_t, n_x) meshgrid t
    u_mean  : (n_t, n_x) ensemble mean
    save_path : output path
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    levels = np.linspace(u_mean.min(), u_mean.max(), 128)
    cf = ax.contourf(x_grid, t_grid, u_mean, levels=levels, cmap="RdBu_r")
    cbar = fig.colorbar(cf, ax=ax, label="u(x, t)  [ensemble mean]")
    cbar.ax.tick_params(labelsize=9)
    ax.contour(x_grid, t_grid, u_mean,
               levels=10, colors="k", linewidths=0.3, alpha=0.4)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    ax.set_title("Ensemble mean — Burgers' equation  u(x, t)", fontsize=13)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(T_MIN, T_MAX)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[ensemble_plot] Mean heatmap saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot B — ensemble uncertainty (std) heatmap                         #
# ------------------------------------------------------------------ #

def plot_ensemble_std_heatmap(
    x_grid:  np.ndarray,
    t_grid:  np.ndarray,
    u_std:   np.ndarray,
    save_path: str = "outputs/ensemble/ensemble_std_heatmap.png",
) -> None:
    """
    Filled-contour plot of the ensemble standard deviation (uncertainty).

    Key feature to look for: uncertainty should be highest near the shock
    region (x ~ 0, t ~ 0.8-1.0) where the solution is hardest to learn,
    and lowest in smooth regions early in time.

    Uses a perceptually-uniform sequential colormap (viridis) so that
    magnitude differences are visually faithful.

    Parameters
    ----------
    x_grid  : (n_t, n_x) meshgrid x
    t_grid  : (n_t, n_x) meshgrid t
    u_std   : (n_t, n_x) ensemble standard deviation
    save_path : output path
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use log-scale colour mapping to reveal low-uncertainty regions
    # that would otherwise be invisible next to the shock.
    std_plot = u_std.copy()
    std_plot = np.clip(std_plot, 1e-6, None)   # avoid log(0)

    vmin = float(std_plot.min())
    vmax = float(std_plot.max())

    # Guard: if all values are identical (e.g. constant-std smoke test),
    # fall back to a linear scale with a small padding so contourf gets
    # at least two distinct levels.
    if vmax <= vmin * 1.001:
        vmax = vmin * 10.0

    from matplotlib.colors import LogNorm
    norm   = LogNorm(vmin=vmin, vmax=vmax)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 64)
    levels = np.unique(levels)   # drop duplicates that arise with small ranges

    cf = ax.contourf(
        x_grid, t_grid, std_plot,
        levels=levels,
        norm=norm,
        cmap="viridis",
    )
    cbar = fig.colorbar(cf, ax=ax, label="std(u)  [log scale]")
    cbar.ax.tick_params(labelsize=9)
    # Locator that works with log-scale colourbars
    cbar.locator = mticker.LogLocator()
    cbar.update_ticks()

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    ax.set_title("Ensemble uncertainty — std(u(x, t))", fontsize=13)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(T_MIN, T_MAX)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[ensemble_plot] Std heatmap saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot C — time slices with uncertainty bands                         #
# ------------------------------------------------------------------ #

def plot_ensemble_time_slices(
    x_grid:      np.ndarray,
    t_grid:      np.ndarray,
    u_mean:      np.ndarray,
    u_std:       np.ndarray,
    time_slices: list[float] = [0.25, 0.50, 0.75],
    n_sigma:     float = 2.0,
    save_path:   str = "outputs/ensemble/ensemble_time_slices.png",
) -> None:
    """
    For each time slice t* plot:
      - Black solid:  FD reference solution (ground truth)
      - Red dashed:   ensemble mean prediction
      - Red shaded:   mean +/- n_sigma * std (uncertainty band)

    Using n_sigma = 2 gives a ~95% Gaussian prediction interval.

    Parameters
    ----------
    x_grid      : (n_t, n_x) meshgrid x
    t_grid      : (n_t, n_x) meshgrid t
    u_mean      : (n_t, n_x) ensemble mean
    u_std       : (n_t, n_x) ensemble std
    time_slices : list of t values to plot
    n_sigma     : half-width of the shaded uncertainty band in std units
    save_path   : output path
    """
    x_vals = x_grid[0, :]    # (n_x,)
    t_vals = t_grid[:, 0]    # (n_t,)

    print("[ensemble_plot] Computing FD reference for time-slice plot...")
    u_ref_full = _fd_reference(x_vals, t_vals, NU)   # (n_t, n_x)

    n_slices = len(time_slices)
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 4), sharey=True)
    if n_slices == 1:
        axes = [axes]

    for ax, t_star in zip(axes, time_slices):
        idx      = int(np.argmin(np.abs(t_vals - t_star)))
        t_actual = t_vals[idx]

        mean_slice = u_mean[idx, :]   # (n_x,)
        std_slice  = u_std[idx, :]    # (n_x,)
        ref_slice  = u_ref_full[idx, :]

        lower = mean_slice - n_sigma * std_slice
        upper = mean_slice + n_sigma * std_slice

        # Shaded band first (drawn below other lines)
        ax.fill_between(
            x_vals, lower, upper,
            color="#e05c2a", alpha=0.20,
            label=f"mean +/- {n_sigma:.0f}*std",
        )
        ax.plot(x_vals, ref_slice,
                color="#1f2328", linewidth=2.0, label="FD reference")
        ax.plot(x_vals, mean_slice,
                color="#e05c2a", linewidth=1.8, linestyle="--",
                label="Ensemble mean")

        ax.set_title(f"t = {t_actual:.2f}", fontsize=11)
        ax.set_xlabel("x", fontsize=11)
        ax.set_xlim(X_MIN, X_MAX)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("u(x, t)", fontsize=11)
    fig.suptitle(
        f"Ensemble mean +/- {n_sigma:.0f}*std  vs  FD reference",
        fontsize=13, y=1.02,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ensemble_plot] Time-slice plot saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot D — calibration / reliability diagram                          #
# ------------------------------------------------------------------ #

def plot_calibration(
    confidence_levels:  np.ndarray,
    empirical_coverage: np.ndarray,
    ece:                float,
    coverage_90:        float,
    save_path: str = "outputs/ensemble/ensemble_calibration.png",
) -> None:
    """
    Reliability diagram: nominal confidence level (x-axis) vs empirical
    coverage (y-axis).

    A perfectly calibrated model would lie on the diagonal y = x.
    Points above the diagonal indicate the model is over-confident
    (the intervals are too narrow); points below indicate under-confidence.

    The shaded grey region shows the calibration gap at each level.

    Parameters
    ----------
    confidence_levels   : 1-D array of nominal confidence levels (0..1)
    empirical_coverage  : 1-D array of measured coverage at each level
    ece                 : Expected Calibration Error (scalar)
    coverage_90         : empirical 90% coverage probability
    save_path           : output path
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], color="#57606a", linewidth=1.2,
            linestyle="--", label="Perfect calibration")

    # Shaded gap between diagonal and actual
    ax.fill_between(
        confidence_levels,
        confidence_levels,
        empirical_coverage,
        alpha=0.15, color="#3b82d4",
    )

    # Actual calibration curve
    ax.plot(confidence_levels, empirical_coverage,
            color="#3b82d4", linewidth=2.2, marker="o", markersize=4,
            label="Ensemble calibration")

    # Annotate ECE and 90% coverage
    ax.text(
        0.04, 0.88,
        f"ECE = {ece:.4f}\n90% coverage = {coverage_90:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f8fa",
                  edgecolor="#e5e7eb", linewidth=0.8),
    )

    ax.set_xlabel("Nominal confidence level", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Calibration diagram — Deep Ensemble", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[ensemble_plot] Calibration plot saved to '{save_path}'")
