"""
compare_methods.py
------------------
Side-by-side comparison of all three UQ methods:
  1. Deep Ensemble    (10 independently trained models)
  2. Bayesian PINN    (mean-field VI / Bayes by Backprop)
  3. MC Dropout       (single trained model, stochastic inference)

Produces
--------
1. outputs/comparison/comparison_table.csv
   One row per method: MSE, ECE, 90% coverage, train time, inference time

2. outputs/comparison/uncertainty_comparison.png
   Three-panel figure with a SHARED colourbar, log-scale std heatmap for
   each method.

3. outputs/comparison/calibration_comparison.png
   All three calibration curves on one axes for direct visual comparison.

Prerequisites
-------------
- run_ensemble.py   → outputs/ensemble/
- run_bayesian.py   → outputs/bayesian/
- run_dropout.py    → outputs/dropout/
"""

import os
import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import torch

from ensemble import load_ensemble, ensemble_predict, calibration_metrics as ens_cal
from bayesian_predict import load_bayesian_model, mc_predict, calibration_metrics as bay_cal
from dropout_predict import load_dropout_model, mc_dropout_predict, calibration_metrics as drop_cal
from data import make_evaluation_grid, X_MIN, X_MAX, T_MIN, T_MAX
from plot import _fd_reference

NU = 0.01 / math.pi

# ================================================================== #
#  Config                                                              #
# ================================================================== #
CFG = {
    # --- Ensemble ---
    "ensemble_dir":     "outputs/ensemble",
    "ensemble_members": 10,
    "ensemble_metrics": "outputs/ensemble/ensemble_metrics.json",

    # --- Bayesian ---
    "bayes_ckpt":    "outputs/bayesian/bayesian_pinn.pt",
    "bayes_metrics": "outputs/bayesian/bayesian_metrics.json",

    # --- Dropout ---
    "dropout_ckpt":    "outputs/dropout/dropout_pinn.pt",
    "dropout_metrics": "outputs/dropout/dropout_metrics.json",
    "dropout_rate":    0.05,

    # --- Shared ---
    "out_dir":    "outputs/comparison",
    "n_hidden":   4,
    "n_neurons":  50,
    "n_x":        256,
    "n_t":        100,
    "n_mc_bayes": 200,
    "n_mc_drop":  100,
    "device":     "auto",
}


def resolve_device(cfg):
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


# ================================================================== #
#  Per-method loaders                                                  #
# ================================================================== #

def _load_ensemble(cfg, device, x_flat, t_flat, x_grid, t_grid):
    print("[compare] Loading Deep Ensemble models...")
    models = load_ensemble(
        cfg["ensemble_dir"], cfg["ensemble_members"],
        cfg["n_hidden"], cfg["n_neurons"], device,
    )
    n_t, n_x = cfg["n_t"], cfg["n_x"]
    t0 = time.time()
    u_mean, u_std, _ = ensemble_predict(models, x_flat, t_flat, (n_t, n_x))
    inf_time = time.time() - t0
    metrics  = ens_cal(u_mean, u_std, x_grid, t_grid)
    u_ref    = _fd_reference(x_grid[0, :], t_grid[:, 0], NU)
    mse      = float(np.mean((u_mean - u_ref) ** 2))
    train_time = float("nan")
    if os.path.isfile(cfg["ensemble_metrics"]):
        with open(cfg["ensemble_metrics"]) as f:
            train_time = json.load(f).get("train_time_s", float("nan"))
    return dict(u_mean=u_mean, u_std=u_std, metrics=metrics,
                mse=mse, train_time=train_time, inf_time=inf_time)


def _load_bayesian(cfg, device, x_flat, t_flat, x_grid, t_grid):
    print("[compare] Loading Bayesian PINN...")
    model = load_bayesian_model(
        cfg["bayes_ckpt"], cfg["n_hidden"], cfg["n_neurons"], device
    )
    n_t, n_x = cfg["n_t"], cfg["n_x"]
    t0 = time.time()
    u_mean, u_std, _ = mc_predict(model, x_flat, t_flat, (n_t, n_x),
                                  n_samples=cfg["n_mc_bayes"])
    inf_time = time.time() - t0
    metrics  = bay_cal(u_mean, u_std, x_grid, t_grid)
    u_ref    = _fd_reference(x_grid[0, :], t_grid[:, 0], NU)
    mse      = float(np.mean((u_mean - u_ref) ** 2))
    train_time = float("nan")
    if os.path.isfile(cfg["bayes_metrics"]):
        with open(cfg["bayes_metrics"]) as f:
            train_time = json.load(f).get("train_time_s", float("nan"))
    return dict(u_mean=u_mean, u_std=u_std, metrics=metrics,
                mse=mse, train_time=train_time, inf_time=inf_time)


def _load_dropout(cfg, device, x_flat, t_flat, x_grid, t_grid):
    print("[compare] Loading MC Dropout PINN...")
    model = load_dropout_model(
        cfg["dropout_ckpt"], cfg["n_hidden"], cfg["n_neurons"],
        cfg["dropout_rate"], device,
    )
    n_t, n_x = cfg["n_t"], cfg["n_x"]
    t0 = time.time()
    u_mean, u_std, _ = mc_dropout_predict(model, x_flat, t_flat, (n_t, n_x),
                                          n_samples=cfg["n_mc_drop"])
    inf_time = time.time() - t0
    metrics  = drop_cal(u_mean, u_std, x_grid, t_grid)
    u_ref    = _fd_reference(x_grid[0, :], t_grid[:, 0], NU)
    mse      = float(np.mean((u_mean - u_ref) ** 2))
    train_time = float("nan")
    if os.path.isfile(cfg["dropout_metrics"]):
        with open(cfg["dropout_metrics"]) as f:
            train_time = json.load(f).get("train_time_s", float("nan"))
    return dict(u_mean=u_mean, u_std=u_std, metrics=metrics,
                mse=mse, train_time=train_time, inf_time=inf_time)


# ================================================================== #
#  Plot A — three-panel uncertainty heatmap (shared colourbar)         #
# ================================================================== #

def plot_uncertainty_comparison(
    x_grid:    np.ndarray,
    t_grid:    np.ndarray,
    ens_std:   np.ndarray,
    bay_std:   np.ndarray,
    drop_std:  np.ndarray,
    save_path: str = "outputs/comparison/uncertainty_comparison.png",
) -> None:
    """
    Three-panel log-scale std heatmaps on a SHARED colour scale.

    The shared scale is crucial: because MC Dropout produces much smaller
    std than the Ensemble, a shared scale immediately communicates the
    relative magnitude of uncertainty across all three methods.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)

    # Combined range across all three methods
    combined = np.concatenate([
        np.clip(ens_std.ravel(),  1e-6, None),
        np.clip(bay_std.ravel(),  1e-6, None),
        np.clip(drop_std.ravel(), 1e-6, None),
    ])
    vmin = float(combined.min())
    vmax = float(combined.max())
    if vmax <= vmin * 1.001:
        vmax = vmin * 10.0

    norm   = LogNorm(vmin=vmin, vmax=vmax)
    levels = np.unique(np.logspace(np.log10(vmin), np.log10(vmax), 64))

    labels = [
        "Deep Ensemble\n(10 members)",
        "Bayesian PINN (VI)\n(200 MC samples)",
        "MC Dropout\n(100 passes, p=0.05)",
    ]
    for ax, u_std, label in zip(axes, [ens_std, bay_std, drop_std], labels):
        std_plot = np.clip(u_std, 1e-6, None)
        cf = ax.contourf(x_grid, t_grid, std_plot,
                         levels=levels, norm=norm, cmap="viridis")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("t", fontsize=10)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(T_MIN, T_MAX)

    cbar = fig.colorbar(cf, ax=axes.tolist(), label="std(u)  [log scale, shared]",
                        shrink=0.75)
    cbar.ax.tick_params(labelsize=8)
    cbar.locator = mticker.LogLocator()
    cbar.update_ticks()

    fig.suptitle("Uncertainty comparison — std(u(x, t))  [shared colour scale]",
                 fontsize=13)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[compare] 3-panel uncertainty comparison saved to '{save_path}'")


# ================================================================== #
#  Plot B — three-curve calibration diagram                            #
# ================================================================== #

def plot_calibration_comparison(
    ens_conf:  np.ndarray, ens_cov:  np.ndarray, ens_ece:  float,
    bay_conf:  np.ndarray, bay_cov:  np.ndarray, bay_ece:  float,
    drop_conf: np.ndarray, drop_cov: np.ndarray, drop_ece: float,
    save_path: str = "outputs/comparison/calibration_comparison.png",
) -> None:
    """All three calibration curves on one axes."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], color="#57606a", lw=1.2, ls="--",
            label="Perfect calibration")

    # Deep Ensemble
    ax.fill_between(ens_conf, ens_conf, ens_cov, alpha=0.08, color="#3b82d4")
    ax.plot(ens_conf, ens_cov, color="#3b82d4", lw=2.2,
            marker="o", markersize=3,
            label=f"Deep Ensemble  (ECE={ens_ece:.4f})")

    # Bayesian PINN
    ax.fill_between(bay_conf, bay_conf, bay_cov, alpha=0.08, color="#7c5cd8")
    ax.plot(bay_conf, bay_cov, color="#7c5cd8", lw=2.2,
            marker="s", markersize=3, ls="-.",
            label=f"Bayesian PINN  (ECE={bay_ece:.4f})")

    # MC Dropout
    ax.fill_between(drop_conf, drop_conf, drop_cov, alpha=0.08, color="#e05c2a")
    ax.plot(drop_conf, drop_cov, color="#e05c2a", lw=2.2,
            marker="^", markersize=3, ls=":",
            label=f"MC Dropout     (ECE={drop_ece:.4f})")

    ax.set_xlabel("Nominal confidence level", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Calibration comparison — all three UQ methods", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[compare] 3-method calibration comparison saved to '{save_path}'")


# ================================================================== #
#  Summary table                                                       #
# ================================================================== #

def build_comparison_table(ens: dict, bay: dict, drop: dict) -> pd.DataFrame:
    def fmt_time(t_s):
        return f"{t_s/60:.1f}" if not (isinstance(t_s, float) and
                                       (t_s != t_s)) else "N/A"

    rows = [
        {
            "Method":             "Deep Ensemble (10 members)",
            "MSE":                f"{ens['mse']:.4e}",
            "ECE":                f"{ens['metrics']['ece']:.4f}",
            "90% Coverage":       f"{ens['metrics']['coverage_90']:.4f}",
            "Std max":            f"{ens['u_std'].max():.3e}",
            "Train time (min)":   fmt_time(ens["train_time"]),
            "Inference time (s)": f"{ens['inf_time']:.1f}",
        },
        {
            "Method":             "Bayesian PINN (VI)",
            "MSE":                f"{bay['mse']:.4e}",
            "ECE":                f"{bay['metrics']['ece']:.4f}",
            "90% Coverage":       f"{bay['metrics']['coverage_90']:.4f}",
            "Std max":            f"{bay['u_std'].max():.3e}",
            "Train time (min)":   fmt_time(bay["train_time"]),
            "Inference time (s)": f"{bay['inf_time']:.1f}",
        },
        {
            "Method":             "MC Dropout (p=0.05)",
            "MSE":                f"{drop['mse']:.4e}",
            "ECE":                f"{drop['metrics']['ece']:.4f}",
            "90% Coverage":       f"{drop['metrics']['coverage_90']:.4f}",
            "Std max":            f"{drop['u_std'].max():.3e}",
            "Train time (min)":   fmt_time(drop["train_time"]),
            "Inference time (s)": f"{drop['inf_time']:.1f}",
        },
    ]
    return pd.DataFrame(rows)


# ================================================================== #
#  Entry point                                                         #
# ================================================================== #

def main():
    cfg    = CFG
    device = resolve_device(cfg)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print(f"[compare] Device: {device}")

    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(
        cfg["n_x"], cfg["n_t"], device
    )

    ens  = _load_ensemble(cfg, device, x_flat, t_flat, x_grid, t_grid)
    bay  = _load_bayesian(cfg, device, x_flat, t_flat, x_grid, t_grid)
    drop = _load_dropout( cfg, device, x_flat, t_flat, x_grid, t_grid)

    # Plot A — 3-panel uncertainty
    plot_uncertainty_comparison(
        x_grid, t_grid,
        ens["u_std"], bay["u_std"], drop["u_std"],
        save_path=os.path.join(cfg["out_dir"], "uncertainty_comparison.png"),
    )

    # Plot B — 3-curve calibration
    plot_calibration_comparison(
        ens["metrics"]["confidence_levels"],
        ens["metrics"]["empirical_coverage"],
        ens["metrics"]["ece"],
        bay["metrics"]["confidence_levels"],
        bay["metrics"]["empirical_coverage"],
        bay["metrics"]["ece"],
        drop["metrics"]["confidence_levels"],
        drop["metrics"]["empirical_coverage"],
        drop["metrics"]["ece"],
        save_path=os.path.join(cfg["out_dir"], "calibration_comparison.png"),
    )

    # Table
    df       = build_comparison_table(ens, bay, drop)
    csv_path = os.path.join(cfg["out_dir"], "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"[compare] CSV saved to '{csv_path}'")

    print()
    print("=" * 70)
    print("  COMPARISON TABLE — ALL THREE UQ METHODS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    expected = ["uncertainty_comparison.png", "calibration_comparison.png",
                "comparison_table.csv"]
    print()
    print("  SAVED FILES")
    print("  " + "-" * 55)
    for fname in expected:
        fpath  = os.path.join(cfg["out_dir"], fname)
        exists = os.path.isfile(fpath)
        status = "OK" if exists else "MISSING"
        size_s = f"  ({os.path.getsize(fpath)/1024:.1f} KB)" if exists else ""
        print(f"  [{status}]  {fpath}{size_s}")
    print("=" * 70)


if __name__ == "__main__":
    main()
