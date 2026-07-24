"""
bayesian_predict.py
-------------------
Monte Carlo inference and calibration for the trained Bayesian PINN.

API mirrors ensemble.py so results are directly comparable.

Functions
---------
load_bayesian_model   : load a saved BayesianBurgersPINN checkpoint
mc_predict            : draw N weight samples → pointwise mean + std
calibration_metrics   : ECE + 90% coverage vs FD reference
evaluate_bayesian     : convenience wrapper (load grid + run all above)
"""

import os
import numpy as np
import torch
from bayesian_model import BayesianBurgersPINN
from plot import _fd_reference
from data import make_evaluation_grid

NU = 0.01 / 3.141592653589793


# ------------------------------------------------------------------ #
#  Load checkpoint                                                     #
# ------------------------------------------------------------------ #

def load_bayesian_model(
    checkpoint_path: str,
    n_hidden:  int = 4,
    n_neurons: int = 50,
    device:    torch.device = torch.device("cpu"),
) -> BayesianBurgersPINN:
    """
    Load a saved BayesianBurgersPINN state dict and return the model in
    eval mode.

    Parameters
    ----------
    checkpoint_path : path to the .pt file saved by bayesian_train.py
    n_hidden        : hidden layers (must match saved architecture)
    n_neurons       : neurons per layer
    device          : torch device

    Returns
    -------
    model : BayesianBurgersPINN in eval() mode
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Bayesian checkpoint not found at '{checkpoint_path}'. "
            "Run run_bayesian.py first."
        )
    model = BayesianBurgersPINN(n_hidden=n_hidden, n_neurons=n_neurons)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[bayes_predict] Loaded checkpoint from '{checkpoint_path}'")
    return model


# ------------------------------------------------------------------ #
#  Monte Carlo forward pass                                            #
# ------------------------------------------------------------------ #

def mc_predict(
    model:      BayesianBurgersPINN,
    x_flat:     torch.Tensor,
    t_flat:     torch.Tensor,
    grid_shape: tuple[int, int],
    n_samples:  int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw n_samples independent weight samples from the posterior and
    collect the resulting predictions.  Mean and std of those samples
    form the posterior predictive mean and uncertainty.

    This directly mirrors ensemble.ensemble_predict(): each weight draw
    plays the same role as one ensemble member.

    Parameters
    ----------
    model       : trained BayesianBurgersPINN
    x_flat      : (N, 1) x coordinates
    t_flat      : (N, 1) t coordinates
    grid_shape  : (n_t, n_x) to reshape the N flat outputs
    n_samples   : number of MC samples (200 recommended for calibration)

    Returns
    -------
    u_mean  : (n_t, n_x) posterior predictive mean
    u_std   : (n_t, n_x) posterior predictive std (uncertainty)
    u_all   : (n_samples, n_t, n_x) individual sample predictions
    """
    n_t, n_x = grid_shape
    preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            # Each call to model() draws a fresh weight sample
            u_flat, _ = model(x_flat, t_flat)           # (N, 1), scalar KL
            preds.append(u_flat.cpu().numpy().reshape(n_t, n_x))

    u_all  = np.stack(preds, axis=0)      # (n_samples, n_t, n_x)
    u_mean = u_all.mean(axis=0)
    u_std  = u_all.std(axis=0, ddof=0)   # population std

    return u_mean, u_std, u_all


# ------------------------------------------------------------------ #
#  Calibration metrics                                                 #
# ------------------------------------------------------------------ #

def calibration_metrics(
    u_mean: np.ndarray,
    u_std:  np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    n_bins: int = 20,
    coverage_z: float = 1.645,
) -> dict:
    """
    ECE and 90% coverage probability vs the Crank-Nicolson FD reference.

    Identical implementation to ensemble.calibration_metrics() so results
    are on the same footing.

    Parameters
    ----------
    u_mean      : (n_t, n_x) posterior predictive mean
    u_std       : (n_t, n_x) posterior predictive std
    x_grid      : (n_t, n_x) meshgrid x
    t_grid      : (n_t, n_x) meshgrid t
    n_bins      : confidence levels for ECE
    coverage_z  : z-score for the named coverage check (1.645 → 90%)

    Returns
    -------
    dict with keys: ece, coverage_90, confidence_levels,
                    empirical_coverage, abs_errors
    """
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]

    print("[bayes_predict] Computing FD reference for calibration...")
    u_ref = _fd_reference(x_vals, t_vals, NU)    # (n_t, n_x)

    abs_errors = np.abs(u_mean - u_ref)
    err_flat   = abs_errors.ravel()
    std_flat   = u_std.ravel()

    # 90% coverage
    inside_90  = err_flat <= coverage_z * std_flat
    coverage_90 = float(inside_90.mean())

    # ECE across multiple confidence levels
    from scipy.stats import norm
    confidence_levels  = np.linspace(0.05, 0.99, n_bins)
    empirical_coverage = np.zeros(n_bins)

    for k, p in enumerate(confidence_levels):
        z_p    = norm.ppf((1.0 + p) / 2.0)
        inside = err_flat <= z_p * std_flat
        empirical_coverage[k] = float(inside.mean())

    ece = float(np.mean(np.abs(empirical_coverage - confidence_levels)))

    print(f"[bayes_predict] ECE = {ece:.4f}  |  90% coverage = {coverage_90:.4f}")
    return {
        "ece":                ece,
        "coverage_90":        coverage_90,
        "confidence_levels":  confidence_levels,
        "empirical_coverage": empirical_coverage,
        "abs_errors":         abs_errors,
    }


# ------------------------------------------------------------------ #
#  Convenience wrapper                                                 #
# ------------------------------------------------------------------ #

def evaluate_bayesian(
    model:     BayesianBurgersPINN,
    device:    torch.device,
    n_x:       int = 256,
    n_t:       int = 100,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run MC sampling on a dense evaluation grid and compute calibration.

    Returns
    -------
    x_grid, t_grid, u_mean, u_std, u_all, metrics
    """
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(n_x, n_t, device)

    print(f"[bayes_predict] Drawing {n_samples} MC samples...")
    u_mean, u_std, u_all = mc_predict(
        model, x_flat, t_flat,
        grid_shape=(n_t, n_x),
        n_samples=n_samples,
    )
    print(f"[bayes_predict] mean range [{u_mean.min():.3f}, {u_mean.max():.3f}]  "
          f"std range [{u_std.min():.4e}, {u_std.max():.4e}]")

    metrics = calibration_metrics(u_mean, u_std, x_grid, t_grid)
    return x_grid, t_grid, u_mean, u_std, u_all, metrics
