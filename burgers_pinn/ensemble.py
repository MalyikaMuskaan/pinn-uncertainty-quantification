"""
ensemble.py
-----------
Deep Ensemble Uncertainty Quantification for the Burgers' PINN.

Background
----------
A Deep Ensemble (Lakshminarayanan et al., 2017) trains M independent models
with different random seeds — different weight initialisations AND different
stochastic collocation-point sequences during training.  At inference time the
M predictions are treated as samples from an implicit mixture distribution:

    mean(x,t)  = (1/M) * sum_m  u_m(x,t)
    var(x,t)   = (1/M) * sum_m  [u_m(x,t) - mean(x,t)]^2

The standard deviation std = sqrt(var) is the ensemble uncertainty estimate.

This file provides:
  - load_ensemble()        : load M saved checkpoints -> list of models
  - ensemble_predict()     : forward pass through all M models, returns mean + std
  - calibration_metrics()  : ECE and 90%-coverage probability vs FD reference
"""

import os
import numpy as np
import torch
from model import BurgersPINN
from plot import _fd_reference       # Crank-Nicolson reference solver
from data import make_evaluation_grid

# Physical constant (must match train.py)
NU = 0.01 / 3.141592653589793


# ------------------------------------------------------------------ #
#  1.  Loading a saved ensemble                                        #
# ------------------------------------------------------------------ #

def load_ensemble(
    checkpoint_dir: str,
    n_members: int = 10,
    n_hidden: int = 4,
    n_neurons: int = 50,
    device: torch.device = torch.device("cpu"),
    filename_pattern: str = "model_{i}.pt",
) -> list[BurgersPINN]:
    """
    Load M independent PINN checkpoints from disk and return them as a list.

    Parameters
    ----------
    checkpoint_dir   : directory containing the saved .pt files
    n_members        : number of ensemble members M
    n_hidden         : hidden layers (must match saved architecture)
    n_neurons        : neurons per layer (must match saved architecture)
    device           : torch device for inference
    filename_pattern : filename template, {i} is replaced by member index 0..M-1

    Returns
    -------
    models : list of M BurgersPINN instances in eval() mode
    """
    models = []
    for i in range(n_members):
        fname = filename_pattern.format(i=i)
        path  = os.path.join(checkpoint_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Ensemble member {i} not found at '{path}'. "
                f"Run run_ensemble.py to train the ensemble first."
            )
        m = BurgersPINN(n_hidden=n_hidden, n_neurons=n_neurons)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        m.eval()
        models.append(m)

    print(f"[ensemble] Loaded {len(models)} models from '{checkpoint_dir}'")
    return models


# ------------------------------------------------------------------ #
#  2.  Ensemble forward pass                                           #
# ------------------------------------------------------------------ #

def ensemble_predict(
    models: list[BurgersPINN],
    x_flat: torch.Tensor,
    t_flat: torch.Tensor,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run every ensemble member on the same (x, t) inputs and compute
    the pointwise mean and standard deviation.

    Parameters
    ----------
    models     : list of M models returned by load_ensemble()
    x_flat     : (N, 1) tensor of x coordinates
    t_flat     : (N, 1) tensor of t coordinates
    grid_shape : (n_t, n_x) shape to reshape the flat outputs into

    Returns
    -------
    u_mean  : (n_t, n_x) array — ensemble mean prediction
    u_std   : (n_t, n_x) array — ensemble standard deviation (uncertainty)
    u_all   : (M, n_t, n_x) array — individual member predictions
    """
    n_t, n_x = grid_shape
    M = len(models)

    # Collect each member's flat prediction; no gradients needed here
    preds = []
    with torch.no_grad():
        for m in models:
            u_flat = m(x_flat, t_flat)          # (N, 1)
            preds.append(u_flat.cpu().numpy().reshape(n_t, n_x))

    u_all  = np.stack(preds, axis=0)            # (M, n_t, n_x)
    u_mean = u_all.mean(axis=0)                 # (n_t, n_x)
    u_std  = u_all.std(axis=0, ddof=0)          # (n_t, n_x)  population std

    return u_mean, u_std, u_all


# ------------------------------------------------------------------ #
#  3.  Calibration metrics                                             #
# ------------------------------------------------------------------ #

def calibration_metrics(
    u_mean: np.ndarray,
    u_std: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    n_bins: int = 20,
    coverage_z: float = 1.645,   # z-score for 90% Gaussian interval
) -> dict:
    """
    Compare the ensemble's uncertainty estimates against the actual errors
    measured relative to the Crank-Nicolson FD reference solution.

    Two metrics are computed:

    Expected Calibration Error (ECE)
    ---------------------------------
    We form confidence intervals of varying widths (z * std) and check what
    fraction of the true values fall inside.  ECE is the mean absolute
    difference between the expected coverage (confidence level) and the
    empirical coverage across n_bins confidence levels from 0 to 1.

    90% Coverage Probability
    ------------------------
    The fraction of domain points where the true value lies inside the
    90% prediction interval  [mean - 1.645*std, mean + 1.645*std].
    A perfectly calibrated model gives 0.90.

    Parameters
    ----------
    u_mean    : (n_t, n_x) ensemble mean
    u_std     : (n_t, n_x) ensemble std (uncertainty)
    x_grid    : (n_t, n_x) meshgrid x array (for FD reference)
    t_grid    : (n_t, n_x) meshgrid t array
    n_bins    : number of confidence levels for ECE calculation
    coverage_z: z-score for the coverage check (1.645 -> 90%)

    Returns
    -------
    metrics dict with keys:
      "ece"              : float — Expected Calibration Error
      "coverage_90"      : float — empirical 90% coverage probability
      "confidence_levels": 1-D array of nominal confidence levels
      "empirical_coverage": 1-D array of empirical coverages at each level
      "abs_errors"       : (n_t, n_x) absolute error vs FD reference
    """
    x_vals = x_grid[0, :]   # 1-D
    t_vals = t_grid[:, 0]   # 1-D

    # --- Compute FD reference on the same grid ---
    print("[ensemble] Computing FD reference for calibration (this may take ~10s)...")
    u_ref = _fd_reference(x_vals, t_vals, NU)   # (n_t, n_x)

    # Absolute error at every grid point
    abs_errors = np.abs(u_mean - u_ref)         # (n_t, n_x)

    # Flatten for vectorised statistics
    err_flat = abs_errors.ravel()               # (N,)
    std_flat = u_std.ravel()                    # (N,)

    # --- 90% coverage ---
    inside_90 = err_flat <= coverage_z * std_flat
    coverage_90 = float(inside_90.mean())

    # --- ECE across multiple confidence levels ---
    # For a Gaussian(mean, std) model the z-score for a two-sided
    # confidence level p is:  z(p) = ppf((1+p)/2)
    from scipy.stats import norm
    confidence_levels = np.linspace(0.05, 0.99, n_bins)
    empirical_coverage = np.zeros(n_bins)

    for k, p in enumerate(confidence_levels):
        z_p = norm.ppf((1.0 + p) / 2.0)                 # two-sided z
        inside = err_flat <= z_p * std_flat
        empirical_coverage[k] = float(inside.mean())

    ece = float(np.mean(np.abs(empirical_coverage - confidence_levels)))

    metrics = {
        "ece":               ece,
        "coverage_90":       coverage_90,
        "confidence_levels": confidence_levels,
        "empirical_coverage": empirical_coverage,
        "abs_errors":        abs_errors,
    }

    print(f"[ensemble] ECE = {ece:.4f}  |  90% coverage = {coverage_90:.4f}")
    return metrics
