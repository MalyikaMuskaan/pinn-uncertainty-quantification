"""
dropout_predict.py
------------------
Monte Carlo Dropout inference and calibration for the trained Dropout PINN.

API mirrors ensemble.py and bayesian_predict.py for direct comparison.

Key concept — MC Dropout inference
------------------------------------
After training, we keep the model in **train() mode** instead of switching
to eval().  In train() mode, nn.Dropout layers remain active, so each
forward pass uses a different random dropout mask.  Running N forward passes
gives N different predictions from the same trained network.

This is exactly the "Dropout as a Bayesian Approximation" idea from
Gal & Ghahramani (2016): the distribution over dropout masks approximates
the posterior distribution over network weights.

The mean and standard deviation of the N predictions serve as the
posterior predictive mean and uncertainty estimate — analogous to how
ensemble_predict() averages over 10 separately trained models, and
mc_predict() in bayesian_predict.py averages over 200 weight samples.

Functions
---------
load_dropout_model   : load checkpoint → DropoutBurgersPINN
mc_dropout_predict   : N stochastic forward passes → mean + std
calibration_metrics  : ECE + 90% coverage vs FD reference
evaluate_dropout     : convenience wrapper
"""

import os
import numpy as np
import torch
from dropout_model import DropoutBurgersPINN
from plot import _fd_reference
from data import make_evaluation_grid

NU = 0.01 / 3.141592653589793


# ------------------------------------------------------------------ #
#  Load checkpoint                                                     #
# ------------------------------------------------------------------ #

def load_dropout_model(
    checkpoint_path: str,
    n_hidden:     int   = 4,
    n_neurons:    int   = 50,
    dropout_rate: float = 0.05,
    device: torch.device = torch.device("cpu"),
) -> DropoutBurgersPINN:
    """
    Load a saved DropoutBurgersPINN checkpoint.

    The model is returned in **train() mode** so dropout stays active
    for MC inference.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Dropout checkpoint not found at '{checkpoint_path}'. "
            "Run run_dropout.py first."
        )
    model = DropoutBurgersPINN(
        n_hidden=n_hidden, n_neurons=n_neurons, dropout_rate=dropout_rate
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.enable_mc_dropout()   # keep dropout active for MC inference
    print(f"[dropout_predict] Loaded checkpoint from '{checkpoint_path}' "
          f"(dropout p={dropout_rate}, MC mode=ON)")
    return model


# ------------------------------------------------------------------ #
#  MC Dropout forward pass                                             #
# ------------------------------------------------------------------ #

def mc_dropout_predict(
    model:      DropoutBurgersPINN,
    x_flat:     torch.Tensor,
    t_flat:     torch.Tensor,
    grid_shape: tuple[int, int],
    n_samples:  int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw n_samples stochastic forward passes (each with a different dropout
    mask) and return the mean and std of the resulting predictions.

    Parameters
    ----------
    model       : DropoutBurgersPINN in train() mode (dropout active)
    x_flat      : (N, 1) x coordinates
    t_flat      : (N, 1) t coordinates
    grid_shape  : (n_t, n_x) to reshape flat outputs
    n_samples   : number of MC passes (100 is a good baseline;
                  increase to 200-500 for smoother uncertainty maps)

    Returns
    -------
    u_mean  : (n_t, n_x) predictive mean
    u_std   : (n_t, n_x) predictive std (uncertainty)
    u_all   : (n_samples, n_t, n_x) individual pass predictions

    Note on expected std magnitude
    --------------------------------
    With a small network (4 x 50) and low dropout rate (p=0.05), the
    expected std is typically O(1e-3) to O(1e-2) — much smaller than
    the Deep Ensemble (O(1e-2) to O(1e-1)).  This is not a bug: with
    only ~2.5 neurons dropped per 50-neuron layer, the variation in the
    forward pass is small.  See COMPARISON.md for a detailed discussion.
    """
    n_t, n_x = grid_shape
    preds = []

    # model must be in train() mode for dropout to fire
    assert model.training, (
        "Model must be in train() mode for MC Dropout inference. "
        "Call model.enable_mc_dropout() before predict."
    )

    with torch.no_grad():
        for _ in range(n_samples):
            u_flat = model(x_flat, t_flat)   # dropout mask re-sampled each call
            preds.append(u_flat.cpu().numpy().reshape(n_t, n_x))

    u_all  = np.stack(preds, axis=0)   # (n_samples, n_t, n_x)
    u_mean = u_all.mean(axis=0)
    u_std  = u_all.std(axis=0, ddof=0)

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
    ECE and 90% coverage probability vs FD reference.

    Identical implementation to ensemble.calibration_metrics() and
    bayesian_predict.calibration_metrics() for apples-to-apples comparison.

    Note: if u_std is near-zero everywhere (which can happen with very small
    dropout rate + small network), ECE will be high and coverage low.
    This is faithfully reported as a calibration finding, not suppressed.
    """
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]

    print("[dropout_predict] Computing FD reference for calibration...")
    u_ref = _fd_reference(x_vals, t_vals, NU)

    abs_errors = np.abs(u_mean - u_ref)
    err_flat   = abs_errors.ravel()
    std_flat   = u_std.ravel()

    # 90% coverage
    coverage_90 = float((err_flat <= coverage_z * std_flat).mean())

    # ECE across multiple confidence levels
    from scipy.stats import norm
    confidence_levels  = np.linspace(0.05, 0.99, n_bins)
    empirical_coverage = np.zeros(n_bins)
    for k, p in enumerate(confidence_levels):
        z_p    = norm.ppf((1.0 + p) / 2.0)
        inside = err_flat <= z_p * std_flat
        empirical_coverage[k] = float(inside.mean())

    ece = float(np.mean(np.abs(empirical_coverage - confidence_levels)))

    # Diagnostic: report std statistics so the caller can flag near-zero cases
    std_max  = float(std_flat.max())
    std_mean = float(std_flat.mean())
    if std_max < 1e-3:
        print(f"[dropout_predict] WARNING: std is near-zero (max={std_max:.2e}). "
              f"This may indicate that p={u_std.mean():.0e} is too low for "
              f"this network size to produce meaningful uncertainty. "
              f"See COMPARISON.md for interpretation.")

    print(f"[dropout_predict] ECE={ece:.4f}  90% cov={coverage_90:.4f}  "
          f"std max={std_max:.3e}  std mean={std_mean:.3e}")

    return {
        "ece":                ece,
        "coverage_90":        coverage_90,
        "confidence_levels":  confidence_levels,
        "empirical_coverage": empirical_coverage,
        "abs_errors":         abs_errors,
        "std_max":            std_max,
        "std_mean":           std_mean,
    }


# ------------------------------------------------------------------ #
#  Convenience wrapper                                                 #
# ------------------------------------------------------------------ #

def evaluate_dropout(
    model:     DropoutBurgersPINN,
    device:    torch.device,
    n_x:       int = 256,
    n_t:       int = 100,
    n_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Full MC Dropout evaluation on a dense grid.

    Returns
    -------
    x_grid, t_grid, u_mean, u_std, u_all, metrics
    """
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(n_x, n_t, device)

    print(f"[dropout_predict] Running {n_samples} MC Dropout passes "
          f"(dropout p={model.dropout_rate})...")
    u_mean, u_std, u_all = mc_dropout_predict(
        model, x_flat, t_flat,
        grid_shape=(n_t, n_x),
        n_samples=n_samples,
    )
    print(f"[dropout_predict] mean [{u_mean.min():.3f}, {u_mean.max():.3f}]  "
          f"std [{u_std.min():.3e}, {u_std.max():.3e}]")

    metrics = calibration_metrics(u_mean, u_std, x_grid, t_grid)
    return x_grid, t_grid, u_mean, u_std, u_all, metrics
