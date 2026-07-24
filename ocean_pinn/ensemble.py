"""
ensemble.py  —  ocean_pinn
---------------------------
Deep Ensemble UQ for the advection-diffusion PINN.

API mirrors burgers_pinn/ensemble.py exactly:
  load_ensemble()      — load M saved checkpoints
  ensemble_predict()   — forward pass → mean + std
  calibration_metrics()— ECE + 90% coverage vs FD reference
"""

import os
import numpy as np
import torch
from model import OceanPINN
from plot import fd_reference
from data import make_evaluation_grid, V, D


def load_ensemble(
    checkpoint_dir: str,
    n_members: int = 10,
    n_hidden: int = 4,
    n_neurons: int = 50,
    device: torch.device = torch.device("cpu"),
    filename_pattern: str = "model_{i}.pt",
) -> list[OceanPINN]:
    """Load M OceanPINN checkpoints and return them in eval() mode."""
    models = []
    for i in range(n_members):
        fname = filename_pattern.format(i=i)
        path  = os.path.join(checkpoint_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Ensemble member {i} not found at '{path}'."
            )
        m = OceanPINN(n_hidden=n_hidden, n_neurons=n_neurons)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        m.eval()
        models.append(m)
    print(f"[ensemble] Loaded {len(models)} models from '{checkpoint_dir}'")
    return models


def ensemble_predict(
    models: list[OceanPINN],
    x_flat: torch.Tensor,
    t_flat: torch.Tensor,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward pass through all M models; return mean, std, all predictions.

    Returns
    -------
    c_mean : (n_t, n_x)
    c_std  : (n_t, n_x)
    c_all  : (M, n_t, n_x)
    """
    n_t, n_x = grid_shape
    preds = []
    with torch.no_grad():
        for m in models:
            c_flat = m(x_flat, t_flat)
            preds.append(c_flat.cpu().numpy().reshape(n_t, n_x))
    c_all  = np.stack(preds, axis=0)
    c_mean = c_all.mean(axis=0)
    c_std  = c_all.std(axis=0, ddof=0)
    return c_mean, c_std, c_all


def calibration_metrics(
    c_mean:  np.ndarray,
    c_std:   np.ndarray,
    x_grid:  np.ndarray,
    t_grid:  np.ndarray,
    n_bins:  int = 20,
    coverage_z: float = 1.645,
) -> dict:
    """
    ECE and 90%-coverage probability vs FD reference.

    Mirrors burgers_pinn/ensemble.calibration_metrics() identically.
    """
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]

    print("[ensemble] Computing FD reference for calibration...")
    c_ref = fd_reference(x_vals, t_vals)   # (n_t, n_x)

    abs_errors = np.abs(c_mean - c_ref)
    err_flat   = abs_errors.ravel()
    std_flat   = c_std.ravel()

    # 90% coverage
    coverage_90 = float((err_flat <= coverage_z * std_flat).mean())

    # ECE
    from scipy.stats import norm
    confidence_levels  = np.linspace(0.05, 0.99, n_bins)
    empirical_coverage = np.zeros(n_bins)
    for k, p in enumerate(confidence_levels):
        z_p = norm.ppf((1.0 + p) / 2.0)
        empirical_coverage[k] = float((err_flat <= z_p * std_flat).mean())

    ece = float(np.mean(np.abs(empirical_coverage - confidence_levels)))
    print(f"[ensemble] ECE = {ece:.4f}  |  90% coverage = {coverage_90:.4f}")

    return {
        "ece":                ece,
        "coverage_90":        coverage_90,
        "confidence_levels":  confidence_levels,
        "empirical_coverage": empirical_coverage,
        "abs_errors":         abs_errors,
    }
