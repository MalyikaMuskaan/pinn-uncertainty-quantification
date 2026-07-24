"""
ensemble.py
-----------
Deep Ensemble approach for the inverse Burgers PINN.

Trains M=10 independent InverseBurgersPINN instances from different random
seeds (different weight initialisations AND different collocation sequences).
Each member recovers its own estimate of ν.

The spread across the M recovered ν values is our uncertainty measure for
the parameter — exactly the same philosophical approach as the forward-problem
ensemble in burgers_pinn/ensemble.py, but now applied to a scalar parameter
rather than a field.

Public API
----------
run_ensemble(cfg) -> EnsembleResult
    Train all members; return aggregated ν statistics and per-member histories.

EnsembleResult (dataclass-like dict)
    nu_estimates   : (M,) array of final ν values
    nu_mean        : scalar mean
    nu_std         : scalar std
    ci_90_lo/hi    : 5th / 95th percentile of the M estimates
    true_in_ci     : bool — does NU_TRUE fall inside the 90% CI?
    nu_histories   : list of nu_history from each member's train() call
    members        : list of trained model objects
"""

import os
import time
import numpy as np
import torch

from data  import NU_TRUE
from train import train as train_single


# ------------------------------------------------------------------ #
#  Default configuration                                               #
# ------------------------------------------------------------------ #

DEFAULT_CFG = dict(
    n_members    = 10,
    n_sensors    = 50,
    noise_frac   = 0.01,
    n_hidden     = 4,
    n_neurons    = 50,
    nu_init      = 0.1,       # ~31× true value  (intentionally wrong)
    n_col        = 10_000,
    n_ic         = 200,
    n_bc         = 200,
    n_epochs     = 8_000,
    lr           = 1e-3,
    print_every  = 500,
    lambda_pde   = 1.0,
    lambda_ic    = 10.0,
    lambda_bc    = 10.0,
    lambda_data  = 100.0,
    out_dir      = "outputs/ensemble",
    device_str   = "auto",
    verbose      = False,   # set True for per-epoch prints
)


# ------------------------------------------------------------------ #
#  Ensemble runner                                                     #
# ------------------------------------------------------------------ #

def run_ensemble(cfg: dict | None = None) -> dict:
    """
    Train M InverseBurgersPINN instances from different seeds.

    Each member uses:
      - seed i  →  torch.manual_seed(i), np.random.seed(i)
      - sensor_seed i  →  different but reproducible sensor draw
        (all members see data from the SAME sensor locations but with
        independently drawn noise realisations)

    Parameters
    ----------
    cfg : configuration dict; defaults applied from DEFAULT_CFG

    Returns
    -------
    result dict with keys:
      nu_estimates, nu_mean, nu_std, ci_90_lo, ci_90_hi,
      true_in_ci, nu_histories, members, train_times
    """
    if cfg is None:
        cfg = {}
    c = {**DEFAULT_CFG, **cfg}

    os.makedirs(c["out_dir"], exist_ok=True)

    if c["device_str"] == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = c["device_str"]

    M = c["n_members"]
    nu_estimates: list[float]                  = []
    nu_histories: list[list[tuple[int,float]]] = []
    members:      list                         = []
    train_times:  list[float]                  = []
    all_loss_histories: list[dict]             = []

    print(f"\n[ensemble] Training {M} inverse PINN members "
          f"(sensors={c['n_sensors']}, noise={c['noise_frac']:.1%}, "
          f"nu_init={c['nu_init']:.4f})")
    print(f"[ensemble] True nu = {NU_TRUE:.6f}")
    print("-" * 65)

    for i in range(M):
        # Reproducible random state per member
        torch.manual_seed(i)
        np.random.seed(i)

        print(f"[ensemble] Member {i+1}/{M}  (seed={i})", end="  ", flush=True)
        t0 = time.time()

        result = train_single(
            n_sensors    = c["n_sensors"],
            noise_frac   = c["noise_frac"],
            sensor_seed  = i,              # each member sees independently noisy data
            n_hidden     = c["n_hidden"],
            n_neurons    = c["n_neurons"],
            nu_init      = c["nu_init"],
            n_col        = c["n_col"],
            n_ic         = c["n_ic"],
            n_bc         = c["n_bc"],
            n_epochs     = c["n_epochs"],
            lr           = c["lr"],
            print_every  = c["print_every"],
            lambda_pde   = c["lambda_pde"],
            lambda_ic    = c["lambda_ic"],
            lambda_bc    = c["lambda_bc"],
            lambda_data  = c["lambda_data"],
            save_path    = os.path.join(c["out_dir"], f"member_{i}.pt"),
            device_str   = device_str,
            verbose      = c["verbose"],
        )

        elapsed = time.time() - t0
        print(f"nu={result['nu_final']:.6f}  ({elapsed:.0f}s)")

        nu_estimates.append(result["nu_final"])
        nu_histories.append(result["nu_history"])
        members.append(result["model"])
        train_times.append(result["train_time"])
        all_loss_histories.append(result["loss_history"])

    nu_arr = np.array(nu_estimates)
    nu_mean  = float(nu_arr.mean())
    nu_std   = float(nu_arr.std(ddof=1))
    ci_lo    = float(np.percentile(nu_arr, 5))
    ci_hi    = float(np.percentile(nu_arr, 95))
    in_ci    = bool(ci_lo <= NU_TRUE <= ci_hi)
    err_pct  = abs(nu_mean - NU_TRUE) / NU_TRUE * 100

    print("-" * 65)
    print(f"[ensemble] nu estimates: {nu_arr}")
    print(f"[ensemble] mean={nu_mean:.6f}  std={nu_std:.6f}  "
          f"90%CI=[{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"[ensemble] true nu={NU_TRUE:.6f}  in CI: {in_ci}  "
          f"mean error: {err_pct:.2f}%")

    return dict(
        nu_estimates        = nu_arr,
        nu_mean             = nu_mean,
        nu_std              = nu_std,
        ci_90_lo            = ci_lo,
        ci_90_hi            = ci_hi,
        true_in_ci          = in_ci,
        nu_histories        = nu_histories,
        members             = members,
        train_times         = train_times,
        all_loss_histories  = all_loss_histories,
        cfg                 = c,
    )
