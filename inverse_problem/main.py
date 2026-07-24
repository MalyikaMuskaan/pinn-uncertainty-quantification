"""
main.py
-------
Full pipeline for the inverse Burgers PINN experiment.

Steps
-----
1. Train a 10-member Deep Ensemble; each member independently recovers ν.
2. Plot ν convergence curves (Plot A) and solution comparison (Plot B).
3. Run a robustness sweep across 3 noise levels × 3 sensor-count settings.
4. Plot robustness summary box-plots (Plot C), table (Plot D), and
   ν-distribution histograms (Plot E).
5. Write a JSON metrics file.

Run
---
    cd inverse_problem
    python main.py

All outputs land in  inverse_problem/outputs/.
"""

import os
import sys
import json
import time
import numpy as np
import torch

# Ensure inverse_problem/ is on the path when run from another directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data       import NU_TRUE
from ensemble   import run_ensemble, DEFAULT_CFG
from robustness import run_robustness_sweep
from plot       import (
    plot_nu_convergence,
    plot_solution_comparison,
    plot_robustness_summary,
    plot_robustness_table,
    plot_nu_distributions,
)

# ================================================================== #
#  Configuration                                                       #
# ================================================================== #

# Paths
OUT_DIR        = "outputs"
ENS_DIR        = os.path.join(OUT_DIR, "ensemble")
ROB_DIR        = os.path.join(OUT_DIR, "robustness")
METRICS_PATH   = os.path.join(OUT_DIR, "metrics.json")

# Architecture (keep identical to burgers_pinn baseline for fair comparison)
N_HIDDEN  = 4
N_NEURONS = 50

# Inverse-problem specific
NU_INIT      = 0.10       # initial guess ≈ 31× true value (0.003183)
N_SENSORS    = 50         # default sensor count
NOISE_FRAC   = 0.01       # default noise level  (1 % of signal range)
N_EPOCHS     = 8_000
N_MEMBERS    = 10
LR           = 1e-3
LAMBDA_DATA  = 100.0      # data loss weight (up-weighted vs PDE terms)

# Robustness sweep axes
NOISE_LEVELS  = [0.005, 0.01, 0.02]    # 0.5%, 1%, 2%
SENSOR_COUNTS = [20, 50, 100]

# Device
DEVICE_STR = "auto"


# ================================================================== #
#  Utility                                                             #
# ================================================================== #

def _device():
    if DEVICE_STR == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE_STR)


def _header(msg):
    bar = "=" * 68
    print(f"\n{bar}\n  {msg}\n{bar}")


# ================================================================== #
#  Step 1 — Main ensemble (default noise + sensor settings)            #
# ================================================================== #

def step_main_ensemble() -> dict:
    _header("STEP 1 -- Main ensemble: recovering nu from sparse noisy data")

    cfg = {
        **DEFAULT_CFG,
        "n_members":   N_MEMBERS,
        "n_sensors":   N_SENSORS,
        "noise_frac":  NOISE_FRAC,
        "n_hidden":    N_HIDDEN,
        "n_neurons":   N_NEURONS,
        "nu_init":     NU_INIT,
        "n_epochs":    N_EPOCHS,
        "lr":          LR,
        "lambda_data": LAMBDA_DATA,
        "out_dir":     ENS_DIR,
        "device_str":  DEVICE_STR,
        "verbose":     True,
        "print_every": 1000,
    }

    result = run_ensemble(cfg)

    # --- Summarise ---
    print(f"\n  nu true  = {NU_TRUE:.6f}")
    print(f"  nu mean  = {result['nu_mean']:.6f}")
    print(f"  nu std   = {result['nu_std']:.6f}")
    print(f"  90% CI   = [{result['ci_90_lo']:.6f}, {result['ci_90_hi']:.6f}]")
    print(f"  true in CI: {result['true_in_ci']}")
    err_pct = abs(result["nu_mean"] - NU_TRUE) / NU_TRUE * 100
    print(f"  mean error: {err_pct:.2f}%")

    return result


# ================================================================== #
#  Step 2 — Plots for the main ensemble                                #
# ================================================================== #

def step_main_plots(ens_result: dict):
    _header("STEP 2 — Generating main ensemble plots")

    device = _device()

    # Plot A -- nu convergence
    plot_nu_convergence(
        nu_histories = ens_result["nu_histories"],
        save_path    = os.path.join(OUT_DIR, "nu_convergence.png"),
        title_suffix = (f"sensors={N_SENSORS}, noise={NOISE_FRAC:.1%}, "
                        f"nu0={NU_INIT:.4f}, M={N_MEMBERS} members"),
    )

    # Plot B — solution comparison
    plot_solution_comparison(
        members    = ens_result["members"],
        nu_learned = ens_result["nu_mean"],
        device     = device,
        save_path  = os.path.join(OUT_DIR, "solution_comparison.png"),
    )


# ================================================================== #
#  Step 3 — Robustness sweep                                           #
# ================================================================== #

def step_robustness() -> list[dict]:
    _header("STEP 3 — Robustness sweep (noise × sparsity)")

    results = run_robustness_sweep(
        noise_levels  = NOISE_LEVELS,
        sensor_counts = SENSOR_COUNTS,
        n_members     = N_MEMBERS,
        n_hidden      = N_HIDDEN,
        n_neurons     = N_NEURONS,
        nu_init       = NU_INIT,
        n_epochs      = N_EPOCHS,
        lr            = LR,
        lambda_data   = LAMBDA_DATA,
        out_dir       = ROB_DIR,
        device_str    = DEVICE_STR,
        print_every   = 1000,
    )
    return results


# ================================================================== #
#  Step 4 — Robustness plots                                           #
# ================================================================== #

def step_robustness_plots(rob_results: list[dict]):
    _header("STEP 4 — Generating robustness plots")

    plot_robustness_summary(
        rob_results,
        save_path = os.path.join(OUT_DIR, "robustness_summary.png"),
    )
    plot_robustness_table(
        rob_results,
        save_path = os.path.join(OUT_DIR, "robustness_table.png"),
    )
    plot_nu_distributions(
        rob_results,
        save_path = os.path.join(OUT_DIR, "nu_distributions.png"),
    )


# ================================================================== #
#  Step 5 — Save metrics JSON                                          #
# ================================================================== #

def step_save_metrics(ens_result: dict, rob_results: list[dict]):
    _header("STEP 5 — Saving metrics")
    os.makedirs(OUT_DIR, exist_ok=True)

    err_pct = abs(ens_result["nu_mean"] - NU_TRUE) / NU_TRUE * 100

    # Serialisable robustness rows (drop nu_histories)
    rob_rows = [
        {k: v for k, v in r.items() if k != "nu_histories"}
        for r in rob_results
    ]

    metrics = {
        "nu_true":          float(NU_TRUE),
        "nu_init":          NU_INIT,
        "main_ensemble": {
            "n_members":    N_MEMBERS,
            "n_sensors":    N_SENSORS,
            "noise_frac":   NOISE_FRAC,
            "nu_mean":      ens_result["nu_mean"],
            "nu_std":       ens_result["nu_std"],
            "ci_90_lo":     ens_result["ci_90_lo"],
            "ci_90_hi":     ens_result["ci_90_hi"],
            "true_in_ci":   ens_result["true_in_ci"],
            "mean_err_pct": err_pct,
            "nu_estimates": ens_result["nu_estimates"].tolist(),
        },
        "robustness": rob_rows,
    }

    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[main] Metrics saved to '{METRICS_PATH}'")


# ================================================================== #
#  Entry point                                                         #
# ================================================================== #

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_total = time.time()

    # Run pipeline
    ens_result  = step_main_ensemble()
    step_main_plots(ens_result)
    rob_results = step_robustness()
    step_robustness_plots(rob_results)
    step_save_metrics(ens_result, rob_results)

    elapsed = (time.time() - t_total) / 60
    _header(f"ALL DONE — total wall time: {elapsed:.1f} min")

    # Final summary to terminal
    print(f"\n  True  nu = {NU_TRUE:.6f}")
    print(f"  Learned nu (ensemble mean, default settings):")
    print(f"    mean = {ens_result['nu_mean']:.6f}")
    print(f"    std  = {ens_result['nu_std']:.6f}")
    print(f"    90%CI = [{ens_result['ci_90_lo']:.6f}, {ens_result['ci_90_hi']:.6f}]")
    print(f"    true in CI: {ens_result['true_in_ci']}")
    err = abs(ens_result["nu_mean"] - NU_TRUE) / NU_TRUE * 100
    print(f"    mean error: {err:.2f}%\n")

    print("  Output files:")
    files = [
        "outputs/nu_convergence.png",
        "outputs/solution_comparison.png",
        "outputs/robustness_summary.png",
        "outputs/robustness_table.png",
        "outputs/nu_distributions.png",
        "outputs/metrics.json",
    ]
    for f in files:
        full = os.path.join(os.path.dirname(__file__), f)
        exists = os.path.isfile(full)
        tag = "OK" if exists else "MISSING"
        size = f"  ({os.path.getsize(full)/1024:.1f} KB)" if exists else ""
        print(f"  [{tag}]  {f}{size}")
    print()


if __name__ == "__main__":
    main()
