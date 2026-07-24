"""
resume.py
---------
Resume the inverse_problem pipeline from Step 3 onward.

The main ensemble (Step 1) and plots A/B (Step 2) are already complete.
This script:
  3. Runs the full robustness sweep (all 9 noise×sensor conditions).
  4. Generates robustness plots (C, D, E).
  5. Saves the metrics JSON (loading main-ensemble metrics from checkpoint files).

Run
---
    cd inverse_problem
    python resume.py
"""

import os
import sys
import json
import time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data       import NU_TRUE
from model      import InverseBurgersPINN
from robustness import run_robustness_sweep
from plot       import (
    plot_robustness_summary,
    plot_robustness_table,
    plot_nu_distributions,
)

# ================================================================== #
#  Configuration (must match main.py)                                  #
# ================================================================== #

OUT_DIR       = "outputs"
ENS_DIR       = os.path.join(OUT_DIR, "ensemble")
ROB_DIR       = os.path.join(OUT_DIR, "robustness")
METRICS_PATH  = os.path.join(OUT_DIR, "metrics.json")

N_HIDDEN      = 4
N_NEURONS     = 50
NU_INIT       = 0.10
N_SENSORS     = 50
NOISE_FRAC    = 0.01
N_EPOCHS      = 8_000
N_MEMBERS     = 10
LR            = 1e-3
LAMBDA_DATA   = 100.0

NOISE_LEVELS  = [0.005, 0.01, 0.02]
SENSOR_COUNTS = [20, 50, 100]

DEVICE_STR = "auto"


def _device():
    if DEVICE_STR == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE_STR)


def _header(msg):
    bar = "=" * 68
    print(f"\n{bar}\n  {msg}\n{bar}")


# ================================================================== #
#  Load main-ensemble summary from saved checkpoints                   #
# ================================================================== #

def load_main_ensemble_summary() -> dict:
    """
    Reconstruct nu_mean/std/ci from the 10 saved checkpoint files.
    The model objects are also loaded so plots B can be re-used if needed.
    """
    device = _device()
    nu_vals = []
    members = []
    for i in range(N_MEMBERS):
        ckpt_path = os.path.join(ENS_DIR, f"member_{i}.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        nu_vals.append(ckpt["nu_final"])
        model = InverseBurgersPINN(
            n_hidden  = ckpt.get("n_hidden",  N_HIDDEN),
            n_neurons = ckpt.get("n_neurons", N_NEURONS),
            nu_init   = 1e-3,        # dummy — overwritten by state_dict
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        members.append(model)

    arr    = np.array(nu_vals)
    mu     = float(arr.mean())
    sigma  = float(arr.std(ddof=1))
    ci_lo  = float(np.percentile(arr, 5))
    ci_hi  = float(np.percentile(arr, 95))
    in_ci  = bool(ci_lo <= NU_TRUE <= ci_hi)

    print(f"[resume] Loaded main ensemble from {ENS_DIR}/")
    print(f"         nu estimates: {arr}")
    print(f"         mean={mu:.6f}  std={sigma:.6f}  90%CI=[{ci_lo:.6f},{ci_hi:.6f}]  in_CI={in_ci}")
    return dict(
        nu_estimates = arr,
        nu_mean      = mu,
        nu_std       = sigma,
        ci_90_lo     = ci_lo,
        ci_90_hi     = ci_hi,
        true_in_ci   = in_ci,
        members      = members,
        nu_histories = None,   # not stored in checkpoints
    )


# ================================================================== #
#  Step 3 — Robustness sweep                                           #
# ================================================================== #

def step_robustness() -> list[dict]:
    _header("STEP 3 — Robustness sweep (noise × sparsity)")
    return run_robustness_sweep(
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


# ================================================================== #
#  Entry point                                                         #
# ================================================================== #

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_total = time.time()

    ens_result  = load_main_ensemble_summary()
    rob_results = step_robustness()

    _header("STEP 4 — Generating robustness plots")
    plot_robustness_summary(rob_results, save_path=os.path.join(OUT_DIR, "robustness_summary.png"))
    plot_robustness_table  (rob_results, save_path=os.path.join(OUT_DIR, "robustness_table.png"))
    plot_nu_distributions  (rob_results, save_path=os.path.join(OUT_DIR, "nu_distributions.png"))

    _header("STEP 5 — Saving metrics JSON")
    err_pct  = abs(ens_result["nu_mean"] - NU_TRUE) / NU_TRUE * 100
    rob_rows = [{k: v for k, v in r.items() if k not in ("nu_histories", "members")}
                for r in rob_results]
    metrics = {
        "nu_true":       float(NU_TRUE),
        "nu_init":       NU_INIT,
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
    print(f"[resume] Metrics saved to '{METRICS_PATH}'")

    elapsed = (time.time() - t_total) / 60
    _header(f"ALL DONE — total wall time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
