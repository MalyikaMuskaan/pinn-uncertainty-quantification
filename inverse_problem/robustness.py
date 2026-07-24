"""
robustness.py
-------------
Robustness sweep for the inverse Burgers PINN.

Varies two axes independently:
  • noise_levels  — σ as fraction of signal range  (e.g. 0.005, 0.01, 0.02)
  • sensor_counts — number of sensor points         (e.g. 20, 50, 100)

For each (noise, n_sensors) combination, an ensemble of M members is trained
and the following statistics on the recovered ν are recorded:
    nu_mean, nu_std, ci_90_lo, ci_90_hi, true_in_ci, mean_abs_err_pct

The results are saved as a JSON file and returned as a list of dicts so that
plot.py can build the summary table / figure.
"""

import os
import json
import time
import numpy as np
import torch

from data  import NU_TRUE
from train import train as train_single


# ------------------------------------------------------------------ #
#  Single condition: train a mini-ensemble of M members                #
# ------------------------------------------------------------------ #

def _run_condition(
    noise_frac:  float,
    n_sensors:   int,
    n_members:   int   = 10,
    n_hidden:    int   = 4,
    n_neurons:   int   = 50,
    nu_init:     float = 0.03,
    n_col:       int   = 10_000,
    n_ic:        int   = 200,
    n_bc:        int   = 200,
    n_epochs:    int   = 3_500,
    adam_epochs: int   = 2_000,
    lr:          float = 1e-3,
    print_every: int   = 500,
    lambda_pde:  float = 1.0,
    lambda_ic:   float = 10.0,
    lambda_bc:   float = 10.0,
    lambda_data: float = 100.0,
    out_dir:     str   = "outputs/robustness",
    device_str:  str   = "auto",
    seed_offset: int   = 0,
) -> dict:
    """
    Train n_members PINNs for a single (noise_frac, n_sensors) condition.

    Returns a summary dict with ν statistics.
    """
    if device_str == "auto":
        _device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device_str = device_str

    nu_vals: list[float] = []
    nu_hist_all: list = []

    cond_dir = os.path.join(
        out_dir,
        f"noise{noise_frac:.4f}_sensors{n_sensors}"
    )
    os.makedirs(cond_dir, exist_ok=True)

    for i in range(n_members):
        save_path = os.path.join(cond_dir, f"member_{i}.pt")

        # Resume: load from checkpoint if already complete
        if os.path.isfile(save_path):
            ckpt = torch.load(save_path, map_location=_device_str)
            nu_vals.append(float(ckpt["nu_final"]))
            nu_hist_all.append(ckpt.get("nu_history", []))
            print(f"    [skip] member_{i} already exists  (nu={nu_vals[-1]:.6f})")
            continue

        torch.manual_seed(seed_offset + i)
        np.random.seed(seed_offset + i)

        res = train_single(
            n_sensors    = n_sensors,
            noise_frac   = noise_frac,
            sensor_seed  = seed_offset + i,
            n_hidden     = n_hidden,
            n_neurons    = n_neurons,
            nu_init      = nu_init,
            n_col        = n_col,
            n_ic         = n_ic,
            n_bc         = n_bc,
            n_epochs     = n_epochs,
            adam_epochs  = adam_epochs,
            lr           = lr,
            print_every  = print_every,
            lambda_pde   = lambda_pde,
            lambda_ic    = lambda_ic,
            lambda_bc    = lambda_bc,
            lambda_data  = lambda_data,
            save_path    = save_path,
            device_str   = _device_str,
            verbose      = False,
        )
        nu_vals.append(res["nu_final"])
        nu_hist_all.append(res["nu_history"])

    arr     = np.array(nu_vals)
    mu      = float(arr.mean())
    sigma   = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci_lo   = float(np.percentile(arr, 5))
    ci_hi   = float(np.percentile(arr, 95))
    in_ci   = bool(ci_lo <= NU_TRUE <= ci_hi)
    err_pct = abs(mu - NU_TRUE) / NU_TRUE * 100

    return dict(
        noise_frac   = noise_frac,
        n_sensors    = n_sensors,
        nu_mean      = mu,
        nu_std       = sigma,
        ci_90_lo     = ci_lo,
        ci_90_hi     = ci_hi,
        true_in_ci   = in_ci,
        mean_err_pct = err_pct,
        nu_estimates = arr.tolist(),
        nu_histories = nu_hist_all,  # kept for per-condition convergence plots
    )


# ------------------------------------------------------------------ #
#  Full robustness sweep                                               #
# ------------------------------------------------------------------ #

def run_robustness_sweep(
    noise_levels:  list[float] = [0.005, 0.01, 0.02],
    sensor_counts: list[int]   = [20, 50, 100],
    n_members:     int         = 10,
    n_hidden:      int         = 4,
    n_neurons:     int         = 50,
    nu_init:       float       = 0.03,
    n_col:         int         = 10_000,
    n_ic:          int         = 200,
    n_bc:          int         = 200,
    n_epochs:      int         = 3_500,
    adam_epochs:   int         = 2_000,
    lr:            float       = 1e-3,
    print_every:   int         = 500,
    lambda_pde:    float       = 1.0,
    lambda_ic:     float       = 10.0,
    lambda_bc:     float       = 10.0,
    lambda_data:   float       = 100.0,
    out_dir:       str         = "outputs/robustness",
    device_str:    str         = "auto",
) -> list[dict]:
    """
    Train M-member ensembles for every (noise, n_sensors) combination.

    Returns
    -------
    results : list of summary dicts, one per condition.
              Each dict is also written as a row in outputs/robustness/summary.json
    """
    os.makedirs(out_dir, exist_ok=True)
    total = len(noise_levels) * len(sensor_counts)
    print(f"\n[robustness] Starting sweep: {total} conditions × {n_members} members "
          f"= {total * n_members} training runs")
    print(f"[robustness] true nu = {NU_TRUE:.6f}")
    print(f"[robustness] schedule: Adam {adam_epochs} ep → L-BFGS {n_epochs - adam_epochs} ep"
          f"  (total {n_epochs})  nu_init={nu_init}\n")

    results: list[dict] = []
    run_idx = 0

    for noise in noise_levels:
        for n_s in sensor_counts:
            run_idx += 1
            seed_offset = run_idx * 100   # non-overlapping seeds across conditions

            t0 = time.time()
            print(f"  [{run_idx}/{total}] noise={noise:.1%}  sensors={n_s} ...",
                  end="  ", flush=True)

            summary = _run_condition(
                noise_frac   = noise,
                n_sensors    = n_s,
                n_members    = n_members,
                n_hidden     = n_hidden,
                n_neurons    = n_neurons,
                nu_init      = nu_init,
                n_col        = n_col,
                n_ic         = n_ic,
                n_bc         = n_bc,
                n_epochs     = n_epochs,
                adam_epochs  = adam_epochs,
                lr           = lr,
                print_every  = print_every,
                lambda_pde   = lambda_pde,
                lambda_ic    = lambda_ic,
                lambda_bc    = lambda_bc,
                lambda_data  = lambda_data,
                out_dir      = out_dir,
                device_str   = device_str,
                seed_offset  = seed_offset,
            )

            elapsed = time.time() - t0
            print(f"nu_mean={summary['nu_mean']:.6f}  std={summary['nu_std']:.6f}  "
                  f"err={summary['mean_err_pct']:.2f}%  in_CI={summary['true_in_ci']}  "
                  f"({elapsed:.0f}s)")

            results.append(summary)

            # Incrementally save (drop large nu_histories from JSON)
            json_rows = [{k: v for k, v in r.items() if k != "nu_histories"}
                         for r in results]
            with open(os.path.join(out_dir, "summary.json"), "w") as fh:
                json.dump(json_rows, fh, indent=2)

    print(f"\n[robustness] Sweep complete. Results written to {out_dir}/summary.json")
    return results
