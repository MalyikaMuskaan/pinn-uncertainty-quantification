"""
ablation.py
-----------
Phase 6B — Ablation Study

Two independent ablations on the Burgers' PINN setup:

Ablation A — Ensemble Size
--------------------------
Trains Deep Ensembles of M = 3, 5, 10, 20 members using the same per-member
schedule as run_ensemble.py.  For each M, computes:
  - Expected Calibration Error (ECE)
  - Empirical 90% coverage probability
  - Mean squared error of ensemble mean vs FD reference

The M = 10 case reuses existing checkpoints from outputs/ensemble/ if present,
avoiding redundant training.  M = 3 and M = 5 are sub-ensembles of those same
10 members (first 3 and first 5 checkpoints), so no extra training is needed
for them either.  M = 20 requires training 10 new members (seeds 10–19).

Ablation B — Loss Weighting
----------------------------
Trains the vanilla PINN (single model) three times with different loss weight
configurations:

  (a) Baseline:      λ_pde=1, λ_ic=10, λ_bc=10  (current project default)
  (b) Uniform:       λ_pde=1, λ_ic=1,  λ_bc=1   (unweighted)
  (c) Auto-balanced: λ_pde=1, λ_ic and λ_bc set so
                     λ_ic·L_ic0 ≈ λ_pde·L_pde0  and
                     λ_bc·L_bc0 ≈ λ_pde·L_pde0
                     (same strategy that fixed the inverse problem)

Each variant is evaluated by final MSE vs the FD reference.

Outputs (all under outputs/ablation/)
--------------------------------------
  ensemble_size/
      model_M{M}_member_{j}.pt     member checkpoints (M=20 only; others reuse)
      ensemble_size_metrics.json   ECE, coverage, MSE per M
      ablation_ensemble_size.png   ECE + coverage vs M (dual y-axis)
  loss_weighting/
      model_scheme_{a,b,c}.pt      checkpoint per weighting scheme
      loss_weighting_metrics.json  MSE + final losses per scheme
      ablation_loss_weighting.png  bar chart of final MSE per scheme

Usage
-----
    cd burgers_pinn
    python ablation.py

    # To run only one ablation:
    python ablation.py --only ensemble
    python ablation.py --only weighting
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import BurgersPINN
from data  import (
    sample_collocation_points,
    sample_initial_condition_points,
    sample_boundary_condition_points,
    make_evaluation_grid,
)
from ensemble import ensemble_predict, calibration_metrics, load_ensemble
from plot     import _fd_reference

# Standard NU — used for FD reference in all ablations
NU_BASELINE = 0.01 / 3.141592653589793


# ================================================================== #
#  Configuration                                                       #
# ================================================================== #

# Architecture and training (identical to run_ensemble.py members)
MEMBER_CFG = dict(
    n_col      = 10_000,
    n_ic       = 200,
    n_bc       = 200,
    n_hidden   = 4,
    n_neurons  = 50,
    n_epochs   = 5_000,
    lr         = 1e-3,
    lambda_pde = 1.0,
    lambda_ic  = 10.0,
    lambda_bc  = 10.0,
    print_every= 1_000,
)

ENSEMBLE_SIZES = [3, 5, 10, 20]   # M values to test
EXISTING_CKPT_DIR = "outputs/ensemble"   # M≤10 members live here
NEW_CKPT_DIR      = "outputs/ablation/ensemble_size"  # M=20 extras go here

N_X, N_T = 256, 100   # evaluation grid


# ================================================================== #
#  Shared training helper (nu fixed at baseline)                       #
# ================================================================== #

def _train_member(
    seed:      int,
    save_path: str,
    cfg:       dict,
    device:    torch.device,
    lambda_ic: float | None = None,
    lambda_bc: float | None = None,
) -> BurgersPINN:
    """
    Train a single BurgersPINN member for the baseline ν, return the model.

    lambda_ic / lambda_bc override the values in cfg when provided (used by
    the loss-weighting ablation without touching the shared MEMBER_CFG dict).
    """
    if os.path.isfile(save_path):
        print(f"  [ablation] checkpoint exists: '{save_path}', loading...")
        model = BurgersPINN(cfg["n_hidden"], cfg["n_neurons"]).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        return model

    torch.manual_seed(seed)
    np.random.seed(seed)

    lp = cfg["lambda_pde"]
    li = cfg["lambda_ic"] if lambda_ic is None else lambda_ic
    lb = cfg["lambda_bc"] if lambda_bc is None else lambda_bc

    model = BurgersPINN(cfg["n_hidden"], cfg["n_neurons"]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2000
    )

    x_ic, t_ic, u_ic = sample_initial_condition_points(cfg["n_ic"], device)
    x_bc, t_bc, u_bc = sample_boundary_condition_points(cfg["n_bc"], device)

    for epoch in range(1, cfg["n_epochs"] + 1):
        model.train()
        x_col, t_col = sample_collocation_points(cfg["n_col"], device)

        u     = model(x_col, t_col)
        ones  = torch.ones_like(u)
        u_x   = torch.autograd.grad(u, x_col, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)[0]
        u_t   = torch.autograd.grad(u, t_col, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)[0]
        u_xx  = torch.autograd.grad(u_x, x_col,
                                    grad_outputs=torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        f      = u_t + u * u_x - NU_BASELINE * u_xx
        l_pde  = torch.mean(f ** 2)
        l_ic   = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
        l_bc   = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
        total  = lp * l_pde + li * l_ic + lb * l_bc

        optimiser.zero_grad()
        total.backward()
        optimiser.step()
        scheduler.step(total.detach())

        if epoch % cfg["print_every"] == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"    ep {epoch:>6d}  pde={l_pde.item():.3e}  "
                  f"ic={l_ic.item():.3e}  lr={lr_now:.1e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return model


# ================================================================== #
#  Ablation A — Ensemble Size                                          #
# ================================================================== #

def _ckpt_path_for_member(j: int, cfg: dict) -> str:
    """
    Return the checkpoint path for ensemble member j.
    Members 0–9 live in the existing outputs/ensemble/ directory.
    Members 10–19 (needed for M=20) go into outputs/ablation/ensemble_size/.
    """
    if j < 10:
        return os.path.join(EXISTING_CKPT_DIR, f"model_{j}.pt")
    return os.path.join(NEW_CKPT_DIR, f"member_{j}.pt")


def run_ensemble_size_ablation(
    device:  torch.device,
    out_dir: str,
) -> list[dict]:
    """
    For each M in ENSEMBLE_SIZES:
      1. Ensure M member checkpoints exist (train missing ones).
      2. Load the first M members.
      3. Run ensemble_predict → calibration_metrics.
      4. Record ECE, 90% coverage, MSE.

    Returns a list of result dicts, one per M.
    """
    print()
    print("=" * 65)
    print("  ABLATION A — Ensemble Size")
    print("=" * 65)

    # The maximum M we need is 20; ensure all 20 checkpoints exist.
    max_M = max(ENSEMBLE_SIZES)
    print(f"[ablation-A] Ensuring {max_M} member checkpoints exist...")
    for j in range(max_M):
        ckpt = _ckpt_path_for_member(j, MEMBER_CFG)
        if not os.path.isfile(ckpt):
            print(f"  Training missing member {j} (seed={j}) ...")
            t0 = time.time()
            _train_member(
                seed=j, save_path=ckpt,
                cfg=MEMBER_CFG, device=device,
            )
            print(f"  Member {j} done in {(time.time()-t0)/60:.1f} min")

    # Evaluation grid
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(N_X, N_T, device)

    all_results = []

    for M in ENSEMBLE_SIZES:
        print(f"\n[ablation-A] M = {M} members")

        # Load first M members
        ckpt_paths = [_ckpt_path_for_member(j, MEMBER_CFG) for j in range(M)]
        models = []
        for p in ckpt_paths:
            m = BurgersPINN(MEMBER_CFG["n_hidden"], MEMBER_CFG["n_neurons"])
            m.load_state_dict(torch.load(p, map_location=device))
            m.to(device).eval()
            models.append(m)

        u_mean, u_std, _ = ensemble_predict(
            models, x_flat, t_flat, grid_shape=(N_T, N_X)
        )

        # MSE vs FD reference
        x_vals = x_grid[0, :]
        t_vals = t_grid[:, 0]
        u_ref  = _fd_reference(x_vals, t_vals, NU_BASELINE)
        mse    = float(np.mean((u_mean - u_ref) ** 2))

        # Calibration (reuses existing calibration_metrics but passes
        # pre-computed u_ref to avoid re-running FD twice)
        metrics = calibration_metrics(u_mean, u_std, x_grid, t_grid)

        result = {
            "M":          M,
            "ece":        metrics["ece"],
            "coverage_90": metrics["coverage_90"],
            "mse":        mse,
        }
        all_results.append(result)
        print(f"  M={M:>2d}  ECE={metrics['ece']:.4f}  "
              f"cov90={metrics['coverage_90']:.4f}  MSE={mse:.4e}")

    # -- Save metrics --
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "ensemble_size_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[ablation-A] Metrics saved to '{metrics_path}'")

    # -- Plot --
    _plot_ensemble_size(all_results, out_dir)
    return all_results


def _plot_ensemble_size(results: list[dict], out_dir: str) -> None:
    """
    Dual y-axis plot: ECE (left, lower is better) and 90% coverage (right,
    closer to 0.9 is better) vs ensemble size M.
    """
    Ms        = [r["M"]          for r in results]
    eces      = [r["ece"]        for r in results]
    coverages = [r["coverage_90"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    color_ece = "#3b82d4"
    color_cov = "#e05c2a"

    ax1.plot(Ms, eces, color=color_ece, marker="o", linewidth=2.0,
             markersize=8, label="ECE ↓")
    ax1.set_xlabel("Ensemble size  M", fontsize=12)
    ax1.set_ylabel("ECE  (lower = better)", fontsize=11, color=color_ece)
    ax1.tick_params(axis="y", labelcolor=color_ece)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(Ms, coverages, color=color_cov, marker="s", linewidth=2.0,
             markersize=8, linestyle="--", label="90% coverage ↑")
    ax2.axhline(0.9, color=color_cov, linewidth=1.0, linestyle=":",
                alpha=0.6, label="Ideal 90% coverage")
    ax2.set_ylabel("Empirical 90% coverage", fontsize=11, color=color_cov)
    ax2.tick_params(axis="y", labelcolor=color_cov)
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    ax1.set_title(
        "Ablation A — Ensemble Size\n"
        "ECE and 90% Coverage vs Number of Members",
        fontsize=12,
    )
    ax1.set_xticks(Ms)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_ensemble_size.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation-A] Plot saved to '{path}'")


# ================================================================== #
#  Ablation B — Loss Weighting                                         #
# ================================================================== #

WEIGHTING_SCHEMES = {
    "a_baseline": {
        "label":      "(a) Baseline  λ_ic=10, λ_bc=10",
        "lambda_pde": 1.0,
        "lambda_ic":  10.0,
        "lambda_bc":  10.0,
        "auto":       False,
    },
    "b_uniform": {
        "label":      "(b) Uniform  λ_ic=1, λ_bc=1",
        "lambda_pde": 1.0,
        "lambda_ic":  1.0,
        "lambda_bc":  1.0,
        "auto":       False,
    },
    "c_auto": {
        "label":      "(c) Auto-balanced  (λ·L ≈ equal at init)",
        "lambda_pde": 1.0,
        "lambda_ic":  None,   # computed at runtime
        "lambda_bc":  None,   # computed at runtime
        "auto":       True,
    },
}


def _auto_balance_weights(
    model:  BurgersPINN,
    device: torch.device,
    cfg:    dict,
    lambda_pde: float,
) -> tuple[float, float]:
    """
    Compute lambda_ic and lambda_bc so that:
        lambda_pde * L_pde0 ≈ lambda_ic * L_ic0 ≈ lambda_bc * L_bc0
    at the model's initial parameter values (before any gradient steps).

    This is the same strategy used in inverse_problem/train.py.
    """
    model.eval()

    # Sample a temporary collocation set for the initial evaluation
    with torch.no_grad():
        xc0_np = np.random.uniform(-1.0, 1.0, (cfg["n_col"], 1)).astype(np.float32)
        tc0_np = np.random.uniform( 0.0, 1.0, (cfg["n_col"], 1)).astype(np.float32)
        xc0 = torch.tensor(xc0_np, device=device)
        tc0 = torch.tensor(tc0_np, device=device)
        x_ic, t_ic, u_ic = sample_initial_condition_points(cfg["n_ic"], device)
        x_bc, t_bc, u_bc = sample_boundary_condition_points(cfg["n_bc"], device)

    with torch.enable_grad():
        xc0_g = xc0.detach().requires_grad_(True)
        tc0_g = tc0.detach().requires_grad_(True)

        u     = model(xc0_g, tc0_g)
        ones  = torch.ones_like(u)
        u_x   = torch.autograd.grad(u, xc0_g, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)[0]
        u_t   = torch.autograd.grad(u, tc0_g, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)[0]
        u_xx  = torch.autograd.grad(u_x, xc0_g,
                                    grad_outputs=torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        f      = u_t + u * u_x - NU_BASELINE * u_xx
        l_pde0 = torch.mean(f ** 2).item()

    with torch.no_grad():
        l_ic0 = torch.mean((model(x_ic, t_ic) - u_ic) ** 2).item()
        l_bc0 = torch.mean((model(x_bc, t_bc) - u_bc) ** 2).item()

    model.train()

    lam_ic = (lambda_pde * l_pde0 / l_ic0) if l_ic0 > 0 else 10.0
    lam_bc = (lambda_pde * l_pde0 / l_bc0) if l_bc0 > 0 else 10.0

    print(f"  [auto-balance] l_pde0={l_pde0:.3e}  l_ic0={l_ic0:.3e}  l_bc0={l_bc0:.3e}")
    print(f"  [auto-balance] lambda_ic={lam_ic:.3f}  lambda_bc={lam_bc:.3f}")
    return lam_ic, lam_bc


def run_loss_weighting_ablation(
    device:  torch.device,
    out_dir: str,
) -> list[dict]:
    """
    Train one PINN per weighting scheme, evaluate MSE vs FD reference.

    Returns a list of result dicts, one per scheme.
    """
    print()
    print("=" * 65)
    print("  ABLATION B — Loss Weighting")
    print("=" * 65)

    os.makedirs(out_dir, exist_ok=True)
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(N_X, N_T, device)
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]
    u_ref  = _fd_reference(x_vals, t_vals, NU_BASELINE)

    all_results = []

    for scheme_key, scheme in WEIGHTING_SCHEMES.items():
        print(f"\n[ablation-B] Scheme: {scheme['label']}")
        ckpt_path = os.path.join(out_dir, f"model_scheme_{scheme_key[-1]}.pt")

        lam_ic = scheme["lambda_ic"]
        lam_bc = scheme["lambda_bc"]

        if os.path.isfile(ckpt_path):
            print(f"  Checkpoint exists: '{ckpt_path}', loading...")
            model = BurgersPINN(
                MEMBER_CFG["n_hidden"], MEMBER_CFG["n_neurons"]
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            # For auto-balanced scheme, compute lambdas from a freshly init'd model
            if scheme["auto"]:
                probe = BurgersPINN(
                    MEMBER_CFG["n_hidden"], MEMBER_CFG["n_neurons"]
                ).to(device)
                lam_ic, lam_bc = _auto_balance_weights(
                    probe, device, MEMBER_CFG, scheme["lambda_pde"]
                )
                del probe

            t0 = time.time()
            model = _train_member(
                seed=42,
                save_path=ckpt_path,
                cfg=MEMBER_CFG,
                device=device,
                lambda_ic=lam_ic,
                lambda_bc=lam_bc,
            )
            print(f"  Training done in {(time.time()-t0)/60:.1f} min")

        # -- Evaluate --
        model.eval()
        with torch.no_grad():
            u_pred = model(x_flat, t_flat).cpu().numpy().reshape(N_T, N_X)

        mse    = float(np.mean((u_pred - u_ref) ** 2))
        rel_l2 = float(np.linalg.norm(u_pred - u_ref) /
                       (np.linalg.norm(u_ref) + 1e-12))

        result = {
            "scheme":     scheme_key,
            "label":      scheme["label"],
            "lambda_ic":  lam_ic,
            "lambda_bc":  lam_bc,
            "mse":        mse,
            "rel_l2":     rel_l2,
        }
        all_results.append(result)
        print(f"  MSE={mse:.4e}   Rel-L2={rel_l2:.4e}  ({rel_l2:.2%})")

    # -- Save metrics --
    metrics_path = os.path.join(out_dir, "loss_weighting_metrics.json")
    # Convert None values to null-safe strings for JSON serialization
    safe = []
    for r in all_results:
        d = dict(r)
        if d["lambda_ic"] is None:
            d["lambda_ic"] = "auto"
        if d["lambda_bc"] is None:
            d["lambda_bc"] = "auto"
        safe.append(d)
    with open(metrics_path, "w") as f:
        json.dump(safe, f, indent=2)
    print(f"\n[ablation-B] Metrics saved to '{metrics_path}'")

    # -- Plot --
    _plot_loss_weighting(all_results, out_dir)
    return all_results


def _plot_loss_weighting(results: list[dict], out_dir: str) -> None:
    """
    Horizontal bar chart of final MSE per weighting scheme.
    Bars are sorted so the best (lowest MSE) is at the top.
    """
    labels = [r["label"]  for r in results]
    mses   = [r["mse"]    for r in results]
    rel_l2 = [r["rel_l2"] for r in results]

    # Sort ascending MSE (best at top in horizontal bar chart)
    order  = np.argsort(mses)
    labels = [labels[i] for i in order]
    mses   = [mses[i]   for i in order]
    rel_l2 = [rel_l2[i] for i in order]

    colors = ["#34d399" if mse == min(mses) else "#93c5fd" for mse in mses]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, mses, color=colors, edgecolor="#57606a", linewidth=0.7)

    # Annotate bars with Rel-L2
    for bar, rl in zip(bars, rel_l2):
        ax.text(
            bar.get_width() * 1.03, bar.get_y() + bar.get_height() / 2,
            f"Rel-L2: {rl:.2%}",
            va="center", fontsize=9, color="#1f2328",
        )

    ax.set_xlabel("Final MSE vs FD reference  (lower = better)", fontsize=11)
    ax.set_title(
        "Ablation B — Loss Weighting\nFinal MSE per weighting scheme",
        fontsize=12,
    )
    ax.set_xscale("log")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "ablation_loss_weighting.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation-B] Plot saved to '{path}'")


# ================================================================== #
#  Entry point                                                         #
# ================================================================== #

def main(only: str | None = None) -> dict:
    """
    Run ablations A and B (or just one if `only` is "ensemble" or "weighting").

    Returns a dict with keys 'ensemble_size' and/or 'loss_weighting'.
    """
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] Device: {device_obj}")

    out_root = "outputs/ablation"
    output   = {}

    run_a = only is None or only == "ensemble"
    run_b = only is None or only == "weighting"

    if run_a:
        results_a = run_ensemble_size_ablation(
            device  = device_obj,
            out_dir = os.path.join(out_root, "ensemble_size"),
        )
        output["ensemble_size"] = results_a

        print()
        print("=" * 65)
        print("  ABLATION A — SUMMARY")
        print(f"  {'M':>4}   {'ECE':>8}   {'Cov90':>8}   {'MSE':>12}")
        print("  " + "-" * 45)
        for r in results_a:
            print(f"  {r['M']:>4d}   {r['ece']:>8.4f}   "
                  f"{r['coverage_90']:>8.4f}   {r['mse']:>12.4e}")
        print("=" * 65)

    if run_b:
        results_b = run_loss_weighting_ablation(
            device  = device_obj,
            out_dir = os.path.join(out_root, "loss_weighting"),
        )
        output["loss_weighting"] = results_b

        print()
        print("=" * 65)
        print("  ABLATION B — SUMMARY")
        print(f"  {'Scheme':<42}   {'MSE':>12}   {'Rel-L2':>8}")
        print("  " + "-" * 68)
        for r in results_b:
            print(f"  {r['label']:<42}   {r['mse']:>12.4e}   {r['rel_l2']:>8.2%}")
        print("=" * 65)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 6B ablation study for Burgers' PINN"
    )
    parser.add_argument(
        "--only",
        choices=["ensemble", "weighting"],
        default=None,
        help="Run only one ablation (omit to run both)",
    )
    args = parser.parse_args()
    main(only=args.only)
