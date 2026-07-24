"""
evaluate.py
-----------
Compare the Fourier Neural Operator against the per-instance PINN on the
Burgers' equation test set.

What is measured
----------------
(a) Accuracy   — relative L2 error on the 100 held-out test ICs
(b) Inference  — wall-clock time to predict one test instance (post-training)
(c) Training   — FNO one-time training cost vs PINN per-instance retraining cost

PINN comparison note
--------------------
The PINN (from inverse_problem/) was designed for a SINGLE fixed IC
(-sin(πx)) and must be retrained from scratch for every new IC.  For a fair
comparison we retrain a forward PINN (without the learnable ν — ν is fixed at
the true value) on each test IC.  This is expensive: one retrain per instance.
We time a single PINN retrain to estimate the per-instance cost.

To keep evaluation time manageable on GPU we time the PINN on just
N_PINN_EVAL instances (default 3); the reported per-instance cost is the mean.

Outputs
-------
  outputs/eval_fno_vs_pinn.json       — summary metrics
  outputs/plots/comparison_NNN.png    — FNO vs PINN vs ground truth plots
  outputs/plots/summary_table.png     — rendered metrics table

Usage
-----
    python evaluate.py                          # uses saved FNO checkpoint
    python evaluate.py --n_pinn_eval 5         # retrain 5 PINN instances
    python evaluate.py --skip_pinn             # FNO metrics only (no PINN)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from model    import FNO1d
from data_gen import load_dataset, fd_solve, NU, X_MIN, X_MAX, T_MIN, T_MAX

# ---------------------------------------------------------------------------
# Optional PINN import — gracefully skip if not available from this directory
# ---------------------------------------------------------------------------
_PINN_DIR = os.path.join(os.path.dirname(__file__), "..", "inverse_problem")

def _import_pinn():
    """Import PINN model + train from the sibling inverse_problem directory."""
    if _PINN_DIR not in sys.path:
        sys.path.insert(0, _PINN_DIR)
    from model import InverseBurgersPINN          # noqa: F401
    import train as pinn_train_mod                # noqa: F401
    return InverseBurgersPINN, pinn_train_mod


# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #

def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """Relative L2 error: ||pred - true||₂ / ||true||₂."""
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-10))


# ------------------------------------------------------------------ #
#  FNO evaluation on test set                                          #
# ------------------------------------------------------------------ #

def evaluate_fno(
    model:     FNO1d,
    ds:        dict,
    device:    torch.device,
) -> dict:
    """
    Evaluate FNO on the test split.

    Returns
    -------
    dict with keys:
        rel_l2_mean, rel_l2_std, rel_l2_all  — per-instance relative L2 errors
        inference_time_mean                   — seconds per instance (GPU/CPU)
        inference_time_all                    — list of per-instance times
    """
    model.eval()

    u0_test = torch.tensor(ds["u0_test"], dtype=torch.float32).to(device)
    u_test  = ds["u_test"]                 # (N_test, N_x, N_t) numpy

    rel_l2_all    : list[float] = []
    infer_times   : list[float] = []

    with torch.no_grad():
        for i in range(len(u0_test)):
            u0_i = u0_test[i : i + 1]     # (1, N_x)

            # Time inference on a single instance
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0   = time.perf_counter()
            pred = model(u0_i)             # (1, N_x, N_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_times.append(time.perf_counter() - t0)

            pred_np = pred[0].cpu().numpy()   # (N_x, N_t)
            err     = relative_l2(pred_np, u_test[i])
            rel_l2_all.append(err)

    return dict(
        rel_l2_mean         = float(np.mean(rel_l2_all)),
        rel_l2_std          = float(np.std(rel_l2_all)),
        rel_l2_all          = rel_l2_all,
        inference_time_mean = float(np.mean(infer_times)),
        inference_time_all  = infer_times,
    )


# ------------------------------------------------------------------ #
#  Per-instance PINN retraining + evaluation                           #
# ------------------------------------------------------------------ #

def _build_pinn_ic_data(u0: np.ndarray, x_grid: np.ndarray, device):
    """Return (x_ic, t_ic, u_ic) tensors for a given IC array."""
    x_t = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.zeros_like(x_t)
    u_t = torch.tensor(u0.reshape(-1, 1),     dtype=torch.float32, device=device)
    return x_t, t_t, u_t


def evaluate_pinn_instances(
    ds:           dict,
    device:       torch.device,
    n_eval:       int = 3,
    pinn_epochs:  int = 3_500,
    pinn_adam:    int = 2_000,
) -> dict:
    """
    Retrain a forward PINN (fixed ν = NU_TRUE) for `n_eval` test instances.

    Returns
    -------
    dict with keys:
        rel_l2_mean, rel_l2_std, rel_l2_all
        train_time_mean, train_time_all
        inference_time_mean, inference_time_all
    """
    try:
        InverseBurgersPINN, pinn_train_mod = _import_pinn()
    except ImportError as e:
        print(f"[evaluate] Cannot import PINN: {e}")
        return {}

    # We re-use the PINN architecture but fix ν at the true value.
    # Concretely: we train a standard forward PINN with IC/BC/PDE losses only,
    # no sensor data, ν fixed.  We do this by importing and slightly repurposing
    # the existing infrastructure.

    from data_gen import fd_solve, NU
    import importlib, types

    x_grid = ds["x_grid"]
    t_grid = ds["t_grid"]

    rel_l2_all   : list[float] = []
    train_times  : list[float] = []
    infer_times  : list[float] = []

    for idx in range(n_eval):
        u0   = ds["u0_test"][idx]       # (N_x,)
        u_gt = ds["u_test"][idx]        # (N_x, N_t)

        print(f"  [PINN] retraining instance {idx+1}/{n_eval} ...", end=" ", flush=True)
        t0 = time.time()

        # ---- Build a minimal forward PINN (ν fixed, no sensor data) ----
        # We call train_single with n_sensors=0 via monkey-patching make_sensor_data.
        # Simpler: inline the PINN training here with the correct IC.

        model_pinn = InverseBurgersPINN(
            n_hidden  = 4,
            n_neurons = 50,
            nu_init   = NU,      # start at the true value — fair forward PINN
        ).to(device)

        # Freeze ν so this is a pure forward problem
        model_pinn.raw_nu.requires_grad_(False)

        # IC data for this specific initial condition
        x_ic, t_ic, u_ic = _build_pinn_ic_data(u0, x_grid, device)

        # BC data (u = 0 at x = ±1)
        t_bc_np = np.linspace(T_MIN, T_MAX, 200, dtype=np.float32).reshape(-1, 1)
        x_left  = np.full_like(t_bc_np, X_MIN)
        x_right = np.full_like(t_bc_np, X_MAX)
        x_bc = torch.tensor(np.vstack([x_left,  x_right]), device=device)
        t_bc = torch.tensor(np.vstack([t_bc_np, t_bc_np]), device=device)
        u_bc = torch.zeros(400, 1, device=device)

        optimiser_pinn = torch.optim.Adam(
            [p for p in model_pinn.parameters() if p.requires_grad], lr=1e-3)
        scheduler_pinn = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser_pinn, factor=0.5, patience=1500)

        n_col = 10_000

        for ep in range(1, pinn_adam + 1):
            model_pinn.train()
            x_col = torch.tensor(
                np.random.uniform(X_MIN, X_MAX, (n_col, 1)).astype(np.float32),
                device=device, requires_grad=True)
            t_col = torch.tensor(
                np.random.uniform(T_MIN, T_MAX, (n_col, 1)).astype(np.float32),
                device=device, requires_grad=True)

            u_pred = model_pinn(x_col, t_col)
            u_x = torch.autograd.grad(u_pred, x_col,
                grad_outputs=torch.ones_like(u_pred),
                create_graph=True, retain_graph=True)[0]
            u_t = torch.autograd.grad(u_pred, t_col,
                grad_outputs=torch.ones_like(u_pred),
                create_graph=True, retain_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_col,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True, retain_graph=True)[0]

            pde_res = u_t + u_pred * u_x - model_pinn.nu * u_xx
            l_pde   = torch.mean(pde_res ** 2)
            l_ic    = torch.mean((model_pinn(x_ic, t_ic) - u_ic) ** 2)
            l_bc    = torch.mean((model_pinn(x_bc, t_bc) - u_bc) ** 2)
            loss    = l_pde + 10.0 * l_ic + 10.0 * l_bc

            optimiser_pinn.zero_grad()
            loss.backward()
            optimiser_pinn.step()
            scheduler_pinn.step(loss.detach())

        train_time_i = time.time() - t0
        print(f"done in {train_time_i:.0f}s")
        train_times.append(train_time_i)

        # ---- Inference: predict on the full (x, t) grid ----
        model_pinn.eval()
        x_flat = torch.tensor(
            np.tile(x_grid, len(t_grid)).reshape(-1, 1).astype(np.float32),
            device=device)
        t_flat = torch.tensor(
            np.repeat(t_grid, len(x_grid)).reshape(-1, 1).astype(np.float32),
            device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        ti0 = time.perf_counter()
        with torch.no_grad():
            u_pred_flat = model_pinn(x_flat, t_flat).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_times.append(time.perf_counter() - ti0)

        # u_pred_flat shape: (N_x * N_t, 1) — need (N_x, N_t)
        # x_flat is tiled: [ x0,x1,...,xNx, x0,...] over t_grid steps
        # So reshape as (N_t, N_x) then transpose
        u_pred_np = u_pred_flat.reshape(len(t_grid), len(x_grid)).T  # (N_x, N_t)
        rel_l2_all.append(relative_l2(u_pred_np, u_gt))

    return dict(
        rel_l2_mean          = float(np.mean(rel_l2_all)),
        rel_l2_std           = float(np.std(rel_l2_all)),
        rel_l2_all           = rel_l2_all,
        train_time_mean      = float(np.mean(train_times)),
        train_time_all       = train_times,
        inference_time_mean  = float(np.mean(infer_times)),
        inference_time_all   = infer_times,
    )


# ------------------------------------------------------------------ #
#  Plots                                                               #
# ------------------------------------------------------------------ #

def plot_comparisons(model, ds, device, out_dir, n_plot=3):
    """Plot FNO prediction vs ground truth for `n_plot` test instances."""
    model.eval()
    x_grid = ds["x_grid"]
    t_grid = ds["t_grid"]
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(min(n_plot, len(ds["u0_test"]))):
            u0_t  = torch.tensor(ds["u0_test"][i:i+1],
                                 dtype=torch.float32, device=device)
            pred  = model(u0_t)[0].cpu().numpy()   # (N_x, N_t)
            true  = ds["u_test"][i]                # (N_x, N_t)
            err   = relative_l2(pred, true)

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            t_slices = [0, len(t_grid) // 2, len(t_grid) - 1]
            labels   = [f"t={t_grid[k]:.2f}" for k in t_slices]

            for ax, k, lbl in zip(axes, t_slices, labels):
                ax.plot(x_grid, true[:, k], color="#333333",
                        lw=2.0, label="Ground truth (FD)")
                ax.plot(x_grid, pred[:, k], color="#3b82d4",
                        lw=1.8, ls="--", label="FNO prediction")
                ax.set_title(lbl, fontsize=10)
                ax.set_xlabel("x")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            axes[0].set_ylabel("u(x, t)")
            fig.suptitle(f"FNO vs Ground Truth — test instance {i}  "
                         f"(rel L2 = {err:.4f})", fontsize=11)
            plt.tight_layout()
            save_path = os.path.join(plots_dir, f"comparison_{i:03d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[plot] {save_path}")


def plot_summary_table(metrics: dict, out_dir: str) -> None:
    """Render a summary comparison table as a figure."""
    fno  = metrics.get("fno",  {})
    pinn = metrics.get("pinn", {})

    rows = [
        ["Metric",              "FNO (one-time train)", "PINN (per-instance)"],
        ["Rel L2 error (mean)", f"{fno.get('rel_l2_mean', float('nan')):.4f}",
                                f"{pinn.get('rel_l2_mean', 'N/A')}"],
        ["Rel L2 error (std)",  f"{fno.get('rel_l2_std',  float('nan')):.4f}",
                                f"{pinn.get('rel_l2_std', 'N/A')}"],
        ["Inference time / instance",
                                f"{fno.get('inference_time_mean', float('nan'))*1e3:.2f} ms",
                                f"{pinn.get('inference_time_mean', float('nan')):.2f} s"],
        ["Training cost",       f"{metrics.get('fno_total_train_time', float('nan')):.0f} s (all ICs)",
                                f"{pinn.get('train_time_mean', float('nan')):.0f} s (per IC)"],
        ["Generalisation",      "Any IC in distribution", "Only the trained IC"],
        ["Physics constraint",  "No (data-driven)",       "Yes (PDE residual)"],
    ]

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.45 * n_rows))
    ax.axis("off")
    t = ax.table(cellText=rows[1:], colLabels=rows[0],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.0, 1.6)

    for j in range(3):
        t[0, j].set_facecolor("#3b82d4")
        t[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle("FNO vs PINN — Burgers equation comparison",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "plots", "summary_table.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device={device}")

    # ---- Load dataset ----
    ds = load_dataset(args.data_path)
    n_x = ds["x_grid"].shape[0]
    n_t = ds["t_grid"].shape[0]

    # ---- Load FNO checkpoint ----
    ckpt_path = os.path.join(args.out_dir, "fno_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"FNO checkpoint not found at '{ckpt_path}'.  "
            f"Run  python train.py  first.")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = FNO1d(n_x=ckpt["n_x"], n_t=ckpt["n_t"],
                  modes=ckpt["modes"], width=ckpt["width"],
                  depth=ckpt["depth"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[evaluate] Loaded FNO from '{ckpt_path}'  "
          f"(best val loss={ckpt['val_loss']:.4e}  epoch={ckpt['epoch']})")

    # ---- Evaluate FNO ----
    print("[evaluate] Evaluating FNO on test set ...")
    fno_metrics = evaluate_fno(model, ds, device)
    print(f"[evaluate] FNO  rel_L2 mean={fno_metrics['rel_l2_mean']:.4f}  "
          f"std={fno_metrics['rel_l2_std']:.4f}  "
          f"infer={fno_metrics['inference_time_mean']*1e3:.2f}ms/instance")

    # ---- Load FNO training time from history (if available) ----
    hist_path = os.path.join(args.out_dir, "train_history.npz")
    fno_total_train_time = float("nan")
    if os.path.exists(hist_path):
        pass   # train_history.npz stores loss curves, not wall time
    # Wall time is printed during training; user should note it manually.

    metrics = dict(fno=fno_metrics,
                   fno_total_train_time=fno_total_train_time)

    # ---- Optionally evaluate PINN ----
    if not args.skip_pinn:
        print(f"[evaluate] Retraining PINN on {args.n_pinn_eval} test instances ...")
        pinn_metrics = evaluate_pinn_instances(
            ds, device,
            n_eval=args.n_pinn_eval,
            pinn_epochs=args.pinn_epochs,
            pinn_adam=args.pinn_adam,
        )
        if pinn_metrics:
            print(f"[evaluate] PINN rel_L2 mean={pinn_metrics['rel_l2_mean']:.4f}  "
                  f"train={pinn_metrics['train_time_mean']:.0f}s/instance  "
                  f"infer={pinn_metrics['inference_time_mean']:.3f}s/instance")
            metrics["pinn"] = pinn_metrics
    else:
        print("[evaluate] --skip_pinn set; skipping PINN retraining.")

    # ---- Save JSON summary ----
    summary_path = os.path.join(args.out_dir, "eval_fno_vs_pinn.json")

    def _serialise(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, list):
            return [_serialise(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        return obj

    with open(summary_path, "w") as fh:
        json.dump(_serialise(metrics), fh, indent=2)
    print(f"[evaluate] Summary saved to '{summary_path}'")

    # ---- Plots ----
    plot_comparisons(model, ds, device, args.out_dir, n_plot=args.n_plot)
    plot_summary_table(metrics, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    default="outputs/dataset.npz")
    parser.add_argument("--out_dir",      default="outputs")
    parser.add_argument("--n_plot",       type=int,  default=3)
    parser.add_argument("--n_pinn_eval",  type=int,  default=3)
    parser.add_argument("--pinn_epochs",  type=int,  default=3_500)
    parser.add_argument("--pinn_adam",    type=int,  default=2_000)
    parser.add_argument("--skip_pinn",    action="store_true",
                        help="Skip PINN retraining; report FNO metrics only")
    main(parser.parse_args())
