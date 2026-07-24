"""
diagnose.py
-----------
Diagnostic investigation for the best-performing inverse-problem case:
  noise = 0.005 (0.5%),  sensors = 100

Runs 3 ensemble members with FULL per-epoch history logging and produces:
  - Plot 1: nu trajectory over ALL epochs (per member)
  - Plot 2: All four loss components over training
  - Text:   nu true, nu init, final nu, parameterisation details
  - Experiment A: same case with 2x epochs (16 000)
  - Experiment B: same case with nu-parameter LR divided by 10
  - Summary plot comparing baseline vs Exp-A vs Exp-B

Run
---
    cd inverse_problem
    python diagnose.py
"""

import os, sys, math, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data import (
    NU_TRUE, sample_collocation_points,
    sample_initial_condition_points, sample_boundary_condition_points,
    make_sensor_data,
)
from model import InverseBurgersPINN

OUT_DIR = "outputs/diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================== #
#  Shared config -- best case                                          #
# ================================================================== #
N_SENSORS   = 100
NOISE_FRAC  = 0.005
N_MEMBERS   = 3
NU_INIT     = 0.10
N_EPOCHS    = 8_000
LR          = 1e-3
LAMBDA_PDE  = 1.0
LAMBDA_IC   = 10.0
LAMBDA_BC   = 10.0
LAMBDA_DATA = 100.0
N_COL       = 10_000
N_IC        = 200
N_BC        = 200


# ================================================================== #
#  Custom training loop that logs EVERY epoch                         #
# ================================================================== #

def train_full_history(
    seed:         int,
    n_epochs:     int   = N_EPOCHS,
    lr:           float = LR,
    nu_lr_factor: float = 1.0,
    label:        str   = "",
    verbose:      bool  = True,
) -> dict:
    """
    Train one InverseBurgersPINN with per-epoch logging of all losses and nu.
    nu_lr_factor: scaling applied to raw_nu LR relative to network weights LR.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    model = InverseBurgersPINN(n_hidden=4, n_neurons=50, nu_init=NU_INIT).to(device)

    net_params = list(model.network.parameters())
    nu_params  = [model.raw_nu]
    optimiser  = torch.optim.Adam([
        {"params": net_params, "lr": lr},
        {"params": nu_params,  "lr": lr * nu_lr_factor},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=1500
    )

    x_ic, t_ic, u_ic = sample_initial_condition_points(N_IC, device)
    x_bc, t_bc, u_bc = sample_boundary_condition_points(N_BC, device)
    x_s,  t_s,  u_s  = make_sensor_data(N_SENSORS, NOISE_FRAC, seed, device)

    hist_nu    = np.empty(n_epochs, dtype=np.float64)
    hist_pde   = np.empty(n_epochs, dtype=np.float64)
    hist_ic    = np.empty(n_epochs, dtype=np.float64)
    hist_bc    = np.empty(n_epochs, dtype=np.float64)
    hist_data  = np.empty(n_epochs, dtype=np.float64)
    hist_total = np.empty(n_epochs, dtype=np.float64)

    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        optimiser.zero_grad()

        x_col, t_col = sample_collocation_points(N_COL, device)
        u_col  = model(x_col, t_col)
        u_x    = torch.autograd.grad(u_col, x_col, torch.ones_like(u_col),
                                     create_graph=True, retain_graph=True)[0]
        u_t    = torch.autograd.grad(u_col, t_col, torch.ones_like(u_col),
                                     create_graph=True, retain_graph=True)[0]
        u_xx   = torch.autograd.grad(u_x,   x_col, torch.ones_like(u_x),
                                     create_graph=True, retain_graph=True)[0]
        res    = u_t + u_col * u_x - model.nu * u_xx

        l_pde  = torch.mean(res ** 2)
        l_ic   = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
        l_bc   = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
        l_data = torch.mean((model(x_s, t_s) - u_s) ** 2)

        total = (LAMBDA_PDE  * l_pde +
                 LAMBDA_IC   * l_ic  +
                 LAMBDA_BC   * l_bc  +
                 LAMBDA_DATA * l_data)

        total.backward()
        optimiser.step()
        scheduler.step(total.detach())

        hist_nu[ep]    = model.nu_value()
        hist_pde[ep]   = l_pde.item()
        hist_ic[ep]    = l_ic.item()
        hist_bc[ep]    = l_bc.item()
        hist_data[ep]  = l_data.item()
        hist_total[ep] = total.item()

        if verbose and (ep + 1) % 1000 == 0:
            print(f"  [{label} seed={seed}] ep {ep+1:>6d}  "
                  f"tot={total.item():.3e}  pde={l_pde.item():.3e}  "
                  f"data={l_data.item():.3e}  nu={hist_nu[ep]:.6f}")

    elapsed = time.time() - t0
    nu_final = model.nu_value()
    print(f"  [{label} seed={seed}] DONE {elapsed:.0f}s  "
          f"nu_final={nu_final:.6f}  err={abs(nu_final-NU_TRUE)/NU_TRUE*100:.1f}%")

    return dict(
        seed=seed, label=label, n_epochs=n_epochs,
        nu_final=nu_final, elapsed=elapsed,
        hist_nu=hist_nu, hist_pde=hist_pde, hist_ic=hist_ic,
        hist_bc=hist_bc, hist_data=hist_data, hist_total=hist_total,
    )


# ================================================================== #
#  Plotting helpers                                                    #
# ================================================================== #

def plot_nu_trajectory(results, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        ax.plot(np.arange(1, r["n_epochs"]+1), r["hist_nu"],
                alpha=0.8, lw=1.3, label=f"member {r['seed']}")
    ax.axhline(NU_TRUE, color="red",    lw=2,   ls="--", label=f"nu true = {NU_TRUE:.5f}")
    ax.axhline(NU_INIT, color="orange", lw=1.5, ls=":",  label=f"nu init = {NU_INIT:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("nu estimate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {save_path}")


def plot_loss_components(r, save_path):
    ep = np.arange(1, r["n_epochs"]+1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(ep, r["hist_pde"],  label=f"PDE residual  (x{LAMBDA_PDE:.0f})",  lw=1.2)
    ax.plot(ep, r["hist_ic"],   label=f"IC loss  (x{LAMBDA_IC:.0f})",        lw=1.2)
    ax.plot(ep, r["hist_bc"],   label=f"BC loss  (x{LAMBDA_BC:.0f})",        lw=1.2)
    ax.plot(ep, r["hist_data"], label=f"Data loss  (x{LAMBDA_DATA:.0f})",    lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (raw, unweighted)")
    ax.set_title("Loss components -- raw values")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ep, LAMBDA_PDE  * r["hist_pde"],  label="lam*PDE",  lw=1.2)
    ax.plot(ep, LAMBDA_IC   * r["hist_ic"],   label="lam*IC",   lw=1.2)
    ax.plot(ep, LAMBDA_BC   * r["hist_bc"],   label="lam*BC",   lw=1.2)
    ax.plot(ep, LAMBDA_DATA * r["hist_data"], label="lam*Data", lw=1.2)
    ax.plot(ep, r["hist_total"], label="Total", lw=1.8, ls="--", color="black")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (weighted)")
    ax.set_title("Loss components -- weighted contributions to total")
    ax.legend(fontsize=8)

    fig.suptitle(f"Loss breakdown -- seed {r['seed']}  "
                 f"(noise={NOISE_FRAC:.1%}, sensors={N_SENSORS})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {save_path}")


def plot_experiment_comparison(baseline, exp_a, exp_b, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    groups = [
        (baseline, f"Baseline ({N_EPOCHS:,} epochs, LR x1.0)"),
        (exp_a,    f"Exp A -- 2x epochs ({2*N_EPOCHS:,})"),
        (exp_b,    f"Exp B -- nu-LR div 10 ({N_EPOCHS:,} epochs)"),
    ]
    for ax, (results, title) in zip(axes, groups):
        for r in results:
            ax.plot(np.arange(1, r["n_epochs"]+1), r["hist_nu"],
                    alpha=0.8, lw=1.2, label=f"seed {r['seed']}")
        ax.axhline(NU_TRUE, color="red",    lw=2,   ls="--", label=f"nu true={NU_TRUE:.5f}")
        ax.axhline(NU_INIT, color="orange", lw=1.5, ls=":",  label=f"nu init={NU_INIT:.3f}")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("nu estimate")
    fig.suptitle(f"Inverse problem -- nu recovery experiments  "
                 f"(noise={NOISE_FRAC:.1%}, sensors={N_SENSORS})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {save_path}")


# ================================================================== #
#  Main                                                               #
# ================================================================== #

def _header(msg):
    print(f"\n{'='*68}\n  {msg}\n{'='*68}")


def main():
    _header("INVERSE PROBLEM DIAGNOSTIC -- noise=0.5%, sensors=100")

    raw_init = math.log(math.expm1(NU_INIT))
    raw_true = math.log(math.expm1(NU_TRUE))
    print(f"\n  nu true      = {NU_TRUE:.8f}  ({NU_TRUE:.6e})")
    print(f"  nu init      = {NU_INIT:.8f}  (ratio nu_init/nu_true = {NU_INIT/NU_TRUE:.1f}x)")
    print(f"  Parameterisation: nu = softplus(raw_nu)  -- strictly positive, smooth")
    print(f"  raw_nu init  = {raw_init:.5f}  (inverse softplus of {NU_INIT})")
    print(f"  raw_nu true  = {raw_true:.5f}  (inverse softplus of {NU_TRUE:.6f})")
    print(f"  delta raw_nu to cover = {raw_init - raw_true:.4f}")
    print(f"\n  Optimiser  : Adam, single param group")
    print(f"  LR (all params, including raw_nu) = {LR:.0e}")
    print(f"  ReduceLROnPlateau: factor=0.5, patience=1500")
    print(f"  Loss weights: lam_pde={LAMBDA_PDE}, lam_ic={LAMBDA_IC}, "
          f"lam_bc={LAMBDA_BC}, lam_data={LAMBDA_DATA}")
    print(f"\n  -> nu and network weights share the SAME learning rate.")
    print(f"  -> lam_data ({LAMBDA_DATA}) is {int(LAMBDA_DATA/LAMBDA_PDE)}x lam_pde -- data loss heavily upweighted.")

    # ---- Baseline: 3 members, full per-epoch history ----
    _header("STEP 1 -- Baseline: 3 members, full per-epoch history")
    baseline = []
    for seed in range(N_MEMBERS):
        r = train_full_history(seed=seed, n_epochs=N_EPOCHS, lr=LR,
                               nu_lr_factor=1.0, label="baseline")
        baseline.append(r)

    plot_nu_trajectory(
        baseline,
        title=f"nu recovery trajectory -- baseline (noise={NOISE_FRAC:.1%}, sensors={N_SENSORS})",
        save_path=os.path.join(OUT_DIR, "nu_trajectory_baseline.png"),
    )
    plot_loss_components(baseline[0],
                         save_path=os.path.join(OUT_DIR, "loss_components_seed0.png"))

    # ---- Loss component analysis ----
    _header("STEP 2 -- Loss component analysis (baseline, last epoch, seed=0)")
    r0 = baseline[0]
    ep_last = N_EPOCHS - 1
    pde_raw   = r0["hist_pde"][ep_last]
    data_raw  = r0["hist_data"][ep_last]
    ic_raw    = r0["hist_ic"][ep_last]
    bc_raw    = r0["hist_bc"][ep_last]
    pde_wtd   = LAMBDA_PDE  * pde_raw
    data_wtd  = LAMBDA_DATA * data_raw
    ic_wtd    = LAMBDA_IC   * ic_raw
    bc_wtd    = LAMBDA_BC   * bc_raw
    total_wtd = r0["hist_total"][ep_last]

    print(f"\n  At epoch {N_EPOCHS} (seed=0):")
    print(f"    PDE residual  raw  = {pde_raw:.3e}   weighted = {pde_wtd:.3e}  "
          f"({100*pde_wtd/total_wtd:.1f}% of total)")
    print(f"    Data loss     raw  = {data_raw:.3e}   weighted = {data_wtd:.3e}  "
          f"({100*data_wtd/total_wtd:.1f}% of total)")
    print(f"    IC loss       raw  = {ic_raw:.3e}   weighted = {ic_wtd:.3e}  "
          f"({100*ic_wtd/total_wtd:.1f}% of total)")
    print(f"    BC loss       raw  = {bc_raw:.3e}   weighted = {bc_wtd:.3e}  "
          f"({100*bc_wtd/total_wtd:.1f}% of total)")
    print(f"    Total (weighted)   = {total_wtd:.3e}")
    print(f"\n  PDE/Data ratio (weighted): {pde_wtd/data_wtd:.2f}x")
    print(f"  -> If >> 1: PDE term dominates; data cannot pull nu down.")
    print(f"  -> If << 1: data satisfied but PDE is not constraining nu.")

    # Is nu still moving at the end?
    window = 500
    nu_end_mean  = r0["hist_nu"][-window:].mean()
    nu_end_std   = r0["hist_nu"][-window:].std()
    nu_slope_ep  = r0["hist_nu"][-1] - r0["hist_nu"][-window]
    print(f"\n  nu in last {window} epochs (seed=0):")
    print(f"    mean = {nu_end_mean:.6f}   std = {nu_end_std:.6f}")
    print(f"    change over last {window} epochs = {nu_slope_ep:+.6f}")
    still_moving = abs(nu_slope_ep) > 5 * nu_end_std
    print(f"    {'STILL MOVING' if still_moving else 'PLATEAUED'} "
          f"(|delta| {'>' if still_moving else '<='} 5*std)")

    # ---- Experiment A: 2x epochs ----
    _header("STEP 3 -- Experiment A: 2x epochs (16,000)")
    exp_a = []
    for seed in range(N_MEMBERS):
        r = train_full_history(seed=seed, n_epochs=2*N_EPOCHS, lr=LR,
                               nu_lr_factor=1.0, label="expA_16k")
        exp_a.append(r)

    # ---- Experiment B: nu-LR / 10 ----
    _header("STEP 4 -- Experiment B: nu-LR divided by 10")
    exp_b = []
    for seed in range(N_MEMBERS):
        r = train_full_history(seed=seed, n_epochs=N_EPOCHS, lr=LR,
                               nu_lr_factor=0.1, label="expB_nuLR0.1")
        exp_b.append(r)

    # ---- Comparison plot ----
    plot_experiment_comparison(
        baseline, exp_a, exp_b,
        save_path=os.path.join(OUT_DIR, "experiment_comparison.png"),
    )

    # ---- Summary table ----
    _header("SUMMARY -- nu recovery results")
    print(f"\n  nu true = {NU_TRUE:.6f}")
    header = f"  {'Experiment':<35} {'seed':>4}  {'nu final':>10}  {'err%':>7}  {'plateaued?':>12}"
    print(f"\n{header}")
    print(f"  {'-'*72}")

    def _summarise(results, label):
        for r in results:
            err = abs(r["nu_final"] - NU_TRUE) / NU_TRUE * 100
            delta_tail = abs(r["hist_nu"][-1] - r["hist_nu"][-min(1000, r["n_epochs"]//4)])
            conv = "yes" if delta_tail < 0.01 * max(r["nu_final"], 1e-9) else "no (drift)"
            print(f"  {label:<35} {r['seed']:>4}  {r['nu_final']:>10.6f}  {err:>7.1f}%  {conv:>12}")

    _summarise(baseline, f"Baseline ({N_EPOCHS:,} ep, LR x1)")
    _summarise(exp_a,    f"Exp A ({2*N_EPOCHS:,} ep, LR x1)")
    _summarise(exp_b,    f"Exp B ({N_EPOCHS:,} ep, nu-LR /10)")

    # ---- Diagnosis ----
    _header("DIAGNOSIS")
    base_nu_mean = np.mean([r["nu_final"] for r in baseline])
    expa_nu_mean = np.mean([r["nu_final"] for r in exp_a])
    expb_nu_mean = np.mean([r["nu_final"] for r in exp_b])

    base_err = abs(base_nu_mean - NU_TRUE) / NU_TRUE * 100
    expa_err = abs(expa_nu_mean - NU_TRUE) / NU_TRUE * 100
    expb_err = abs(expb_nu_mean - NU_TRUE) / NU_TRUE * 100

    print(f"\n  Baseline mean nu = {base_nu_mean:.6f}  (err {base_err:.1f}%)")
    print(f"  Exp A    mean nu = {expa_nu_mean:.6f}  (err {expa_err:.1f}%)")
    print(f"  Exp B    mean nu = {expb_nu_mean:.6f}  (err {expb_err:.1f}%)")

    verdict_a = expa_err < 0.7 * base_err
    verdict_b = expb_err < 0.7 * base_err

    print(f"\n  [{'YES' if verdict_a else 'NO '}] 2x epochs substantially reduces error "
          f"-> {'UNDERTRAINED' if verdict_a else 'not simply undertrained'}")
    print(f"  [{'YES' if verdict_b else 'NO '}] Lower nu-LR substantially reduces error "
          f"-> {'LR ISSUE for nu' if verdict_b else 'not a pure LR overshoot issue'}")

    print(f"\n  Output files in: {OUT_DIR}/")
    for f in ["nu_trajectory_baseline.png", "loss_components_seed0.png",
              "experiment_comparison.png"]:
        path = os.path.join(OUT_DIR, f)
        tag  = "OK" if os.path.isfile(path) else "MISSING"
        print(f"    [{tag}] {f}")


if __name__ == "__main__":
    main()
