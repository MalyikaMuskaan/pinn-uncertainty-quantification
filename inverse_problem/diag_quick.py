"""
diag_quick.py  — 1 member, 8000 epochs, print every 200 epochs.
Strategy: Adam pre-conditioning (ep 1-4000) → L-BFGS refinement (ep 4001-8000).
  - lambda_data auto-balanced at init so lam_pde*l_pde == lam_data*l_data
  - L-BFGS operates on all params (network + nu) — standard PINN practice
  - Collocation points FIXED for entire L-BFGS phase (no resampling) so the
    loss landscape is smooth and the strong-Wolfe line search doesn't diverge
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (NU_TRUE, sample_collocation_points,
    sample_initial_condition_points, sample_boundary_condition_points,
    make_sensor_data)
from model import InverseBurgersPINN

N_SENSORS   = 100
NOISE_FRAC  = 0.005
NU_INIT     = 0.03
N_EPOCHS    = 8_000
ADAM_EPOCHS = 4_000   # switch to L-BFGS after this many Adam steps
LR_ADAM     = 1e-3
LR_LBFGS    = 1.0     # L-BFGS line-searches internally; 1.0 is standard
LP, LI, LB  = 1.0, 10.0, 10.0   # LD computed dynamically below
SEED        = 0

import math
raw_init = math.log(math.expm1(NU_INIT))
raw_true = math.log(math.expm1(NU_TRUE))
print(f"nu_true     = {NU_TRUE:.8f}")
print(f"nu_init     = {NU_INIT:.8f}  (ratio {NU_INIT/NU_TRUE:.1f}x true)")
print(f"raw_nu_init = {raw_init:.5f}  raw_nu_true = {raw_true:.5f}  delta = {raw_init-raw_true:.4f}")
print(f"Phase 1: Adam  lr={LR_ADAM:.0e}  ep 1-{ADAM_EPOCHS}")
print(f"Phase 2: L-BFGS  lr={LR_LBFGS}  ep {ADAM_EPOCHS+1}-{N_EPOCHS}")

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cpu")
model  = InverseBurgersPINN(4, 50, NU_INIT).to(device)

x_ic, t_ic, u_ic = sample_initial_condition_points(200, device)
x_bc, t_bc, u_bc = sample_boundary_condition_points(200, device)
x_s,  t_s,  u_s  = make_sensor_data(N_SENSORS, NOISE_FRAC, SEED, device)

# ── Auto-balance LD so LP*l_pde ≈ LD*l_data at initialisation ────────────────
with torch.enable_grad():
    xc0, tc0 = sample_collocation_points(10000, device)
    uc0  = model(xc0, tc0)
    ux0  = torch.autograd.grad(uc0, xc0, torch.ones_like(uc0),
                                create_graph=True, retain_graph=True)[0]
    ut0  = torch.autograd.grad(uc0, tc0, torch.ones_like(uc0),
                                create_graph=True, retain_graph=True)[0]
    uxx0 = torch.autograd.grad(ux0, xc0, torch.ones_like(ux0),
                                create_graph=False)[0]
    res0    = ut0 + uc0 * ux0 - model.nu * uxx0
    l_pde0  = torch.mean(res0 ** 2).item()
    l_data0 = torch.mean((model(x_s, t_s) - u_s) ** 2).item()

LD = LP * (l_pde0 / l_data0) if l_data0 > 0 else 1.0
print(f"Loss weights: lam_pde={LP}  lam_ic={LI}  lam_bc={LB}  lam_data={LD:.4f}"
      f"  (auto-balanced: l_pde0={l_pde0:.3e}  l_data0={l_data0:.3e})")
print()
print(f"  {'ep':>5}  {'opt':>6}  {'nu':>10}  {'lam*PDE':>10}  {'lam*Data':>10}"
      f"  {'lam*IC':>9}  {'lam*BC':>9}  {'Total':>10}  {'D/P%':>5}  {'|g_nu|':>8}")
print(f"  {'-'*110}")

# ── Helper: compute all loss terms given fixed collocation points ─────────────
def compute_losses(xc, tc):
    uc  = model(xc, tc)
    ux  = torch.autograd.grad(uc,  xc, torch.ones_like(uc),
                               create_graph=True, retain_graph=True)[0]
    ut  = torch.autograd.grad(uc,  tc, torch.ones_like(uc),
                               create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(ux,  xc, torch.ones_like(ux),
                               create_graph=True, retain_graph=True)[0]
    res    = ut + uc * ux - model.nu * uxx
    l_pde  = torch.mean(res ** 2)
    l_ic   = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
    l_bc   = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
    l_data = torch.mean((model(x_s,  t_s)  - u_s)  ** 2)
    return LP*l_pde + LI*l_ic + LB*l_bc + LD*l_data, l_pde, l_ic, l_bc, l_data

# ── Optimizers ────────────────────────────────────────────────────────────────
adam  = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=LR_LBFGS,
    max_iter=20,          # inner line-search iterations per step() call
    max_eval=25,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=50,
    line_search_fn="strong_wolfe",
)

# Storage for closure — L-BFGS closure re-evaluates with the same xc/tc
_closure_state: dict = {}

def lbfgs_closure():
    lbfgs.zero_grad()
    total, *_ = compute_losses(_closure_state["xc"], _closure_state["tc"])
    total.backward()
    return total

# ── Training loop ─────────────────────────────────────────────────────────────
# Pre-sample a FIXED collocation set for the L-BFGS phase so the loss
# landscape is deterministic and the strong-Wolfe line search stays stable.
xc_lbfgs, tc_lbfgs = sample_collocation_points(10000, device)

t0 = time.time()
nan_triggered = False

for ep in range(1, N_EPOCHS + 1):
    model.train()

    if ep <= ADAM_EPOCHS:
        # ── Phase 1: Adam (resample each epoch) ──────────────────────────────
        xc, tc = sample_collocation_points(10000, device)
        adam.zero_grad()
        total, l_pde, l_ic, l_bc, l_data = compute_losses(xc, tc)
        total.backward()
        g_nu = model.raw_nu.grad.norm().item() if model.raw_nu.grad is not None else 0.0
        adam.step()
    else:
        # ── Phase 2: L-BFGS (fixed collocation set) ──────────────────────────
        if nan_triggered:
            # NaN guard: skip step, keep last valid state for logging
            if ep % 200 == 0:
                phase = "LBFGS"
                print(f"  {ep:>5}  {phase:>6}  {'[NaN — stopped]':>10}")
            continue
        _closure_state["xc"] = xc_lbfgs
        _closure_state["tc"] = tc_lbfgs
        lbfgs.step(lbfgs_closure)
        # Re-evaluate for logging
        with torch.enable_grad():
            total, l_pde, l_ic, l_bc, l_data = compute_losses(xc_lbfgs, tc_lbfgs)
        g_nu = model.raw_nu.grad.norm().item() if model.raw_nu.grad is not None else 0.0
        # NaN check
        if not torch.isfinite(total):
            nan_triggered = True
            print(f"  [ep {ep}] NaN detected — halting L-BFGS, holding last valid nu="
                  f"{model.nu_value():.6f}")
            continue

    if ep % 200 == 0 or ep == 1:
        pde_w  = LP * l_pde.item()
        data_w = LD * l_data.item()
        ic_w   = LI * l_ic.item()
        bc_w   = LB * l_bc.item()
        tot    = total.item()
        dp_pct = 100.0 * data_w / pde_w if pde_w > 0 else float("nan")
        phase  = "Adam" if ep <= ADAM_EPOCHS else "LBFGS"
        print(f"  {ep:>5}  {phase:>6}  {model.nu_value():>10.6f}"
              f"  {pde_w:>10.3e}  {data_w:>10.3e}"
              f"  {ic_w:>9.3e}  {bc_w:>9.3e}  {tot:>10.3e}"
              f"  {dp_pct:>5.1f}  {g_nu:>8.4f}")

elapsed = time.time() - t0
nu_f = model.nu_value()
print(f"\n  Elapsed: {elapsed:.0f}s")
print(f"  nu_final = {nu_f:.6f}   err = {abs(nu_f-NU_TRUE)/NU_TRUE*100:.1f}%")
