"""
train.py
--------
Training loop for the inverse Burgers PINN.

Two-phase schedule
------------------
Phase 1 — Adam (epochs 1 … adam_epochs):
    Collocation points are resampled every epoch so the optimiser explores
    the full domain.  A ReduceLROnPlateau scheduler is active.

Phase 2 — L-BFGS (epochs adam_epochs+1 … n_epochs):
    A FIXED collocation set (sampled once before phase 2 begins) is used so
    the loss landscape is deterministic and the strong-Wolfe line search stays
    stable.  The Adam scheduler is disabled.

Lambda-data auto-balancing
--------------------------
Before training begins we evaluate l_pde and l_data at the initial parameter
values and set:

    lambda_data = lambda_pde * l_pde0 / l_data0

so that the weighted PDE term and weighted data term start at the same scale.
The caller-supplied lambda_data is used only as a fallback if l_data0 == 0.

Total loss
----------
    L = λ_pde * L_pde  +  λ_ic * L_ic  +  λ_bc * L_bc  +  λ_data * L_data

ν is updated jointly with the network weights by both optimisers.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn

from model import InverseBurgersPINN
from data import (
    sample_collocation_points,
    sample_initial_condition_points,
    sample_boundary_condition_points,
    make_sensor_data,
)


# ------------------------------------------------------------------ #
#  PDE residual (uses model.nu — the learnable viscosity)             #
# ------------------------------------------------------------------ #

def pde_residual(
    model: InverseBurgersPINN,
    x:     torch.Tensor,
    t:     torch.Tensor,
) -> torch.Tensor:
    """
    Burgers residual:  f = u_t + u·u_x − ν·u_xx
    ν comes from model.nu (a softplus-transformed learnable parameter).
    """
    u = model(x, t)

    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]

    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True)[0]

    return u_t + u * u_x - model.nu * u_xx


# ------------------------------------------------------------------ #
#  Individual loss terms                                               #
# ------------------------------------------------------------------ #

def loss_pde(model, x_col, t_col):
    return torch.mean(pde_residual(model, x_col, t_col) ** 2)


def loss_ic(model, x_ic, t_ic, u_ic):
    return torch.mean((model(x_ic, t_ic) - u_ic) ** 2)


def loss_bc(model, x_bc, t_bc, u_bc):
    return torch.mean((model(x_bc, t_bc) - u_bc) ** 2)


def loss_data(model, x_s, t_s, u_s):
    """MSE between PINN output and noisy sensor readings."""
    return torch.mean((model(x_s, t_s) - u_s) ** 2)


# ------------------------------------------------------------------ #
#  Shared loss computation helper                                      #
# ------------------------------------------------------------------ #

def _compute_losses(model, xc, tc, x_ic, t_ic, u_ic,
                    x_bc, t_bc, u_bc, x_s, t_s, u_s,
                    lambda_pde, lambda_ic, lambda_bc, lambda_data):
    """Return (total, l_pde, l_ic, l_bc, l_data) as tensors."""
    l_pde  = loss_pde(model, xc, tc)
    l_ic   = loss_ic(model, x_ic, t_ic, u_ic)
    l_bc   = loss_bc(model, x_bc, t_bc, u_bc)
    l_data = loss_data(model, x_s, t_s, u_s)
    total  = (lambda_pde  * l_pde +
              lambda_ic   * l_ic  +
              lambda_bc   * l_bc  +
              lambda_data * l_data)
    return total, l_pde, l_ic, l_bc, l_data


# ------------------------------------------------------------------ #
#  Main training function                                              #
# ------------------------------------------------------------------ #

def train(
    # Sensor data
    n_sensors:   int   = 50,
    noise_frac:  float = 0.01,
    sensor_seed: int   = 0,
    # Architecture
    n_hidden:    int   = 4,
    n_neurons:   int   = 50,
    nu_init:     float = 0.03,   # intentionally wrong starting guess
    # Training schedule
    n_col:       int   = 10_000,
    n_ic:        int   = 200,
    n_bc:        int   = 200,
    n_epochs:    int   = 3_500,  # total epochs (Adam + L-BFGS)
    adam_epochs: int   = 2_000,  # how many of those are Adam; rest → L-BFGS
    lr:          float = 1e-3,
    lr_lbfgs:    float = 1.0,
    print_every: int   = 500,
    # Loss weights (lambda_data will be auto-balanced at init)
    lambda_pde:  float = 1.0,
    lambda_ic:   float = 10.0,
    lambda_bc:   float = 10.0,
    lambda_data: float = 100.0,  # fallback if auto-balance cannot run
    # Misc
    save_path:   str   = "outputs/inverse_pinn.pt",
    device_str:  str   = "auto",
    verbose:     bool  = True,
) -> dict:
    """
    Train the inverse PINN with a two-phase Adam → L-BFGS schedule and return
    a results dict containing:
      - model        : trained InverseBurgersPINN
      - nu_history   : list of (epoch, nu_estimate) pairs — one per log step
      - train_time   : wall-clock seconds
      - nu_final     : final recovered ν value
      - loss_history : dict of lists (pde, ic, bc, data, total)
    """

    # ---- Device ----
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"[train] device={device}  sensors={n_sensors}  noise={noise_frac:.1%}"
              f"  nu_init={nu_init:.5f}  adam_epochs={adam_epochs}/{n_epochs}")

    # ---- Model ----
    model = InverseBurgersPINN(n_hidden=n_hidden, n_neurons=n_neurons,
                                nu_init=nu_init).to(device)

    # ---- Fixed training data ----
    x_ic, t_ic, u_ic = sample_initial_condition_points(n_ic, device)
    x_bc, t_bc, u_bc = sample_boundary_condition_points(n_bc, device)
    x_s,  t_s,  u_s  = make_sensor_data(n_sensors, noise_frac, sensor_seed, device)

    # ---- Auto-balance lambda_data ----------------------------------------
    # Evaluate l_pde and l_data once with a temporary collocation set so that
    # lambda_pde * l_pde  ≈  lambda_data * l_data  at epoch 0.
    with torch.no_grad():
        xc0, tc0 = sample_collocation_points(n_col, device)
    # Need gradients w.r.t. x/t for the PDE residual, but NOT model params.
    model.eval()
    with torch.enable_grad():
        xc0.requires_grad_(True)
        tc0.requires_grad_(True)
        l_pde0  = loss_pde(model, xc0, tc0).item()
        l_data0 = loss_data(model, x_s, t_s, u_s).item()
    model.train()

    if l_data0 > 0:
        lambda_data = lambda_pde * l_pde0 / l_data0
    # else: keep the caller-supplied fallback

    if verbose:
        print(f"[train] auto-balanced lambda_data={lambda_data:.4f}"
              f"  (l_pde0={l_pde0:.3e}  l_data0={l_data0:.3e})")

    # ---- Optimisers ----
    adam  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        adam, mode="min", factor=0.5, patience=1500)

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lr_lbfgs,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    # Pre-sample the FIXED collocation set used throughout the L-BFGS phase.
    xc_lbfgs, tc_lbfgs = sample_collocation_points(n_col, device)

    # ---- History buffers ----
    nu_history:     list[tuple[int, float]] = []
    loss_hist_pde:  list[float] = []
    loss_hist_ic:   list[float] = []
    loss_hist_bc:   list[float] = []
    loss_hist_data: list[float] = []
    loss_hist_tot:  list[float] = []

    # Closure state for L-BFGS (updated each step)
    _cs: dict = {}

    def lbfgs_closure():
        lbfgs.zero_grad()
        total, *_ = _compute_losses(
            model, _cs["xc"], _cs["tc"],
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_s, t_s, u_s,
            lambda_pde, lambda_ic, lambda_bc, lambda_data,
        )
        total.backward()
        return total

    t_start = time.time()
    nan_triggered = False

    for epoch in range(1, n_epochs + 1):
        model.train()

        if epoch <= adam_epochs:
            # ── Phase 1: Adam — resample collocation each epoch ──────────────
            xc, tc = sample_collocation_points(n_col, device)
            adam.zero_grad()
            total, l_pde, l_ic, l_bc, l_data = _compute_losses(
                model, xc, tc,
                x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_s, t_s, u_s,
                lambda_pde, lambda_ic, lambda_bc, lambda_data,
            )
            total.backward()
            adam.step()
            scheduler.step(total.detach())

        else:
            # ── Phase 2: L-BFGS — fixed collocation set ──────────────────────
            if nan_triggered:
                if epoch % print_every == 0:
                    print(f"  ep {epoch:>6d} | [NaN — L-BFGS halted]"
                          f"  nu={model.nu_value():.6f}")
                continue

            _cs["xc"] = xc_lbfgs
            _cs["tc"] = tc_lbfgs
            lbfgs.step(lbfgs_closure)

            # Re-evaluate for logging (no grad needed for scalars)
            with torch.enable_grad():
                total, l_pde, l_ic, l_bc, l_data = _compute_losses(
                    model, xc_lbfgs, tc_lbfgs,
                    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_s, t_s, u_s,
                    lambda_pde, lambda_ic, lambda_bc, lambda_data,
                )

            if not torch.isfinite(total):
                nan_triggered = True
                if verbose:
                    print(f"  [ep {epoch}] NaN detected — halting L-BFGS, "
                          f"holding nu={model.nu_value():.6f}")
                continue

        # ── Logging ──────────────────────────────────────────────────────────
        if epoch % print_every == 0 or epoch == 1:
            nu_val = model.nu_value()
            nu_history.append((epoch, nu_val))
            loss_hist_pde.append(l_pde.item())
            loss_hist_ic.append(l_ic.item())
            loss_hist_bc.append(l_bc.item())
            loss_hist_data.append(l_data.item())
            loss_hist_tot.append(total.item())

            if verbose:
                phase  = "Adam" if epoch <= adam_epochs else "LBFGS"
                lr_now = adam.param_groups[0]["lr"]
                print(
                    f"  ep {epoch:>6d} [{phase:>5}] | "
                    f"tot={total.item():.3e}  "
                    f"pde={l_pde.item():.3e}  "
                    f"data={l_data.item():.3e}  "
                    f"nu={nu_val:.6f}  "
                    f"lr={lr_now:.1e}"
                )

    train_time = time.time() - t_start
    nu_final   = model.nu_value()

    if verbose:
        from data import NU_TRUE
        err_pct = abs(nu_final - NU_TRUE) / NU_TRUE * 100
        print(f"[train] Done in {train_time:.1f}s  "
              f"nu_final={nu_final:.6f}  "
              f"nu_true={NU_TRUE:.6f}  "
              f"err={err_pct:.2f}%")

    # ---- Save checkpoint ----
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "state_dict":  model.state_dict(),
        "nu_final":    nu_final,
        "n_hidden":    n_hidden,
        "n_neurons":   n_neurons,
        "nu_history":  nu_history,
    }, save_path)

    return dict(
        model        = model,
        nu_history   = nu_history,       # list[(epoch, nu)]
        nu_final     = nu_final,
        train_time   = train_time,
        loss_history = dict(
            pde   = loss_hist_pde,
            ic    = loss_hist_ic,
            bc    = loss_hist_bc,
            data  = loss_hist_data,
            total = loss_hist_tot,
        ),
        n_sensors    = n_sensors,
        noise_frac   = noise_frac,
        nu_init      = nu_init,
    )
