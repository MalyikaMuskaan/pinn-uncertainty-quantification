"""
train.py
--------
Training loop for the 2-D Darcy flow PINN.

PDE
---
    -div( k(x,y) * grad(u(x,y)) ) = f(x,y)   on [0,1]²
    u = 0  on ∂Ω

Expanding the strong form used as the residual:

    R(x,y) = -(k_x·u_x + k·u_xx + k_y·u_y + k·u_yy) - f(x,y)

All partial derivatives are computed exactly via PyTorch autograd.

Loss
----
    L = λ_pde · L_pde  +  λ_bc · L_bc

    L_pde = mean( R(x,y)² )   over collocation points
    L_bc  = mean( u(x_b,y_b)² ) over boundary points (target = 0)

λ_bc is auto-balanced at initialisation so that λ_pde·L_pde ≈ λ_bc·L_bc,
following the same pattern proven effective in inverse_problem/train.py.

Optimisation schedule
---------------------
Phase 1 — Adam (adam_epochs epochs):
    Collocation points re-sampled every epoch for domain coverage.
    ReduceLROnPlateau scheduler halves lr on stagnation.

Phase 2 — L-BFGS (remaining epochs):
    Fixed collocation set (stable landscape for strong-Wolfe line search).
    Same configuration as inverse_problem/train.py — proven to sharpen
    converged accuracy by 30-60% on the Burgers inverse problem.

This is a pure forward problem (no unknown parameters), so there is no
separate parameter group — all network weights are updated uniformly.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from model import DarcyPINN
from data import (
    sample_collocation_points,
    sample_boundary_points,
    source_term,
    permeability,
    PI,
)


# ------------------------------------------------------------------ #
#  PDE residual                                                        #
# ------------------------------------------------------------------ #

def pde_residual(
    model: DarcyPINN,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Darcy residual at collocation points using autograd.

    Residual:
        R = -(k_x·u_x + k·u_xx + k_y·u_y + k·u_yy) - f

    where k = permeability(x,y) and f = source_term(x,y).

    Parameters
    ----------
    model : DarcyPINN — the network approximating u(x,y)
    x, y  : (N, 1) tensors, requires_grad=True

    Returns
    -------
    R : (N, 1) residual tensor
    """
    u = model(x, y)    # (N, 1)

    # First-order spatial derivatives
    # create_graph=True: retain the graph so we can differentiate again for u_xx / u_yy
    grad_ones = torch.ones_like(u)

    u_x = torch.autograd.grad(
        u, x, grad_outputs=grad_ones,
        create_graph=True, retain_graph=True,
    )[0]   # (N, 1)

    u_y = torch.autograd.grad(
        u, y, grad_outputs=grad_ones,
        create_graph=True, retain_graph=True,
    )[0]   # (N, 1)

    # Second-order spatial derivatives
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True,
    )[0]   # (N, 1)

    u_yy = torch.autograd.grad(
        u_y, y, grad_outputs=torch.ones_like(u_y),
        create_graph=True, retain_graph=True,
    )[0]   # (N, 1)

    # Permeability and its gradients (analytical, no autograd needed)
    # k = 1 + 0.5·sin(πx)·sin(πy)
    # k_x = 0.5π·cos(πx)·sin(πy)
    # k_y = 0.5π·sin(πx)·cos(πy)
    # Using detach() avoids building a second graph through k.
    sin_px = torch.sin(PI * x).detach()
    sin_py = torch.sin(PI * y).detach()
    cos_px = torch.cos(PI * x).detach()
    cos_py = torch.cos(PI * y).detach()

    k   = 1.0 + 0.5 * sin_px * sin_py
    k_x = 0.5 * PI * cos_px * sin_py
    k_y = 0.5 * PI * sin_px * cos_py

    # Source term f(x,y) — computed analytically, no graph needed
    f = source_term(x.detach(), y.detach())

    # -div(k grad u) - f  =  -(k_x u_x + k u_xx + k_y u_y + k u_yy) - f
    R = -(k_x * u_x + k * u_xx + k_y * u_y + k * u_yy) - f

    return R


# ------------------------------------------------------------------ #
#  Individual loss terms                                               #
# ------------------------------------------------------------------ #

def loss_pde(model: DarcyPINN, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean-squared PDE residual over all collocation points."""
    return torch.mean(pde_residual(model, x, y) ** 2)


def loss_bc(
    model: DarcyPINN,
    x_bc: torch.Tensor,
    y_bc: torch.Tensor,
    u_bc: torch.Tensor,
) -> torch.Tensor:
    """MSE between network output and zero Dirichlet boundary values."""
    return torch.mean((model(x_bc, y_bc) - u_bc) ** 2)


# ------------------------------------------------------------------ #
#  Main training function                                              #
# ------------------------------------------------------------------ #

def train(
    # Architecture
    n_hidden:    int   = 5,
    n_neurons:   int   = 64,
    # Training data sizes
    n_col:       int   = 10_000,   # collocation points (re-sampled each Adam epoch)
    n_per_edge:  int   = 200,      # boundary points per edge (total = 4 × n_per_edge)
    # Training schedule
    n_epochs:    int   = 5_000,    # total epochs (Adam + L-BFGS)
    adam_epochs: int   = 3_000,    # Adam phase; remaining → L-BFGS
    lr:          float = 1e-3,     # Adam initial learning rate
    lr_lbfgs:    float = 1.0,      # L-BFGS step size
    print_every: int   = 500,
    # Loss weights
    lambda_pde:  float = 1.0,
    lambda_bc:   float = 10.0,     # fallback; will be auto-balanced at init
    # Misc
    save_path:   str   = "outputs/darcy_pinn.pt",
    device_str:  str   = "auto",
    verbose:     bool  = True,
) -> dict:
    """
    Train the Darcy PINN with a two-phase Adam → L-BFGS schedule.

    Returns a dict containing:
        model        : trained DarcyPINN
        train_time   : wall-clock seconds
        loss_history : dict of lists — pde, bc, total (one value per log step)
    """

    # ---- Device --------------------------------------------------------
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"[train] device={device}  n_col={n_col}  n_per_edge={n_per_edge}")
        print(f"[train] schedule: Adam({adam_epochs}) + L-BFGS({n_epochs - adam_epochs})"
              f"  total={n_epochs} epochs")

    # ---- Model ---------------------------------------------------------
    model = DarcyPINN(n_hidden=n_hidden, n_neurons=n_neurons).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[train] DarcyPINN: {n_params:,} parameters")

    # ---- Fixed training data -------------------------------------------
    x_bc, y_bc, u_bc = sample_boundary_points(n_per_edge, device)

    # ---- Auto-balance λ_bc ---------------------------------------------
    # Evaluate l_pde and l_bc once at random init so that
    #   lambda_pde * l_pde_0  ≈  lambda_bc * l_bc_0
    model.eval()
    with torch.no_grad():
        xc0_np = np.random.uniform(0.0, 1.0, (n_col, 1)).astype(np.float32)
        yc0_np = np.random.uniform(0.0, 1.0, (n_col, 1)).astype(np.float32)
        xc0 = torch.tensor(xc0_np, device=device)
        yc0 = torch.tensor(yc0_np, device=device)
    model.train()

    # PDE loss at init requires gradients w.r.t. coordinates
    with torch.enable_grad():
        xc0_g = xc0.detach().requires_grad_(True)
        yc0_g = yc0.detach().requires_grad_(True)
        l_pde0 = loss_pde(model, xc0_g, yc0_g).item()

    with torch.no_grad():
        l_bc0 = loss_bc(model, x_bc, y_bc, u_bc).item()

    if l_bc0 > 0:
        lambda_bc = lambda_pde * l_pde0 / l_bc0
    # else: keep caller-supplied fallback

    if verbose:
        print(f"[train] auto-balanced lambda_bc={lambda_bc:.4f}"
              f"  (l_pde0={l_pde0:.3e}  l_bc0={l_bc0:.3e})")

    # ---- Optimisers ----------------------------------------------------
    adam = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        adam, mode="min", factor=0.5, patience=1500,
    )

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

    # Fixed collocation set for L-BFGS phase (deterministic landscape)
    xc_lbfgs, yc_lbfgs = sample_collocation_points(n_col, device)

    # ---- History buffers -----------------------------------------------
    loss_hist_pde:   list[float] = []
    loss_hist_bc:    list[float] = []
    loss_hist_total: list[float] = []

    # Closure state (mutable dict avoids nonlocal)
    _cs: dict = {}

    def lbfgs_closure() -> torch.Tensor:
        lbfgs.zero_grad()
        l_p  = loss_pde(model, _cs["x"], _cs["y"])
        l_b  = loss_bc(model, x_bc, y_bc, u_bc)
        tot  = lambda_pde * l_p + lambda_bc * l_b
        tot.backward()
        return tot

    # ---- Training loop -------------------------------------------------
    t_start = time.time()
    nan_triggered = False

    for epoch in range(1, n_epochs + 1):
        model.train()

        if epoch <= adam_epochs:
            # Phase 1 — Adam: re-sample collocation each epoch
            xc, yc = sample_collocation_points(n_col, device)
            adam.zero_grad()
            l_p   = loss_pde(model, xc, yc)
            l_b   = loss_bc(model, x_bc, y_bc, u_bc)
            total = lambda_pde * l_p + lambda_bc * l_b
            total.backward()
            adam.step()
            scheduler.step(total.detach())

        else:
            # Phase 2 — L-BFGS: fixed collocation
            if nan_triggered:
                continue

            _cs["x"] = xc_lbfgs
            _cs["y"] = yc_lbfgs
            lbfgs.step(lbfgs_closure)

            # Re-evaluate for logging
            with torch.enable_grad():
                l_p   = loss_pde(model, xc_lbfgs, yc_lbfgs)
                l_b   = loss_bc(model, x_bc, y_bc, u_bc)
                total = lambda_pde * l_p + lambda_bc * l_b

            if not torch.isfinite(total):
                nan_triggered = True
                if verbose:
                    print(f"  [ep {epoch}] NaN detected — halting L-BFGS")
                continue

        # ---- Logging ---------------------------------------------------
        if epoch % print_every == 0 or epoch == 1:
            loss_hist_pde.append(l_p.item())
            loss_hist_bc.append(l_b.item())
            loss_hist_total.append(total.item())

            if verbose:
                phase  = "Adam " if epoch <= adam_epochs else "LBFGS"
                lr_now = adam.param_groups[0]["lr"]
                print(
                    f"  ep {epoch:>6d} [{phase}] | "
                    f"total={total.item():.3e}  "
                    f"pde={l_p.item():.3e}  "
                    f"bc={l_b.item():.3e}  "
                    f"lr={lr_now:.1e}"
                )

    train_time = time.time() - t_start

    if verbose:
        print(f"[train] Finished in {train_time:.1f}s")

    # ---- Save checkpoint -----------------------------------------------
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "state_dict":    model.state_dict(),
        "n_hidden":      n_hidden,
        "n_neurons":     n_neurons,
        "loss_history":  dict(
            pde   = loss_hist_pde,
            bc    = loss_hist_bc,
            total = loss_hist_total,
        ),
        "train_time":    train_time,
    }, save_path)
    if verbose:
        print(f"[train] Checkpoint saved to '{save_path}'")

    return dict(
        model        = model,
        train_time   = train_time,
        loss_history = dict(
            pde   = loss_hist_pde,
            bc    = loss_hist_bc,
            total = loss_hist_total,
        ),
    )
