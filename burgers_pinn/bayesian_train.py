"""
bayesian_train.py
-----------------
Training loop for the Bayesian PINN (mean-field VI) on the Burgers' equation.

Loss
----
The ELBO (Evidence Lower BOund) loss used in Bayes by Backprop is:

  L_total = L_pde + L_ic + L_bc + kl_weight * KL[q(w) || p(w)]

The first three terms are identical to train.py (same PDE residual, IC and BC
formulations) so the comparison with the Deep Ensemble is fair.

The KL term regularises the posterior weights toward the standard-Gaussian
prior.  kl_weight controls the trade-off:
  - Too large  → posterior collapses to the prior (underfitting)
  - Too small  → posterior ignores the prior (overfitting, no uncertainty)

We use the "data-size scaling" convention:
  kl_weight = 1 / n_col
This is the standard choice from Blundell et al. (2015) and ensures the KL
contribution scales correctly regardless of how many collocation points are used.

MC estimates of the data-fit losses
------------------------------------
Because the weights are random, each forward pass gives a different prediction.
We average n_mc_samples forward passes per epoch to get a low-variance gradient
estimate of the data-fit terms.  n_mc_samples=1 (default) is fine for training
but you can increase it for smoother loss curves.
"""

import os
import time
import torch
import numpy as np
from bayesian_model import BayesianBurgersPINN
from train import pde_residual, loss_initial_condition, loss_boundary_condition
from data import (
    sample_collocation_points,
    sample_initial_condition_points,
    sample_boundary_condition_points,
)

NU = 0.01 / 3.141592653589793


# ------------------------------------------------------------------ #
#  Bayesian PDE loss  (wraps the existing pde_residual from train.py) #
# ------------------------------------------------------------------ #

def bayes_loss_pde(
    model: BayesianBurgersPINN,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PDE residual loss + KL for one stochastic forward pass.

    The BayesianBurgersPINN.forward() draws a weight sample and returns (u, kl).
    We hook into it by wrapping the model so that pde_residual (which calls
    model(x, t)) receives the sampled u and ignores kl — then we collect kl
    separately with a second forward pass on the IC/BC path below.

    To avoid calling forward twice just for the KL, we use a small wrapper
    that caches the KL from the last call.

    Parameters
    ----------
    model  : BayesianBurgersPINN
    x_col  : (N,1) collocation x  (requires_grad=True)
    t_col  : (N,1) collocation t  (requires_grad=True)

    Returns
    -------
    l_pde : scalar PDE MSE loss
    kl    : scalar KL term from this weight sample
    """
    # Forward pass — returns (u, kl)
    u = model(x_col, t_col)      # u is (N,1), kl is scalar
    # u is actually a tuple; pde_residual calls model(x, t) expecting a tensor.
    # We need to extract u and kl before computing residual.
    # Solution: separate the two concerns with a thin lambda wrapper.
    u_val, kl = u

    # Compute PDE residual using autograd on u_val
    u_x = torch.autograd.grad(
        u_val, x_col,
        grad_outputs=torch.ones_like(u_val),
        create_graph=True, retain_graph=True,
    )[0]
    u_t = torch.autograd.grad(
        u_val, t_col,
        grad_outputs=torch.ones_like(u_val),
        create_graph=True, retain_graph=True,
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x_col,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True,
    )[0]

    f     = u_t + u_val * u_x - NU * u_xx
    l_pde = torch.mean(f ** 2)
    return l_pde, kl


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

def train_bayesian(
    n_col: int = 10_000,
    n_ic:  int = 200,
    n_bc:  int = 200,
    n_hidden:  int = 4,
    n_neurons: int = 50,
    n_epochs:  int = 5_000,
    lr:        float = 1e-3,
    # kl_weight = None  →  automatically set to 1/n_col (standard scaling)
    kl_weight: float | None = None,
    n_mc_samples: int = 1,           # MC samples per epoch for data-fit terms
    lambda_pde:  float = 1.0,
    lambda_ic:   float = 10.0,
    lambda_bc:   float = 10.0,
    print_every: int = 500,
    save_path:   str = "outputs/bayesian/bayesian_pinn.pt",
    device_str:  str = "auto",
) -> tuple[BayesianBurgersPINN, list[float], list[float], list[float], list[float], float]:
    """
    Train the Bayesian PINN.

    Returns
    -------
    model       : trained BayesianBurgersPINN
    history_pde : PDE loss per logged checkpoint
    history_ic  : IC  loss
    history_bc  : BC  loss
    history_kl  : KL  loss
    train_time  : total wall-clock seconds
    """
    # --- Device ---
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[bayes_train] Device: {device}")

    # --- Model ---
    model = BayesianBurgersPINN(n_hidden=n_hidden, n_neurons=n_neurons).to(device)

    # --- KL weight with linear warm-up annealing ---
    # Mean-field VI on PINNs is sensitive to the KL coefficient because
    # the raw KL (sum over ~7800 weights) is O(10^4) while the data-fit
    # MSE terms are O(1).  We use two strategies together:
    #
    # 1. A very small base kl_weight (auto: 1e-7) so the KL contribution
    #    is ~0.001 at most, keeping the loss dominated by data fit.
    #
    # 2. Linear annealing: kl_weight ramps from 0 to its final value over
    #    the first kl_anneal_epochs epochs.  This lets the network first
    #    converge as a MAP estimate before the prior pulls the weights back.
    if kl_weight is None:
        kl_weight = 1e-7
    kl_anneal_epochs = n_epochs // 2   # ramp over the first half of training
    print(f"[bayes_train] kl_weight (final) = {kl_weight:.2e}, "
          f"anneal over first {kl_anneal_epochs} epochs")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2000
    )

    # --- Fixed boundary / initial-condition data ---
    x_ic, t_ic, u_ic = sample_initial_condition_points(n_ic, device)
    x_bc, t_bc, u_bc = sample_boundary_condition_points(n_bc, device)

    history_pde: list[float] = []
    history_ic:  list[float] = []
    history_bc:  list[float] = []
    history_kl:  list[float] = []

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bayes_train] Model has {n_params:,} variational parameters "
          f"(2x a standard PINN of same size)")
    print(f"[bayes_train] Starting training: {n_epochs} epochs")
    print("-" * 65)

    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        # Fresh collocation points each epoch
        x_col, t_col = sample_collocation_points(n_col, device)

        # --- Average data-fit losses over n_mc_samples weight samples ---
        l_pde_acc = torch.zeros(1, device=device)
        l_ic_acc  = torch.zeros(1, device=device)
        l_bc_acc  = torch.zeros(1, device=device)
        kl_acc    = torch.zeros(1, device=device)

        for _ in range(n_mc_samples):
            # PDE loss + KL from one weight sample
            lp, kl = bayes_loss_pde(model, x_col, t_col)
            l_pde_acc = l_pde_acc + lp
            kl_acc    = kl_acc + kl

            # IC loss (draw another weight sample for independence)
            u_pred_ic, _ = model(x_ic, t_ic)
            li = torch.mean((u_pred_ic - u_ic) ** 2)
            l_ic_acc = l_ic_acc + li

            # BC loss
            u_pred_bc, _ = model(x_bc, t_bc)
            lb = torch.mean((u_pred_bc - u_bc) ** 2)
            l_bc_acc = l_bc_acc + lb

        l_pde = l_pde_acc / n_mc_samples
        l_ic  = l_ic_acc  / n_mc_samples
        l_bc  = l_bc_acc  / n_mc_samples
        kl    = kl_acc    / n_mc_samples

        # Linear KL annealing: weight ramps 0 → kl_weight over first half
        anneal_factor = min(1.0, epoch / max(kl_anneal_epochs, 1))
        kl_w_effective = kl_weight * anneal_factor

        total_loss = (lambda_pde * l_pde
                      + lambda_ic  * l_ic
                      + lambda_bc  * l_bc
                      + kl_w_effective * kl)

        total_loss.backward()
        optimiser.step()
        scheduler.step(total_loss.detach())

        if epoch % print_every == 0 or epoch == 1:
            pde_val = l_pde.item()
            ic_val  = l_ic.item()
            bc_val  = l_bc.item()
            kl_val  = kl.item()
            tot_val = total_loss.item()
            cur_lr  = optimiser.param_groups[0]["lr"]

            history_pde.append(pde_val)
            history_ic.append(ic_val)
            history_bc.append(bc_val)
            history_kl.append(kl_val)

            print(
                f"Epoch {epoch:>6d} | "
                f"Total: {tot_val:.4e} | "
                f"PDE: {pde_val:.4e} | "
                f"IC: {ic_val:.4e} | "
                f"BC: {bc_val:.4e} | "
                f"KL: {kl_val:.4e} | "
                f"lr: {cur_lr:.2e}"
            )

    train_time = time.time() - t0
    print("-" * 65)
    print(f"[bayes_train] Done in {train_time/60:.1f} min.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[bayes_train] Checkpoint saved to '{save_path}'")

    return model, history_pde, history_ic, history_bc, history_kl, train_time
