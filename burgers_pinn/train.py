"""
train.py
--------
Training loop for the Burgers' PINN.

The total loss is the sum of three terms:

  L_total = λ_pde * L_pde  +  λ_ic * L_ic  +  λ_bc * L_bc

  L_pde  : PDE residual loss — how well the network satisfies
            ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x² = 0
            at randomly sampled collocation points inside the domain.

  L_ic   : Initial condition loss — MSE between network prediction and
            u(x, 0) = -sin(π x) along t = 0.

  L_bc   : Boundary condition loss — MSE between network prediction and
            u(±1, t) = 0 along the spatial walls.

Key concept — automatic differentiation for PDE derivatives:
  PyTorch's autograd engine computes exact derivatives of the network
  output with respect to its inputs.  We simply call torch.autograd.grad
  twice on x to get the second spatial derivative needed for the viscous term.
"""

import torch
import torch.nn as nn
from model import BurgersPINN
from data import (
    sample_collocation_points,
    sample_initial_condition_points,
    sample_boundary_condition_points,
)

# ------------------------------------------------------------------ #
#  Physical constant                                                   #
# ------------------------------------------------------------------ #
NU = 0.01 / 3.141592653589793   # kinematic viscosity  ν = 0.01 / π


# ------------------------------------------------------------------ #
#  PDE residual computation                                            #
# ------------------------------------------------------------------ #

def pde_residual(
    model: BurgersPINN,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Burgers' equation residual at collocation points.

    Residual:  f = ∂u/∂t  +  u * ∂u/∂x  -  ν * ∂²u/∂x²

    A perfect solution gives f = 0 everywhere in the domain.

    Parameters
    ----------
    model : the PINN whose residual we evaluate
    x     : (N, 1) collocation x-coordinates  (requires_grad=True)
    t     : (N, 1) collocation t-coordinates  (requires_grad=True)

    Returns
    -------
    f : (N, 1) residual tensor
    """
    u = model(x, t)   # forward pass → (N, 1)

    # --- First-order derivatives ---
    # create_graph=True is essential: it tells autograd to build a graph
    # *through* this grad computation so we can differentiate again for u_xx.
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]   # ∂u/∂x  — shape (N, 1)

    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]   # ∂u/∂t  — shape (N, 1)

    # --- Second-order spatial derivative ---
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]   # ∂²u/∂x²  — shape (N, 1)

    # Burgers' equation: residual = u_t + u*u_x - ν*u_xx
    f = u_t + u * u_x - NU * u_xx

    return f


# ------------------------------------------------------------------ #
#  Individual loss terms                                               #
# ------------------------------------------------------------------ #

def loss_pde(
    model: BurgersPINN,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared PDE residual over all collocation points."""
    f = pde_residual(model, x_col, t_col)
    return torch.mean(f ** 2)


def loss_initial_condition(
    model: BurgersPINN,
    x_ic: torch.Tensor,
    t_ic: torch.Tensor,
    u_ic: torch.Tensor,
) -> torch.Tensor:
    """MSE between network output and the initial condition u(x,0)=-sin(πx)."""
    u_pred = model(x_ic, t_ic)
    return torch.mean((u_pred - u_ic) ** 2)


def loss_boundary_condition(
    model: BurgersPINN,
    x_bc: torch.Tensor,
    t_bc: torch.Tensor,
    u_bc: torch.Tensor,
) -> torch.Tensor:
    """MSE between network output and the zero Dirichlet boundary values."""
    u_pred = model(x_bc, t_bc)
    return torch.mean((u_pred - u_bc) ** 2)


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

def train(
    n_col: int = 10_000,      # collocation points for PDE residual
    n_ic:  int = 200,         # initial condition points
    n_bc:  int = 200,         # boundary condition points (per wall)
    n_hidden: int = 4,        # hidden layers in the network
    n_neurons: int = 50,      # neurons per hidden layer
    n_epochs: int = 15_000,   # training epochs
    lr: float = 1e-3,         # Adam learning rate
    print_every: int = 500,   # console logging frequency
    save_path: str = "outputs/burgers_pinn.pt",   # model checkpoint path
    device_str: str = "auto", # "auto" | "cpu" | "cuda"
    # Loss weights — IC and BC are typically weighted more heavily so the
    # network first learns to match boundary data before refining the interior.
    lambda_pde: float = 1.0,
    lambda_ic:  float = 10.0,
    lambda_bc:  float = 10.0,
) -> tuple[BurgersPINN, list[float], list[float], list[float]]:
    """
    Main training function.

    Returns
    -------
    model          : trained BurgersPINN
    history_pde    : list of PDE loss values recorded every print_every epochs
    history_ic     : list of IC  loss values
    history_bc     : list of BC  loss values
    """

    # -------------------------------------------------------------- #
    #  Device selection                                                #
    # -------------------------------------------------------------- #
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[train] Using device: {device}")

    # -------------------------------------------------------------- #
    #  Build model and optimiser                                       #
    # -------------------------------------------------------------- #
    model = BurgersPINN(n_hidden=n_hidden, n_neurons=n_neurons).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning-rate scheduler: reduce lr by half if the total loss
    # stagnates for 2000 epochs.  Helps convergence in the later stages.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2000
    )

    # -------------------------------------------------------------- #
    #  Generate training data                                          #
    # -------------------------------------------------------------- #
    # Collocation points are re-sampled each epoch to prevent the network
    # from overfitting to a fixed set of interior points.
    # IC and BC points are fixed (they represent exact conditions).
    x_ic, t_ic, u_ic = sample_initial_condition_points(n_ic, device)
    x_bc, t_bc, u_bc = sample_boundary_condition_points(n_bc, device)

    # -------------------------------------------------------------- #
    #  Training loop                                                   #
    # -------------------------------------------------------------- #
    history_pde: list[float] = []
    history_ic:  list[float] = []
    history_bc:  list[float] = []

    print(f"[train] Starting training: {n_epochs} epochs, "
          f"{n_col} collocation points per epoch")
    print(f"[train] Loss weights — PDE: {lambda_pde}, IC: {lambda_ic}, BC: {lambda_bc}")
    print("-" * 65)

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        # --- Fresh collocation points each epoch ---
        x_col, t_col = sample_collocation_points(n_col, device)

        # --- Compute individual loss terms ---
        l_pde = loss_pde(model, x_col, t_col)
        l_ic  = loss_initial_condition(model, x_ic, t_ic, u_ic)
        l_bc  = loss_boundary_condition(model, x_bc, t_bc, u_bc)

        # --- Weighted total loss ---
        total_loss = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc

        # --- Backpropagation ---
        total_loss.backward()
        optimiser.step()
        scheduler.step(total_loss.detach())

        # --- Logging ---
        if epoch % print_every == 0 or epoch == 1:
            pde_val = l_pde.item()
            ic_val  = l_ic.item()
            bc_val  = l_bc.item()
            tot_val = total_loss.item()
            current_lr = optimiser.param_groups[0]["lr"]

            history_pde.append(pde_val)
            history_ic.append(ic_val)
            history_bc.append(bc_val)

            print(
                f"Epoch {epoch:>6d} | "
                f"Total: {tot_val:.4e} | "
                f"PDE: {pde_val:.4e} | "
                f"IC: {ic_val:.4e} | "
                f"BC: {bc_val:.4e} | "
                f"lr: {current_lr:.2e}"
            )

    print("-" * 65)
    print(f"[train] Training complete.")

    # -------------------------------------------------------------- #
    #  Save model weights                                              #
    # -------------------------------------------------------------- #
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[train] Model weights saved to '{save_path}'")

    return model, history_pde, history_ic, history_bc
