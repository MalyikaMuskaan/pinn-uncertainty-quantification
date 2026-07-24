"""
train.py  —  ocean_pinn
-----------------------
Training loop for the advection-diffusion PINN.

Total loss:
    L = lambda_pde * L_pde  +  lambda_ic * L_ic  +  lambda_bc * L_bc

  L_pde : mean-squared residual of dc/dt + v*dc/dx - D*d²c/dx² = 0
  L_ic  : MSE between prediction and the Gaussian IC at t=0
  L_bc  : MSE between prediction and zero at x=0 and x=10

Derivatives are computed with torch.autograd.grad (exact, no finite differences).
The training loop is identical in structure to burgers_pinn/train.py.
"""

import os
import torch
from model import OceanPINN
from data import (
    V, D,
    sample_collocation_points,
    sample_initial_condition_points,
    sample_boundary_condition_points,
)


# ------------------------------------------------------------------ #
#  PDE residual                                                        #
# ------------------------------------------------------------------ #

def pde_residual(
    model: OceanPINN,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the advection-diffusion residual at collocation points.

    Residual:  f = dc/dt + v*(dc/dx) - D*(d²c/dx²)

    A perfect solution gives f = 0 everywhere in the domain.

    Parameters
    ----------
    model : the OceanPINN to evaluate
    x     : (N, 1) with requires_grad=True
    t     : (N, 1) with requires_grad=True

    Returns
    -------
    f : (N, 1) residual
    """
    c = model(x, t)

    c_x = torch.autograd.grad(
        c, x,
        grad_outputs=torch.ones_like(c),
        create_graph=True, retain_graph=True,
    )[0]

    c_t = torch.autograd.grad(
        c, t,
        grad_outputs=torch.ones_like(c),
        create_graph=True, retain_graph=True,
    )[0]

    c_xx = torch.autograd.grad(
        c_x, x,
        grad_outputs=torch.ones_like(c_x),
        create_graph=True, retain_graph=True,
    )[0]

    return c_t + V * c_x - D * c_xx


# ------------------------------------------------------------------ #
#  Loss terms                                                          #
# ------------------------------------------------------------------ #

def loss_pde(model, x_col, t_col):
    f = pde_residual(model, x_col, t_col)
    return torch.mean(f ** 2)

def loss_ic(model, x_ic, t_ic, c_ic):
    return torch.mean((model(x_ic, t_ic) - c_ic) ** 2)

def loss_bc(model, x_bc, t_bc, c_bc):
    return torch.mean((model(x_bc, t_bc) - c_bc) ** 2)


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

def train(
    n_col: int = 10_000,
    n_ic:  int = 200,
    n_bc:  int = 200,
    n_hidden:  int = 4,
    n_neurons: int = 50,
    n_epochs:  int = 15_000,
    lr:        float = 1e-3,
    lambda_pde: float = 1.0,
    lambda_ic:  float = 10.0,
    lambda_bc:  float = 10.0,
    print_every: int = 500,
    save_path:   str = "outputs/ocean_pinn.pt",
    device_str:  str = "auto",
) -> tuple[OceanPINN, list[float], list[float], list[float]]:
    """
    Train the advection-diffusion PINN.

    Returns
    -------
    model          : trained OceanPINN
    history_pde    : PDE loss at each logged checkpoint
    history_ic     : IC  loss
    history_bc     : BC  loss
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[train] Device: {device}")

    model     = OceanPINN(n_hidden=n_hidden, n_neurons=n_neurons).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2000
    )

    # Fixed IC / BC data
    x_ic, t_ic, c_ic = sample_initial_condition_points(n_ic, device)
    x_bc, t_bc, c_bc = sample_boundary_condition_points(n_bc, device)

    history_pde: list[float] = []
    history_ic:  list[float] = []
    history_bc:  list[float] = []

    print(f"[train] Starting: {n_epochs} epochs, {n_col} collocation pts/epoch")
    print(f"[train] Loss weights  PDE={lambda_pde}  IC={lambda_ic}  BC={lambda_bc}")
    print(f"[train] PDE: dc/dt + {V}*dc/dx = {D}*d2c/dx2")
    print("-" * 65)

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        x_col, t_col = sample_collocation_points(n_col, device)

        lp = loss_pde(model, x_col, t_col)
        li = loss_ic(model, x_ic, t_ic, c_ic)
        lb = loss_bc(model, x_bc, t_bc, c_bc)

        total = lambda_pde * lp + lambda_ic * li + lambda_bc * lb
        total.backward()
        optimiser.step()
        scheduler.step(total.detach())

        if epoch % print_every == 0 or epoch == 1:
            history_pde.append(lp.item())
            history_ic.append(li.item())
            history_bc.append(lb.item())
            cur_lr = optimiser.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:>6d} | "
                f"Total: {total.item():.4e} | "
                f"PDE: {lp.item():.4e} | "
                f"IC: {li.item():.4e} | "
                f"BC: {lb.item():.4e} | "
                f"lr: {cur_lr:.2e}"
            )

    print("-" * 65)
    print("[train] Training complete.")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[train] Model saved to '{save_path}'")

    return model, history_pde, history_ic, history_bc
