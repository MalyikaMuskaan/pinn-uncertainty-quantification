"""
dropout_train.py
----------------
Training loop for the MC Dropout PINN on the Burgers' equation.

Identical in structure to train.py — same PDE residual, IC loss, BC loss,
collocation re-sampling, Adam + ReduceLROnPlateau.

Key difference: dropout is ACTIVE during training (model stays in train()
mode, which is the default), so the network learns to be robust to the
stochastic node-dropping.  This is identical to standard dropout training.

Loss
----
  L = lambda_pde * L_pde  +  lambda_ic * L_ic  +  lambda_bc * L_bc

There is no extra KL term (unlike the Bayesian PINN).
"""

import os
import time
import torch
from dropout_model import DropoutBurgersPINN
from train import loss_pde, loss_initial_condition, loss_boundary_condition
from data import (
    sample_collocation_points,
    sample_initial_condition_points,
    sample_boundary_condition_points,
)


def train_dropout(
    n_col: int = 10_000,
    n_ic:  int = 200,
    n_bc:  int = 200,
    n_hidden:     int   = 4,
    n_neurons:    int   = 50,
    dropout_rate: float = 0.05,
    n_epochs: int   = 10_000,
    lr:       float = 1e-3,
    lambda_pde: float = 1.0,
    lambda_ic:  float = 10.0,
    lambda_bc:  float = 10.0,
    print_every: int = 500,
    save_path:   str = "outputs/dropout/dropout_pinn.pt",
    device_str:  str = "auto",
) -> tuple[DropoutBurgersPINN, list[float], list[float], list[float], float]:
    """
    Train the MC Dropout PINN.

    Parameters
    ----------
    dropout_rate : Dropout probability p on each hidden layer
                   (0.05 recommended; higher risks accuracy / gradient issues)

    Returns
    -------
    model      : trained DropoutBurgersPINN
    history_pde, history_ic, history_bc : loss histories
    train_time : wall-clock seconds
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[dropout_train] Device: {device}")
    print(f"[dropout_train] Dropout rate: {dropout_rate}")

    model = DropoutBurgersPINN(
        n_hidden=n_hidden, n_neurons=n_neurons, dropout_rate=dropout_rate
    ).to(device)
    # model.train() is the default; dropout is active throughout training

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2000
    )

    x_ic, t_ic, u_ic = sample_initial_condition_points(n_ic, device)
    x_bc, t_bc, u_bc = sample_boundary_condition_points(n_bc, device)

    history_pde: list[float] = []
    history_ic:  list[float] = []
    history_bc:  list[float] = []

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dropout_train] {n_params:,} parameters  "
          f"({n_hidden} hidden layers x {n_neurons} neurons, p={dropout_rate})")
    print(f"[dropout_train] Starting: {n_epochs} epochs, "
          f"{n_col} collocation pts/epoch")
    print("-" * 65)

    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # model stays in train() mode — dropout active
        optimiser.zero_grad()

        x_col, t_col = sample_collocation_points(n_col, device)

        # Reuse existing loss functions from train.py — they call model(x, t)
        # which triggers dropout in the forward pass
        lp = loss_pde(model, x_col, t_col)
        li = loss_initial_condition(model, x_ic, t_ic, u_ic)
        lb = loss_boundary_condition(model, x_bc, t_bc, u_bc)

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

    train_time = time.time() - t0
    print("-" * 65)
    print(f"[dropout_train] Done in {train_time/60:.1f} min.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[dropout_train] Checkpoint saved to '{save_path}'")

    return model, history_pde, history_ic, history_bc, train_time
