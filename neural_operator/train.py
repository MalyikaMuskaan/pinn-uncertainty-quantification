"""
train.py
--------
Supervised training loop for the 1-D FNO on viscous Burgers' equation.

Loss
----
    L = MSE(FNO(u₀), u_ground_truth)

The FNO is trained purely from data — there is no PDE residual term.
This is the key distinction from the PINN: the FNO never sees the governing
equations; it learns the solution operator entirely from labelled examples.

Schedule
--------
  - Adam optimiser, cosine annealing LR schedule
  - Gradient clipping (max_norm=1.0) for stability
  - Best-validation-loss model checkpoint saved throughout

Outputs
-------
  outputs/fno_best.pt      — best checkpoint (state_dict + metadata)
  outputs/train_history.npz — loss curves for later plotting

Usage
-----
    python train.py                        # default settings
    python train.py --epochs 500 --lr 1e-3
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model    import FNO1d
from data_gen import load_dataset


# ------------------------------------------------------------------ #
#  Default hyperparameters                                             #
# ------------------------------------------------------------------ #

DEFAULTS = dict(
    epochs      = 300,
    batch_size  = 32,
    lr          = 1e-3,
    weight_decay= 1e-4,
    modes       = 16,
    width       = 64,
    depth       = 4,
    clip_norm   = 1.0,
    data_path   = "outputs/dataset.npz",
    out_dir     = "outputs",
    print_every = 10,
)


# ------------------------------------------------------------------ #
#  Training function                                                   #
# ------------------------------------------------------------------ #

def train(cfg: dict | None = None) -> dict:
    """
    Train the FNO and return a results dict:
        model          : best trained FNO1d
        train_losses   : list of per-epoch train MSE
        val_losses     : list of per-epoch val  MSE
        best_val_loss  : float
        train_time     : wall-clock seconds
        epochs_run     : int
    """
    c = {**DEFAULTS, **(cfg or {})}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  epochs={c['epochs']}  "
          f"batch={c['batch_size']}  lr={c['lr']}  "
          f"modes={c['modes']}  width={c['width']}  depth={c['depth']}")

    # ---- Data ----
    ds = load_dataset(c["data_path"])

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    u0_tr = to_tensor(ds["u0_train"])   # (800, N_x)
    u_tr  = to_tensor(ds["u_train"])    # (800, N_x, N_t)
    u0_va = to_tensor(ds["u0_val"])     # (100, N_x)
    u_va  = to_tensor(ds["u_val"])      # (100, N_x, N_t)

    train_loader = DataLoader(TensorDataset(u0_tr, u_tr),
                              batch_size=c["batch_size"], shuffle=True,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(TensorDataset(u0_va, u_va),
                              batch_size=c["batch_size"], shuffle=False,
                              pin_memory=(device.type == "cuda"))

    # ---- Model ----
    n_x = ds["x_grid"].shape[0]
    n_t = ds["t_grid"].shape[0]
    model = FNO1d(n_x=n_x, n_t=n_t,
                  modes=c["modes"], width=c["width"], depth=c["depth"]
                  ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] FNO1d parameters: {n_params:,}")

    # ---- Optimiser + scheduler ----
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=c["epochs"], eta_min=1e-6)

    criterion = nn.MSELoss()

    # ---- Training loop ----
    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val  = float("inf")
    best_ckpt = None

    os.makedirs(c["out_dir"], exist_ok=True)
    best_path = os.path.join(c["out_dir"], "fno_best.pt")

    t_start = time.time()

    for epoch in range(1, c["epochs"] + 1):

        # -- Train --
        model.train()
        train_mse = 0.0
        for u0_b, u_b in train_loader:
            u0_b = u0_b.to(device)
            u_b  = u_b.to(device)

            optimiser.zero_grad()
            pred = model(u0_b)          # (batch, N_x, N_t)
            loss = criterion(pred, u_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c["clip_norm"])
            optimiser.step()

            train_mse += loss.item() * len(u0_b)

        train_mse /= len(train_loader.dataset)
        scheduler.step()

        # -- Validate --
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for u0_b, u_b in val_loader:
                u0_b = u0_b.to(device)
                u_b  = u_b.to(device)
                pred = model(u0_b)
                val_mse += criterion(pred, u_b).item() * len(u0_b)
        val_mse /= len(val_loader.dataset)

        train_losses.append(train_mse)
        val_losses.append(val_mse)

        # -- Checkpoint best --
        if val_mse < best_val:
            best_val  = val_mse
            best_ckpt = {
                "state_dict":  model.state_dict(),
                "epoch":       epoch,
                "val_loss":    val_mse,
                "n_x":         n_x,
                "n_t":         n_t,
                "modes":       c["modes"],
                "width":       c["width"],
                "depth":       c["depth"],
            }
            torch.save(best_ckpt, best_path)

        if epoch % c["print_every"] == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"  epoch {epoch:>4d}/{c['epochs']}  "
                  f"train_mse={train_mse:.4e}  val_mse={val_mse:.4e}  "
                  f"best_val={best_val:.4e}  lr={lr_now:.2e}")

    train_time = time.time() - t_start
    print(f"[train] Finished in {train_time:.1f}s  "
          f"best_val_mse={best_val:.4e}  checkpoint='{best_path}'")

    # Save loss curves
    hist_path = os.path.join(c["out_dir"], "train_history.npz")
    np.savez(hist_path,
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses))
    print(f"[train] Loss history saved to '{hist_path}'")

    # Reload best weights into model before returning
    model.load_state_dict(best_ckpt["state_dict"])

    return dict(
        model         = model,
        train_losses  = train_losses,
        val_losses    = val_losses,
        best_val_loss = best_val,
        train_time    = train_time,
        epochs_run    = c["epochs"],
    )


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FNO on Burgers dataset")
    for key, val in DEFAULTS.items():
        t = type(val) if val is not None else str
        parser.add_argument(f"--{key}", type=t, default=val)
    args = parser.parse_args()
    train(vars(args))
