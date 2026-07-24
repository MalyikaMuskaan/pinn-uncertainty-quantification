"""
data.py
-------
Generates the three sets of training points needed by a PINN:

  1. Collocation points  — randomly sampled inside the (x, t) domain.
     The PDE residual is enforced at these points.

  2. Initial condition (IC) points — the line t = 0, x ∈ [-1, 1].
     We enforce u(x, 0) = -sin(π x) here.

  3. Boundary condition (BC) points — the lines x = -1 and x = +1, t ∈ [0, 1].
     We enforce u(±1, t) = 0 here.

All tensors are returned on the requested device and with requires_grad=True
where automatic differentiation will be needed (collocation points).
"""

import torch
import numpy as np


# ------------------------------------------------------------------ #
#  Physical domain constants (used here and re-imported in train.py)  #
# ------------------------------------------------------------------ #
X_MIN, X_MAX = -1.0, 1.0   # spatial domain
T_MIN, T_MAX =  0.0, 1.0   # temporal domain


def sample_collocation_points(
    n_points: int = 10_000,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (x, t) pairs uniformly at random inside the domain.

    These are the "interior" points where the PDE must be satisfied.
    requires_grad=True is set so that PyTorch can compute ∂u/∂x and ∂u/∂t
    via autograd during the PDE residual loss calculation.

    Parameters
    ----------
    n_points : number of collocation points (>=10 000 recommended)
    device   : torch device

    Returns
    -------
    x_col : (n_points, 1) tensor with requires_grad=True
    t_col : (n_points, 1) tensor with requires_grad=True
    """
    # Latin-Hypercube-style uniform random sampling over [X_MIN, X_MAX] x [T_MIN, T_MAX]
    x_np = np.random.uniform(X_MIN, X_MAX, (n_points, 1)).astype(np.float32)
    t_np = np.random.uniform(T_MIN, T_MAX, (n_points, 1)).astype(np.float32)

    x_col = torch.tensor(x_np, device=device, requires_grad=True)
    t_col = torch.tensor(t_np, device=device, requires_grad=True)

    return x_col, t_col


def sample_initial_condition_points(
    n_points: int = 200,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample points on the initial boundary t = 0.

    Returns
    -------
    x_ic  : (n_points, 1) — x positions at t=0
    t_ic  : (n_points, 1) — all zeros
    u_ic  : (n_points, 1) — ground-truth u(x, 0) = -sin(π x)
    """
    x_np = np.linspace(X_MIN, X_MAX, n_points, dtype=np.float32).reshape(-1, 1)
    t_np = np.zeros((n_points, 1), dtype=np.float32)
    u_np = -np.sin(np.pi * x_np).astype(np.float32)

    x_ic = torch.tensor(x_np, device=device)
    t_ic = torch.tensor(t_np, device=device)
    u_ic = torch.tensor(u_np, device=device)

    return x_ic, t_ic, u_ic


def sample_boundary_condition_points(
    n_points: int = 200,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample points on the spatial boundaries x = -1 and x = +1.

    For Burgers' equation with zero Dirichlet BCs, u = 0 at both walls.

    Returns
    -------
    x_bc : (2*n_points, 1) — x = -1 stacked with x = +1
    t_bc : (2*n_points, 1) — uniformly sampled t values
    u_bc : (2*n_points, 1) — all zeros (Dirichlet condition)
    """
    t_np = np.linspace(T_MIN, T_MAX, n_points, dtype=np.float32).reshape(-1, 1)

    # Left wall: x = -1
    x_left  = np.full_like(t_np, X_MIN)
    # Right wall: x = +1
    x_right = np.full_like(t_np, X_MAX)

    x_bc_np = np.vstack([x_left, x_right])          # (2*n_points, 1)
    t_bc_np = np.vstack([t_np,   t_np])              # (2*n_points, 1)
    u_bc_np = np.zeros((2 * n_points, 1), dtype=np.float32)

    x_bc = torch.tensor(x_bc_np, device=device)
    t_bc = torch.tensor(t_bc_np, device=device)
    u_bc = torch.tensor(u_bc_np, device=device)

    return x_bc, t_bc, u_bc


def make_evaluation_grid(
    n_x: int = 256,
    n_t: int = 100,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Build a dense (x, t) grid for evaluating/plotting the trained solution.

    Returns
    -------
    x_flat : (n_x*n_t, 1) tensor  — flattened x coordinates
    t_flat : (n_x*n_t, 1) tensor  — flattened t coordinates
    x_grid : (n_t, n_x) numpy array — meshgrid X (for contourf)
    t_grid : (n_t, n_x) numpy array — meshgrid T (for contourf)
    """
    x_np = np.linspace(X_MIN, X_MAX, n_x, dtype=np.float32)
    t_np = np.linspace(T_MIN, T_MAX, n_t, dtype=np.float32)

    # meshgrid: rows = time, cols = space
    x_grid, t_grid = np.meshgrid(x_np, t_np)   # both (n_t, n_x)

    x_flat = torch.tensor(x_grid.reshape(-1, 1), device=device)
    t_flat = torch.tensor(t_grid.reshape(-1, 1), device=device)

    return x_flat, t_flat, x_grid, t_grid
