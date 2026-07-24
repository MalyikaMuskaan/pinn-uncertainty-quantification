"""
data.py
-------
Data utilities for the 2-D Darcy flow PINN.

PDE setup
---------
Equation:   -div( k(x,y) * grad(u(x,y)) ) = f(x,y)   on Ω = [0,1]²
BC:         u = 0  on ∂Ω (all four edges)

Manufactured solution (method of manufactured solutions)
---------------------------------------------------------
We choose a known exact solution and derive f by substitution:

    u*(x, y) = sin(πx) · sin(πy)                         ... (1)

    k(x, y)  = 1 + 0.5 · sin(πx) · sin(πy)              ... (2)

Derivatives of u*:
    u_x  =  π cos(πx) sin(πy)
    u_y  =  π sin(πx) cos(πy)
    u_xx = -π² sin(πx) sin(πy)
    u_yy = -π² sin(πx) sin(πy)

Derivatives of k:
    k_x  = 0.5π cos(πx) sin(πy)
    k_y  = 0.5π sin(πx) cos(πy)

Expanding -div(k ∇u):
    ∂/∂x[k u_x] = k_x · u_x + k · u_xx
                = 0.5π²cos²(πx)sin²(πy) - π²k·sin(πx)sin(πy)

    ∂/∂y[k u_y] = k_y · u_y + k · u_yy      (symmetric: x↔y)
                = 0.5π²sin²(πx)cos²(πy) - π²k·sin(πx)sin(πy)

    -div(k∇u*) = -(∂/∂x[k u_x] + ∂/∂y[k u_y])
               = 2π²·k·sin(πx)sin(πy)
                 - 0.5π²·cos²(πx)sin²(πy)
                 - 0.5π²·sin²(πx)cos²(πy)

Substituting k = 1 + 0.5·sin(πx)sin(πy):

    f(x,y) = 2π²·(1 + 0.5·sin(πx)sin(πy))·sin(πx)sin(πy)
             - 0.5π²·cos²(πx)·sin²(πy)
             - 0.5π²·sin²(πx)·cos²(πy)               ... (3)

Note: u*(x,y) = sin(πx)sin(πy) satisfies u=0 on all four edges of [0,1]²
because sin(0)=sin(π)=0, so BCs are automatically satisfied by the MMS choice.
"""

import math
import numpy as np
import torch

# ------------------------------------------------------------------ #
#  Domain constants                                                    #
# ------------------------------------------------------------------ #

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0

PI = math.pi


# ------------------------------------------------------------------ #
#  Permeability field and source term (PyTorch, for PINN loss)         #
# ------------------------------------------------------------------ #

def permeability(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    k(x, y) = 1 + 0.5 · sin(πx) · sin(πy)

    Parameters
    ----------
    x, y : (N, 1) tensors — coordinates in [0, 1]²

    Returns
    -------
    k : (N, 1) tensor — permeability at each point
    """
    return 1.0 + 0.5 * torch.sin(PI * x) * torch.sin(PI * y)


def source_term(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    f(x, y) derived from MMS (see module docstring, equation 3):

        f = 2π²·k·sin(πx)sin(πy)
            - 0.5π²·cos²(πx)·sin²(πy)
            - 0.5π²·sin²(πx)·cos²(πy)

    where k = 1 + 0.5·sin(πx)sin(πy).

    Parameters
    ----------
    x, y : (N, 1) tensors — coordinates in [0, 1]²

    Returns
    -------
    f : (N, 1) tensor — source term at each point (no gradient needed)
    """
    sin_px = torch.sin(PI * x)
    sin_py = torch.sin(PI * y)
    cos_px = torch.cos(PI * x)
    cos_py = torch.cos(PI * y)

    k   = 1.0 + 0.5 * sin_px * sin_py
    pi2 = PI ** 2

    term1 = 2.0 * pi2 * k * sin_px * sin_py
    term2 = 0.5 * pi2 * cos_px ** 2 * sin_py ** 2
    term3 = 0.5 * pi2 * sin_px ** 2 * cos_py ** 2

    return term1 - term2 - term3


# ------------------------------------------------------------------ #
#  Exact solution (NumPy, for plotting/evaluation)                     #
# ------------------------------------------------------------------ #

def exact_solution_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    u*(x, y) = sin(πx) · sin(πy)  — used as ground truth for evaluation.

    Accepts either 1-D arrays (broadcast via meshgrid) or pre-meshed arrays
    of the same shape.
    """
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def exact_solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """PyTorch version of the exact solution — used for MSE evaluation."""
    return torch.sin(PI * x) * torch.sin(PI * y)


# ------------------------------------------------------------------ #
#  Collocation point samplers                                          #
# ------------------------------------------------------------------ #

def sample_collocation_points(
    n_points: int = 10_000,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random interior collocation points from (0,1)² (open domain).
    requires_grad=True so that autograd can compute ∂u/∂x and ∂u/∂y.

    Parameters
    ----------
    n_points : number of collocation points
    device   : torch device

    Returns
    -------
    x, y : each (n_points, 1), requires_grad=True
    """
    x_np = np.random.uniform(X_MIN, X_MAX, (n_points, 1)).astype(np.float32)
    y_np = np.random.uniform(Y_MIN, Y_MAX, (n_points, 1)).astype(np.float32)
    x = torch.tensor(x_np, device=device, requires_grad=True)
    y = torch.tensor(y_np, device=device, requires_grad=True)
    return x, y


def sample_boundary_points(
    n_per_edge: int = 200,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample uniformly spaced points on all four edges of [0,1]².
    u = 0 on every boundary point (Dirichlet BC, satisfied by MMS choice).

    Edges sampled:
        bottom : y = 0,  x ∈ [0,1]
        top    : y = 1,  x ∈ [0,1]
        left   : x = 0,  y ∈ [0,1]
        right  : x = 1,  y ∈ [0,1]

    Parameters
    ----------
    n_per_edge : number of points per edge (total = 4 * n_per_edge)
    device     : torch device

    Returns
    -------
    x_bc, y_bc : each (4*n_per_edge, 1)
    u_bc       : (4*n_per_edge, 1) — all zeros
    """
    lin = np.linspace(X_MIN, X_MAX, n_per_edge, dtype=np.float32)

    # bottom: y=0
    x_bot = lin.reshape(-1, 1)
    y_bot = np.zeros_like(x_bot)

    # top: y=1
    x_top = lin.reshape(-1, 1)
    y_top = np.ones_like(x_top)

    # left: x=0
    y_lft = lin.reshape(-1, 1)
    x_lft = np.zeros_like(y_lft)

    # right: x=1
    y_rgt = lin.reshape(-1, 1)
    x_rgt = np.ones_like(y_rgt)

    x_bc = np.vstack([x_bot, x_top, x_lft, x_rgt])
    y_bc = np.vstack([y_bot, y_top, y_lft, y_rgt])
    u_bc = np.zeros((4 * n_per_edge, 1), dtype=np.float32)

    return (
        torch.tensor(x_bc, device=device),
        torch.tensor(y_bc, device=device),
        torch.tensor(u_bc, device=device),
    )


# ------------------------------------------------------------------ #
#  Evaluation grid                                                     #
# ------------------------------------------------------------------ #

def make_eval_grid(
    n: int = 256,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Create a regular n×n grid over [0,1]² for evaluation and plotting.

    Returns
    -------
    x_flat, y_flat : each (n², 1) — flattened grid tensors (no requires_grad)
    X, Y           : each (n, n)  — 2-D meshgrid arrays (NumPy, for plotting)
    """
    lin = np.linspace(X_MIN, X_MAX, n, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin)   # shape (n, n) — Y axis first (row index)

    x_flat = torch.tensor(X.reshape(-1, 1), device=device)
    y_flat = torch.tensor(Y.reshape(-1, 1), device=device)
    return x_flat, y_flat, X, Y
