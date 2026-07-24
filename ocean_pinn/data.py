"""
data.py  —  ocean_pinn
-----------------------
Point generators for the 1D advection-diffusion PINN:

  Domain  : x in [0, 10] km,   t in [0, 5]
  IC      : c(x, 0) = exp( -(x-2)^2 / 0.5 )   — Gaussian pulse at x=2
  BC left : c(0, t) = 0                          — Dirichlet zero
  BC right: c(10, t) = 0                         — Dirichlet zero

Three point sets are needed:
  1. Collocation points — PDE residual enforced here (re-sampled every epoch)
  2. IC points          — enforce the Gaussian initial condition
  3. BC points          — enforce zero concentration at both walls

All constants (domain bounds, IC parameters, physical parameters) are defined
here and imported by train.py and plot.py.
"""

import numpy as np
import torch

# ------------------------------------------------------------------ #
#  Domain and physical constants                                       #
# ------------------------------------------------------------------ #
X_MIN, X_MAX =  0.0, 10.0   # spatial domain [km]
T_MIN, T_MAX =  0.0,  5.0   # temporal domain

# Advection-diffusion parameters
V = 1.0     # advection velocity
D = 0.05    # diffusion coefficient

# Gaussian IC:  c(x,0) = exp(-(x - X0)^2 / WIDTH)
IC_CENTER = 2.0
IC_WIDTH  = 0.5   # controls the spread; smaller = narrower pulse


def _ic(x: np.ndarray) -> np.ndarray:
    """Evaluate the Gaussian initial condition on a numpy array of x values."""
    return np.exp(-((x - IC_CENTER) ** 2) / IC_WIDTH).astype(np.float32)


# ------------------------------------------------------------------ #
#  Collocation points (interior, for PDE residual)                    #
# ------------------------------------------------------------------ #

def sample_collocation_points(
    n_points: int = 10_000,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (x, t) pairs uniformly at random inside the domain.
    requires_grad=True is set so autograd can compute dc/dx and dc/dt.
    """
    x_np = np.random.uniform(X_MIN, X_MAX, (n_points, 1)).astype(np.float32)
    t_np = np.random.uniform(T_MIN, T_MAX, (n_points, 1)).astype(np.float32)
    x_col = torch.tensor(x_np, device=device, requires_grad=True)
    t_col = torch.tensor(t_np, device=device, requires_grad=True)
    return x_col, t_col


# ------------------------------------------------------------------ #
#  Initial condition points                                            #
# ------------------------------------------------------------------ #

def sample_initial_condition_points(
    n_points: int = 200,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns x_ic, t_ic (all zeros), c_ic (Gaussian values).
    """
    x_np = np.linspace(X_MIN, X_MAX, n_points, dtype=np.float32).reshape(-1, 1)
    t_np = np.zeros((n_points, 1), dtype=np.float32)
    c_np = _ic(x_np)

    return (
        torch.tensor(x_np, device=device),
        torch.tensor(t_np, device=device),
        torch.tensor(c_np, device=device),
    )


# ------------------------------------------------------------------ #
#  Boundary condition points                                           #
# ------------------------------------------------------------------ #

def sample_boundary_condition_points(
    n_points: int = 200,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dirichlet zero BCs at x=0 and x=10.
    Returns x_bc (both walls stacked), t_bc, c_bc (all zeros).
    """
    t_np = np.linspace(T_MIN, T_MAX, n_points, dtype=np.float32).reshape(-1, 1)

    x_left  = np.full_like(t_np, X_MIN)
    x_right = np.full_like(t_np, X_MAX)

    x_bc_np = np.vstack([x_left, x_right])
    t_bc_np = np.vstack([t_np, t_np])
    c_bc_np = np.zeros((2 * n_points, 1), dtype=np.float32)

    return (
        torch.tensor(x_bc_np, device=device),
        torch.tensor(t_bc_np, device=device),
        torch.tensor(c_bc_np, device=device),
    )


# ------------------------------------------------------------------ #
#  Dense evaluation grid (for plotting / calibration)                 #
# ------------------------------------------------------------------ #

def make_evaluation_grid(
    n_x: int = 256,
    n_t: int = 100,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Build a dense (x, t) meshgrid for evaluating the trained solution.

    Returns
    -------
    x_flat : (n_x*n_t, 1) tensor
    t_flat : (n_x*n_t, 1) tensor
    x_grid : (n_t, n_x) numpy  (for contourf)
    t_grid : (n_t, n_x) numpy
    """
    x_np = np.linspace(X_MIN, X_MAX, n_x, dtype=np.float32)
    t_np = np.linspace(T_MIN, T_MAX, n_t, dtype=np.float32)
    x_grid, t_grid = np.meshgrid(x_np, t_np)   # (n_t, n_x)

    x_flat = torch.tensor(x_grid.reshape(-1, 1), device=device)
    t_flat = torch.tensor(t_grid.reshape(-1, 1), device=device)
    return x_flat, t_flat, x_grid, t_grid
