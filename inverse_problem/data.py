"""
data.py
-------
Data utilities for the Burgers inverse-problem PINN.

Two responsibilities:
  1. Generate the standard PINN training points (collocation, IC, BC).
  2. Simulate sparse, noisy sensor measurements by sampling the
     Crank-Nicolson FD reference solution.

The sensor data is the ONLY information the network has about the true
solution field — it never sees the full FD solution during training.
The viscosity ν is unknown and must be recovered from these sensors.
"""

import numpy as np
import torch

# ------------------------------------------------------------------ #
#  Domain constants                                                    #
# ------------------------------------------------------------------ #
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX =  0.0, 1.0

NU_TRUE = 0.01 / np.pi   # true viscosity — the value to recover


# ------------------------------------------------------------------ #
#  Crank-Nicolson reference solver (copied verbatim from burgers_pinn) #
# ------------------------------------------------------------------ #

def fd_reference(x: np.ndarray, t: np.ndarray, nu: float) -> np.ndarray:
    """
    High-accuracy Crank-Nicolson solution of viscous Burgers' equation.

    u_t + u*u_x = nu*u_xx,  u(x,0)=-sin(pi*x),  u(-1,t)=u(1,t)=0

    Parameters
    ----------
    x  : 1-D array of query x values
    t  : 1-D array of query t values
    nu : kinematic viscosity

    Returns
    -------
    u_ref : (len(t), len(x)) array
    """
    from scipy.linalg import solve_banded

    Nx = 512
    Nt = 2000
    dx = 2.0 / (Nx + 1)
    dt = 1.0 / Nt

    x_fd = np.linspace(-1.0 + dx, 1.0 - dx, Nx)
    u    = -np.sin(np.pi * x_fd)

    r = nu * dt / (2.0 * dx ** 2)
    ab      = np.zeros((3, Nx))
    ab[0, 1:]  = -r                     # upper
    ab[1, :]   =  1.0 + 2.0 * r        # main
    ab[2, :-1] = -r                     # lower

    t_fd = np.linspace(0.0, 1.0, Nt + 1)
    needed = {int(np.argmin(np.abs(t_fd - tv))) for tv in t}
    needed.add(0)
    u_store = {0: u.copy()}

    for step in range(1, Nt + 1):
        u_x          = np.zeros(Nx)
        u_x[1:-1]    = (u[2:] - u[:-2]) / (2.0 * dx)
        u_x[0]       = (u[1]  - 0.0)    / (2.0 * dx)
        u_x[-1]      = (0.0   - u[-2])  / (2.0 * dx)

        rhs          = np.zeros(Nx)
        rhs[1:-1]    = r*u[:-2] + (1-2*r)*u[1:-1] + r*u[2:] - dt*u[1:-1]*u_x[1:-1]
        rhs[0]       = (1-2*r)*u[0]  + r*u[1]    - dt*u[0] *u_x[0]
        rhs[-1]      = r*u[-2] + (1-2*r)*u[-1]               - dt*u[-1]*u_x[-1]

        u = solve_banded((1, 1), ab, rhs)
        if step in needed:
            u_store[step] = u.copy()

    u_ref = np.zeros((len(t), len(x)))
    for i, tv in enumerate(t):
        idx  = int(np.argmin(np.abs(t_fd - tv)))
        u_fd = u_store.get(idx, u_store[max(u_store.keys())])
        u_ref[i, :] = np.interp(x, x_fd, u_fd, left=0.0, right=0.0)

    return u_ref


# ------------------------------------------------------------------ #
#  Sensor simulation                                                   #
# ------------------------------------------------------------------ #

def make_sensor_data(
    n_sensors: int = 50,
    noise_frac: float = 0.01,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate sparse, noisy sensor measurements from the FD reference.

    Points are drawn uniformly at random from the interior of the domain
    (t > 0 so that initial-condition data is NOT included — the IC is
    enforced separately via a dedicated loss term).

    Parameters
    ----------
    n_sensors  : number of sensor locations
    noise_frac : noise standard deviation as a fraction of |u_ref| range
    seed       : random seed for reproducibility
    device     : torch device

    Returns
    -------
    x_s, t_s : (n_sensors, 1) tensors — sensor (x, t) coordinates
    u_s      : (n_sensors, 1) tensor  — noisy sensor readings
    """
    rng = np.random.default_rng(seed)

    x_np = rng.uniform(X_MIN, X_MAX, n_sensors).astype(np.float32)
    t_np = rng.uniform(0.05, T_MAX, n_sensors).astype(np.float32)   # t > 0

    # Evaluate the true FD solution at the sensor locations
    # We need one value per sensor: query fd_reference row-by-row via interp
    x_grid_fine = np.linspace(X_MIN, X_MAX, 512)
    t_unique, inv_idx = np.unique(np.round(t_np, 6), return_inverse=True)
    u_ref_rows = fd_reference(x_grid_fine, t_unique, NU_TRUE)   # (|t_unique|, 512)

    u_clean = np.zeros(n_sensors, dtype=np.float32)
    for i in range(n_sensors):
        row = u_ref_rows[inv_idx[i], :]
        u_clean[i] = float(np.interp(x_np[i], x_grid_fine, row))

    # Additive Gaussian noise proportional to the signal range
    signal_range = float(np.abs(u_clean).max()) + 1e-8
    noise_std    = noise_frac * signal_range
    noise        = rng.normal(0.0, noise_std, n_sensors).astype(np.float32)
    u_noisy      = u_clean + noise

    x_s = torch.tensor(x_np.reshape(-1, 1), device=device)
    t_s = torch.tensor(t_np.reshape(-1, 1), device=device)
    u_s = torch.tensor(u_noisy.reshape(-1, 1), device=device)

    return x_s, t_s, u_s


# ------------------------------------------------------------------ #
#  Standard PINN training-data generators                              #
# ------------------------------------------------------------------ #

def sample_collocation_points(
    n_points: int = 10_000,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform random collocation points; requires_grad=True for autograd."""
    x_np = np.random.uniform(X_MIN, X_MAX, (n_points, 1)).astype(np.float32)
    t_np = np.random.uniform(T_MIN, T_MAX, (n_points, 1)).astype(np.float32)
    return (torch.tensor(x_np, device=device, requires_grad=True),
            torch.tensor(t_np, device=device, requires_grad=True))


def sample_initial_condition_points(
    n_points: int = 200,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Points on t=0; u(x,0) = -sin(pi*x)."""
    x_np = np.linspace(X_MIN, X_MAX, n_points, dtype=np.float32).reshape(-1, 1)
    t_np = np.zeros((n_points, 1), dtype=np.float32)
    u_np = (-np.sin(np.pi * x_np)).astype(np.float32)
    return (torch.tensor(x_np, device=device),
            torch.tensor(t_np, device=device),
            torch.tensor(u_np, device=device))


def sample_boundary_condition_points(
    n_points: int = 200,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Points on x=-1 and x=+1; u=0 (Dirichlet)."""
    t_np    = np.linspace(T_MIN, T_MAX, n_points, dtype=np.float32).reshape(-1, 1)
    x_left  = np.full_like(t_np, X_MIN)
    x_right = np.full_like(t_np, X_MAX)
    x_bc    = np.vstack([x_left, x_right])
    t_bc    = np.vstack([t_np, t_np])
    u_bc    = np.zeros((2 * n_points, 1), dtype=np.float32)
    return (torch.tensor(x_bc, device=device),
            torch.tensor(t_bc, device=device),
            torch.tensor(u_bc, device=device))


def make_evaluation_grid(
    n_x: int = 256,
    n_t: int = 100,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Dense (x,t) grid for evaluation / plotting."""
    x_np = np.linspace(X_MIN, X_MAX, n_x, dtype=np.float32)
    t_np = np.linspace(T_MIN, T_MAX, n_t, dtype=np.float32)
    x_grid, t_grid = np.meshgrid(x_np, t_np)
    x_flat = torch.tensor(x_grid.reshape(-1, 1), device=device)
    t_flat = torch.tensor(t_grid.reshape(-1, 1), device=device)
    return x_flat, t_flat, x_grid, t_grid
