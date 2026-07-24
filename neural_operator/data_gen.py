"""
data_gen.py
-----------
Generate the FNO training dataset for the viscous Burgers' equation.

What is generated
-----------------
For each of N_SAMPLES initial conditions we:
  1. Draw a random initial condition u₀(x) as a superposition of sine waves
     with randomly sampled amplitudes and wave-numbers (see `random_ic` below).
     This gives a diverse distribution of ICs that is strictly richer than
     the fixed -sin(πx) used by the PINN.
  2. Solve the Crank-Nicolson finite-difference scheme forward in time to
     obtain u(x, t) on a regular (N_x, N_t) grid.
  3. Stack the result into tensors and save to disk.

Spatial grid  : N_x = 256 points on [-1, 1]
Temporal grid : N_t = 100 points on [ 0, 1]
ν             : 0.01/π  (same as the PINN project)

Output
------
Saved to  outputs/dataset.npz  with keys:
    u0      : (N_SAMPLES, N_x)      — initial conditions
    u       : (N_SAMPLES, N_x, N_t) — full solutions
    x_grid  : (N_x,)                — spatial grid
    t_grid  : (N_t,)                — temporal grid

Run once; subsequent imports just call `load_dataset()`.

Usage
-----
    python data_gen.py                    # generates with default settings
    python data_gen.py --n_samples 2000  # larger dataset
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import solve_banded


# ------------------------------------------------------------------ #
#  Physical constants (must match inverse_problem/data.py)            #
# ------------------------------------------------------------------ #
NU   = 0.01 / np.pi    # kinematic viscosity  ≈ 0.003183
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX =  0.0, 1.0

N_SAMPLES  = 1_000
N_X        = 256      # spatial resolution for FNO input/output
N_T        = 100      # temporal resolution for FNO output
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# remaining 0.10 → test


# ------------------------------------------------------------------ #
#  Random initial condition generator                                  #
# ------------------------------------------------------------------ #

def random_ic(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Random initial condition on x ∈ [-1, 1] satisfying u(±1) = 0.

    Form:  u₀(x) = Σ_k  aₖ · sin(kπx)   for k = 1, …, K

    Amplitudes aₖ are drawn i.i.d. from Uniform(-A, A).  The result is
    normalised so that max|u₀| = 1, keeping all ICs at the same amplitude
    scale as the canonical -sin(πx) and ensuring FD solver stability.
    K is drawn uniformly from {1, …, K_max}.

    Boundary conditions u₀(±1) = 0 are automatically satisfied since
    sin(kπ·(±1)) = 0 for all integer k.
    """
    K_max  = 5
    A      = 1.0
    K      = rng.integers(1, K_max + 1)              # number of modes
    amps   = rng.uniform(-A, A, K)
    k_vals = np.arange(1, K + 1, dtype=float)
    u0 = np.sum(
        amps[:, None] * np.sin(k_vals[:, None] * np.pi * x[None, :]),
        axis=0,
    )
    # Normalise to max amplitude 1 so all ICs share the same scale and
    # the Crank-Nicolson solver stays stable under the same dt budget.
    peak = np.abs(u0).max()
    if peak > 1e-8:
        u0 = u0 / peak
    return u0.astype(np.float32)


# ------------------------------------------------------------------ #
#  Crank-Nicolson solver with arbitrary initial condition              #
# ------------------------------------------------------------------ #

def fd_solve(u0_fine: np.ndarray, x_fine: np.ndarray,
             x_out: np.ndarray, t_out: np.ndarray,
             nu: float = NU) -> np.ndarray:
    """
    Integrate viscous Burgers' equation from u0_fine using Crank-Nicolson.

    Parameters
    ----------
    u0_fine : (N_fd,) initial condition on the internal FD grid
    x_fine  : (N_fd,) FD grid (interior points, no boundaries)
    x_out   : (N_x,)  output spatial grid
    t_out   : (N_t,)  output temporal grid (strictly increasing, t_out[0] ≥ 0)
    nu      : viscosity

    Returns
    -------
    u_out : (N_x, N_t)  solution interpolated onto (x_out, t_out)
    """
    Nx   = len(u0_fine)
    # Use enough time steps to satisfy the CFL-like stability condition for the
    # explicit nonlinear advection term:  dt < dx / max|u|.
    # With normalised ICs (max|u0|=1) and dx≈2/512, Nt=4000 gives dt≈2.5e-4,
    # safely below the threshold for smooth solutions.
    Nt   = 4000
    dx   = x_fine[1] - x_fine[0]
    dt   = T_MAX / Nt

    u    = u0_fine.copy().astype(np.float64)
    r    = nu * dt / (2.0 * dx ** 2)

    # Build the tridiagonal LHS matrix (banded storage for solve_banded)
    ab       = np.zeros((3, Nx))
    ab[0, 1:]  = -r          # superdiagonal
    ab[1, :]   =  1.0 + 2*r  # diagonal
    ab[2, :-1] = -r          # subdiagonal

    t_fd   = np.linspace(0.0, T_MAX, Nt + 1)
    # Which FD steps correspond to requested output times?
    needed = {int(np.argmin(np.abs(t_fd - tv))) for tv in t_out}
    needed.add(0)
    u_store = {0: u.copy()}

    for step in range(1, Nt + 1):
        # Central-difference advection term (explicit)
        u_x           = np.zeros(Nx)
        u_x[1:-1]     = (u[2:] - u[:-2]) / (2.0 * dx)
        u_x[0]        = (u[1]  - 0.0)    / (2.0 * dx)    # left ghost = 0 (BC)
        u_x[-1]       = (0.0   - u[-2])  / (2.0 * dx)    # right ghost = 0 (BC)

        rhs           = np.zeros(Nx)
        rhs[1:-1]     = r*u[:-2] + (1-2*r)*u[1:-1] + r*u[2:]   - dt*u[1:-1]*u_x[1:-1]
        rhs[0]        = (1-2*r)*u[0]  + r*u[1]                  - dt*u[0] *u_x[0]
        rhs[-1]       = r*u[-2] + (1-2*r)*u[-1]                 - dt*u[-1]*u_x[-1]

        u = solve_banded((1, 1), ab, rhs)
        if step in needed:
            u_store[step] = u.copy()

    # Interpolate stored snapshots onto (x_out, t_out)
    u_out = np.zeros((len(x_out), len(t_out)), dtype=np.float32)
    for j, tv in enumerate(t_out):
        step_idx = int(np.argmin(np.abs(t_fd - tv)))
        u_snap   = u_store.get(step_idx, u_store[max(u_store.keys())])
        # IC snapshot (step 0) uses x_fine which excludes boundaries (u=0 there)
        u_out[:, j] = np.interp(x_out, x_fine, u_snap,
                                left=0.0, right=0.0).astype(np.float32)
    return u_out


# ------------------------------------------------------------------ #
#  Main generation routine                                             #
# ------------------------------------------------------------------ #

def generate(n_samples: int = N_SAMPLES,
             out_path: str = "outputs/dataset.npz",
             seed: int = 42) -> None:
    """Generate dataset and save to `out_path`."""

    rng = np.random.default_rng(seed)

    # Grids
    x_out  = np.linspace(X_MIN, X_MAX, N_X,  dtype=np.float32)
    t_out  = np.linspace(T_MIN, T_MAX, N_T,  dtype=np.float32)

    # FD internal grid (higher resolution, no boundary points)
    N_fd   = 512
    dx_fd  = 2.0 / (N_fd + 1)
    x_fine = np.linspace(X_MIN + dx_fd, X_MAX - dx_fd, N_fd, dtype=np.float32)

    u0_all = np.zeros((n_samples, N_X),       dtype=np.float32)
    u_all  = np.zeros((n_samples, N_X, N_T),  dtype=np.float32)

    print(f"[data_gen] Generating {n_samples} samples  "
          f"(N_x={N_X}, N_t={N_T}, nu={NU:.6f})")
    t_start = time.time()

    for i in range(n_samples):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed if elapsed > 0 else float("inf")
            eta     = (n_samples - i - 1) / rate if rate > 0 else 0
            print(f"  sample {i+1:>5}/{n_samples}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        # Sample IC on the output grid directly (FD grid is internal)
        u0 = random_ic(x_out, rng)
        # Interpolate IC onto the FD grid for the solver
        u0_fine = np.interp(x_fine, x_out, u0).astype(np.float32)

        u_sol  = fd_solve(u0_fine, x_fine, x_out, t_out, nu=NU)

        u0_all[i] = u0
        u_all[i]  = u_sol

    elapsed_total = time.time() - t_start
    print(f"[data_gen] Done in {elapsed_total:.1f}s  "
          f"({elapsed_total/n_samples:.2f}s/sample)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path,
                        u0=u0_all, u=u_all,
                        x_grid=x_out, t_grid=t_out)
    print(f"[data_gen] Saved to '{out_path}'  "
          f"(u0: {u0_all.shape}, u: {u_all.shape})")


# ------------------------------------------------------------------ #
#  Dataset loader                                                      #
# ------------------------------------------------------------------ #

def load_dataset(path: str = "outputs/dataset.npz") -> dict:
    """
    Load the saved dataset and return a dict with train/val/test splits.

    Keys
    ----
    u0_train, u_train  : (800, N_x) and (800, N_x, N_t)
    u0_val,   u_val    : (100, N_x) and (100, N_x, N_t)
    u0_test,  u_test   : (100, N_x) and (100, N_x, N_t)
    x_grid             : (N_x,)
    t_grid             : (N_t,)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.  "
            f"Run  python data_gen.py  to generate it first."
        )
    data = np.load(path)
    n    = len(data["u0"])
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    return dict(
        u0_train = data["u0"][:n_train],
        u_train  = data["u"][:n_train],
        u0_val   = data["u0"][n_train : n_train + n_val],
        u_val    = data["u"][n_train : n_train + n_val],
        u0_test  = data["u0"][n_train + n_val :],
        u_test   = data["u"][n_train + n_val :],
        x_grid   = data["x_grid"],
        t_grid   = data["t_grid"],
    )


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FNO training dataset")
    parser.add_argument("--n_samples", type=int,  default=N_SAMPLES)
    parser.add_argument("--out_path",  type=str,  default="outputs/dataset.npz")
    parser.add_argument("--seed",      type=int,  default=42)
    args = parser.parse_args()

    generate(n_samples=args.n_samples, out_path=args.out_path, seed=args.seed)
