# Darcy 2D PINN — Notes & Setup

**Equation:** Steady-state 2-D Darcy flow (variable-permeability pressure equation)

```
-∇·(k(x,y) · ∇u(x,y)) = f(x,y)   on Ω = [0,1]²
u = 0                               on ∂Ω (all four edges)
```

This is the first **2-D, time-independent** problem in the project.
Previous sub-projects (Burgers' 1-D, advection-diffusion 1-D, inverse 1-D) all
had a time axis. Darcy flow is steady-state: the network maps `(x, y) → u` with
no `t` input at all, and the PDE has no time derivative.

---

## 1. Permeability Field

```
k(x, y) = 1 + 0.5 · sin(πx) · sin(πy)
```

**Range:** k ∈ [0.5, 1.5] over [0,1]² — always positive, so the problem is
well-posed (elliptic) everywhere.

**Why this choice:**
- Smooth and spatially varying: tests the PINN's ability to handle a
  non-constant coefficient PDE.
- Simple closed-form derivatives: `k_x` and `k_y` are cheap to compute
  analytically, avoiding the need to differentiate k through autograd.
- Bounded away from zero: no near-singular regions, so the PDE is uniformly
  elliptic and well-conditioned.

---

## 2. Manufactured Solution (Method of Manufactured Solutions)

We choose an exact solution and **derive** the source term f by substitution.
This guarantees a known ground truth for validation without solving the PDE
numerically.

**Exact solution:**
```
u*(x, y) = sin(πx) · sin(πy)
```

**Why this choice:**
- Satisfies u=0 on all four edges of [0,1]² automatically:
  sin(0)=sin(π)=0, so the BC is consistent with the MMS.
- Smooth, non-trivial spatial variation: tests both x and y directions
  symmetrically.
- Peak value = 1 at (0.5, 0.5), decaying to 0 at all boundaries.

---

## 3. Source Term Derivation

Expanding `-∇·(k ∇u*)`:

```
∇u* = [ π·cos(πx)·sin(πy),   π·sin(πx)·cos(πy) ]

k_x = 0.5·π·cos(πx)·sin(πy)
k_y = 0.5·π·sin(πx)·cos(πy)

u*_xx = -π²·sin(πx)·sin(πy)
u*_yy = -π²·sin(πx)·sin(πy)

∂/∂x [k · u*_x]  =  k_x · u*_x  +  k · u*_xx
                  =  0.5π²·cos²(πx)·sin²(πy)  -  π²·k·sin(πx)·sin(πy)

∂/∂y [k · u*_y]  =  k_y · u*_y  +  k · u*_yy      (symmetric: x↔y)
                  =  0.5π²·sin²(πx)·cos²(πy)  -  π²·k·sin(πx)·sin(πy)
```

Therefore:

```
f(x,y) = -div(k ∇u*)
        = -(  ∂/∂x[k u*_x]  +  ∂/∂y[k u*_y]  )
        =  2π²·k·sin(πx)sin(πy)
           - 0.5π²·cos²(πx)·sin²(πy)
           - 0.5π²·sin²(πx)·cos²(πy)
```

Substituting `k = 1 + 0.5·sin(πx)sin(πy)`:

```
f(x,y) = 2π²·(1 + 0.5·sin(πx)sin(πy))·sin(πx)sin(πy)
          - 0.5π²·cos²(πx)·sin²(πy)
          - 0.5π²·sin²(πx)·cos²(πy)
```

This is implemented in [`data.py::source_term()`](data.py).

---

## 4. PINN Formulation

The PINN minimises:

```
L = λ_pde · L_pde  +  λ_bc · L_bc

L_pde = mean_over_collocation{ R(x,y)² }
R(x,y) = -(k_x·u_x + k·u_xx + k_y·u_y + k·u_yy) - f(x,y)

L_bc  = mean_over_boundary{ u(x_b, y_b)² }   (target = 0)
```

**Key implementation details:**
- `u_x`, `u_y`, `u_xx`, `u_yy` are computed via `torch.autograd.grad` with
  `create_graph=True` (needed so the backward pass can propagate through these
  derivatives into the network weights).
- `k`, `k_x`, `k_y` are computed analytically (no autograd through k) using
  `.detach()` coordinates — this avoids a secondary graph through the
  permeability and keeps the residual computation efficient.
- `f(x,y)` is also computed analytically from detached coordinates.
- `λ_bc` is auto-balanced at initialisation so that `λ_pde·L_pde ≈ λ_bc·L_bc`
  at epoch 0, following the fix proven effective in the inverse problem
  sub-project.

---

## 5. Optimisation Schedule

| Phase | Epochs | Optimiser | Collocation |
|-------|--------|-----------|-------------|
| 1 — Warm-up | 1–3,000 | Adam (lr=1e-3, ReduceLROnPlateau) | Re-sampled each epoch |
| 2 — Refinement | 3,001–5,000 | L-BFGS (strong-Wolfe line search) | Fixed set |

**Why the two-phase schedule:**
- Adam is robust but only first-order; it explores the loss landscape broadly.
- L-BFGS has quasi-Newton curvature information and consistently tightens
  the solution by a further 30–60% once Adam has found a good basin (confirmed
  in the inverse-problem sub-project on Burgers' equation).
- The fixed collocation set in Phase 2 is essential: L-BFGS requires a
  deterministic loss function for its line search to work correctly.

---

## 6. Architecture

```
Input: (x, y)  →  2 neurons
Hidden: 5 layers × 64 neurons, tanh activation
Output: u       →  1 neuron (linear, no activation)
Total parameters: 2×64 + 5×(64×64) + 64×1 + biases ≈ 24,833
```

This is deeper than the 1-D Burgers PINN (4×50 = ~10k params) to reflect
the increased spatial complexity of the 2-D problem and the variable-
coefficient operator. The tanh activation is critical: the PDE loss requires
∂²u/∂x² and ∂²u/∂y², which need the network to be at least C² — tanh
satisfies this everywhere.

---

## 7. Differences vs 1-D PINNs in This Project

| Aspect | 1-D Burgers / Ocean | 2-D Darcy (this) |
|--------|---------------------|------------------|
| Inputs | (x, t) | (x, y) |
| Derivatives needed | u_t, u_x, u_xx | u_x, u_y, u_xx, u_yy |
| PDE type | Parabolic / hyperbolic (time-dep.) | Elliptic (steady-state) |
| Boundary data | IC at t=0 + BC at x=±1 | Dirichlet BC on all 4 edges |
| Reference solution | Crank-Nicolson FD | Analytical MMS |
| Nonlinearity | u·u_x (Burgers') | None (linear PDE, variable coeff.) |
| Hidden layers | 4 | 5 |
| Neurons | 50 | 64 |

The steady-state nature means there are **no initial condition points** — only
boundary points and interior collocation points. The elliptic problem has
globally coupled spatial structure (information propagates in all directions
simultaneously), which is why more capacity (depth + width) is useful.

---

## 8. Final Results (Colab GPU — T4)

Run environment: Google Colab, T4 GPU, 5,000 epochs (Adam 3,000 + L-BFGS 2,000).

| Metric | Value |
|--------|-------|
| **MSE** vs exact u* (256×256 grid) | **3.26 × 10⁻¹⁰** |
| **Rel. L2 error** | **3.63 × 10⁻⁵  (0.004%)** |
| Training time (T4) | 6.1 min (364 s) |
| Collocation points | 10,000 / epoch (Adam), fixed set (L-BFGS) |
| Boundary points | 200 per edge × 4 edges = 800 total |

The MSE of **3.26 × 10⁻¹⁰** is approximately **2,400× lower** than the
1-D Burgers' ensemble MSE (7.82 × 10⁻⁴), reflecting three advantages of
the Darcy problem:

1. **Linearity** — no u·u_x term; the PDE residual loss surface is convex in
   the network weights, so gradient descent converges reliably without
   stagnation near shock-like features.
2. **Analytical reference** — the MMS exact solution has no discretisation
   error, so the error floor is limited only by network capacity and
   optimisation, not numerical reference accuracy.
3. **L-BFGS effectiveness on smooth elliptic problems** — the steady-state
   elliptic structure gives L-BFGS a particularly well-conditioned loss
   landscape; the quasi-Newton curvature estimate is accurate and the
   strong-Wolfe line search converges in few iterations.

All three output figures are present in `outputs/`:

| Figure | Description |
|--------|-------------|
| [`solution_comparison.png`](outputs/solution_comparison.png) | Three-panel heatmap: PINN prediction / exact u* / pointwise error. The error panel is visually uniform near machine precision across the full domain. |
| [`loss_history.png`](outputs/loss_history.png) | Log-scale training curves (total, PDE, BC) with Adam→L-BFGS transition marked. The L-BFGS phase drives a sharp final descent. |
| [`pde_residual_map.png`](outputs/pde_residual_map.png) | Spatial map of \|R(x,y)\| at 128×128 resolution. Residual is uniformly near-zero across the domain — no localised hotspots. |

---

## 9. File Structure

```
darcy_2d/
├── model.py          DarcyPINN: (x,y) → u, 5×64 tanh, Xavier init
├── data.py           Collocation/boundary samplers, exact solution, source term f
├── train.py          Adam + L-BFGS schedule, auto-balanced λ_bc, checkpoint save
├── plot.py           3-panel solution comparison, loss history, PDE residual map
├── main.py           Entry point: train → plot → print metrics
├── NOTES.md          This file
├── requirements.txt  Python dependencies
└── outputs/          ✓ populated from Colab T4 run
    ├── darcy_pinn.pt             trained checkpoint (5,000 epochs)
    ├── metrics.json              MSE, Rel-L2, train time, hyperparameters
    ├── solution_comparison.png   3-panel heatmap (PINN / exact / error)
    ├── loss_history.png          training loss curves (log scale)
    └── pde_residual_map.png      PDE residual spatial map (128×128)
```

---

## 10. How to Run (Colab GPU)

```python
# In a Colab notebook cell:
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "numpy", "matplotlib", "scipy"], check=True)

import os; os.chdir("darcy_2d")
from main import main
results = main()
```

Or from the terminal:
```bash
cd darcy_2d
python main.py
# Optional flags:
python main.py --n_epochs 5000 --adam_epochs 3000 --n_col 10000
```
