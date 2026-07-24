# Ocean PINN — UQ Comparison Notes

**Problem:** 1D advection-diffusion equation modelling pollutant/heat transport
in an ocean current:

```
∂c/∂t + v·(∂c/∂x) = D·(∂²c/∂x²)
```

| Parameter | Value | Burgers' equivalent |
|---|---|---|
| Advection speed | v = 1.0 | — |
| Diffusivity | D = 0.05 | ν = 0.01/π ≈ 0.00318 |
| Domain (space) | x ∈ [0, 10] km | x ∈ [−1, 1] |
| Domain (time) | t ∈ [0, 5] | t ∈ [0, 1] |
| Initial condition | Gaussian pulse at x = 2 | −sin(πx) |
| Boundary conditions | c(0,t) = c(10,t) = 0 | u(±1,t) = 0 |

---

## 1. PINN Convergence

The advection-diffusion problem converges **dramatically faster** than Burgers':

| Metric | Advection-diffusion | Burgers' |
|---|---|---|
| Final PDE loss | **3.37e-6** | 9.02e-4 |
| Final IC loss | **2.55e-5** | 2.20e-5 |
| Final total loss | 3.33e-4 | 1.24e-3 |
| MSE vs reference | **7.0e-5** | 7.8e-4 |

The key reason is linearity: the advection-diffusion equation is **linear** in
the unknown c. The PDE residual loss surface is quadratic (convex) in the
network weights, making gradient descent reliably convergent. Burgers' equation
has the nonlinear term u·(∂u/∂x), which creates a more rugged loss landscape
and can cause stagnation near the shock region.

---

## 2. Uncertainty Structure — Core Comparison

### Burgers' equation (shock problem)
The ensemble uncertainty heatmap for Burgers' shows a **single sharp,
vertically-oriented yellow spike** at x ≈ 0 for all t > 0.3. The spike
collapses to a needle-thin line by t → 1 as the shock sharpens. Uncertainty
is essentially zero everywhere else in the domain. This structure is
**spatially localised and temporally growing** — it traces the nonlinear
steepening of the solution gradient.

### Advection-diffusion (this problem)
The uncertainty heatmap here is qualitatively different in three key ways:

**1. Uncertainty is non-zero across a wide spatial band, not a point.**
Rather than a thin spike, the high-uncertainty region occupies the entire
footprint of the Gaussian pulse — roughly a diagonal stripe of width ~4 km
that follows the moving peak. This makes physical sense: the ensemble members
disagree about the **exact shape** of the pulse (its amplitude, width, and
position), not just a single discontinuity.

**2. Uncertainty grows with time and with distance from the initial position.**
Regions the pulse has *already passed* (x < 2 + v·t) show lower uncertainty
than the leading edge and far-field tail. This reflects the fact that the PINN
must extrapolate the pulse evolution forward in time from only the IC data at
t = 0. The further in time (and thus further in space) the ensemble must project,
the more its members diverge.

**3. There is a dark (low-uncertainty) trough that follows the pulse's exact
trajectory.** The minimum-uncertainty line runs diagonally at slope dx/dt = v = 1,
precisely along the characteristic. This is the advection characteristic — the
curve along which information travels. The PINN learns the characteristic
direction well (it is enforced by both the IC and the PDE residual), but
uncertainty arises at the flanks where the solution decays toward zero and where
the IC's Gaussian shape must be perfectly reproduced.

---

## 3. Calibration Comparison

| Metric | Advection-diffusion | Burgers' |
|---|---|---|
| ECE | 0.1018 | 0.0835 |
| 90% coverage | 0.7691 | 0.8855 |

The Burgers' ensemble is better calibrated (lower ECE, closer 90% coverage to
the ideal 0.9). There are two explanations:

- **Uncertainty scale mismatch:** In the advection-diffusion case the ensemble
  std is very small (range 4.7e-4 to 9.3e-3) because all 10 members converge
  to a nearly identical, accurate solution. The ensemble has very low *spread*
  but non-zero *error* in the tails of the pulse, so the small uncertainty
  intervals fail to cover those errors — leading to under-coverage.

- **Solution simplicity:** Because the advection-diffusion problem is linear and
  the solution is smooth, all 10 members essentially find the same local minimum.
  Deep Ensembles derive calibrated uncertainty from members ending up in
  *different* basins — with a convex problem, this diversity is reduced.

**Key takeaway:** Deep Ensembles work best as a UQ method when the underlying
optimisation is genuinely multi-modal (as with nonlinear PDEs like Burgers')
so that different seeds lead to meaningfully different solutions. For
well-posed linear problems, the uncertainty is artificially compressed and
slightly under-covers the true error.

---

## 4. Uncertainty Follows Gradient, Not Just Magnitude

Both problems confirm the general principle observed in the PINN UQ
literature (Lakshminarayanan 2017, Psaros et al. 2023):

> **Ensemble uncertainty is concentrated where the solution gradient is
> largest, not where the solution value is largest.**

| Problem | Peak solution region | Peak uncertainty region |
|---|---|---|
| Burgers' | x ≈ ±0.3 (wave crests) | x ≈ 0 (shock gradient) |
| Adv-diff | x ≈ x₀ + v·t (pulse peak) | x ≈ pulse flanks (steepest slope) |

In the advection-diffusion case the peak concentration (pulse centre) is
actually a *local uncertainty minimum* at early times — the ensemble
members agree on the peak value because the IC directly constrains it.
They disagree on the wings where the Gaussian decays and where the
propagation error accumulates.

---

## 5. Implications for Environmental Modelling

For a real ocean transport problem, these results suggest:

- **Where to deploy sensors:** Uncertainty is highest at the *leading edge*
  and *lateral flanks* of a pollutant plume, not at the plume centre.
  Monitoring the advance of the front (which the ensemble is uncertain about)
  is more informative than repeated measurements at the known source location.

- **When uncertainty matters most:** Predictions at late times (large t) and
  far from the source carry the highest uncertainty, consistent with the
  physical intuition that predictability decreases with advection distance.

- **Linear vs nonlinear transport:** If the transport model includes nonlinear
  effects (e.g., concentration-dependent diffusivity, reaction terms, or
  boundary layer interactions), the uncertainty structure would more resemble
  the Burgers' case — sharper, more localised, and harder to capture with
  only 10 ensemble members.

---

## 6. File Structure

```
ocean_pinn/
├── model.py          # OceanPINN (same 4×50 tanh architecture as burgers_pinn)
├── data.py           # Domain constants, Gaussian IC, samplers, eval grid
├── train.py          # PDE residual (autograd) + IC/BC loss + training loop
├── plot.py           # Heatmap, time-slices vs FD reference, loss history
├── ensemble.py       # load_ensemble, ensemble_predict, calibration_metrics
├── ensemble_plot.py  # Mean heatmap, std heatmap, time slices, calibration
├── main.py           # Single-model entry point
├── run_ensemble.py   # Deep Ensemble (10 members) entry point
├── NOTES.md          # This file
└── outputs/
    ├── ocean_pinn.pt
    ├── heatmap.png
    ├── time_slices.png
    ├── loss_history.png
    └── ensemble/
        ├── model_{0..9}.pt
        ├── ensemble_mean_heatmap.png
        ├── ensemble_std_heatmap.png
        ├── ensemble_time_slices.png
        └── ensemble_calibration.png
```
