<div align="center">

![Uncertainty-Aware Physics-Informed Learning](./assets/banner.svg)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-Dashboard-61DAFB?style=for-the-badge&logo=react&logoColor=black)](./dashboard)
[![Status](https://img.shields.io/badge/Status-Complete-4fb8ff?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](#)

[Overview](#overview) |
[Phases](#phase-by-phase-summary) |
[Key Findings](#key-findings-across-phases) |
[Dashboard](#running-the-dashboard) |
[Contact](#contact)

### 🔗 [Live Dashboard](https://pinn-uncertainty-quantification-nine.vercel.app/)

</div>

---

## Overview

A research portfolio exploring physics-informed neural networks (PINNs) combined
with uncertainty quantification (UQ) across a range of PDEs -- from 1-D shocks to
2-D flow, inverse problems, and operator learning.

The project follows one consistent thread throughout: **it is not enough for a
model to produce an answer -- it should also know, and honestly report, how
confident it is.** Each phase tests that idea against a different kind of
difficulty: nonlinearity, sparse data, higher dimensions, and generalization
across problem instances.

> An interactive dashboard presenting all results lives in [`dashboard/`](./dashboard/) -- see [Running the dashboard](#running-the-dashboard).

<br>

## Phase-by-phase summary

<table>
<tr><td width="90"><b>P0-2</b></td><td>

### Burgers' equation baseline
A vanilla PINN solving the viscous Burgers' equation, validated against a Crank-Nicolson finite-difference reference solver rather than an approximate closed form.

| Metric | Value |
|---|---|
| MSE | `7.8e-4` |
| Shock | 🟢 cleanly captured |

[`burgers_pinn/`](./burgers_pinn/)

</td></tr>
</table>

<details>
<summary><b>P3 -- Uncertainty quantification comparison</b> (click to expand)</summary>
<br>

Three UQ methods compared on identical terms:

| Method | MSE | ECE (calibration) | 90% Coverage |
|---|---|---|---|
| 🟢 **Deep Ensemble (winner)** | `7.82e-4` | **0.084** | **88.6%** |
| 🔴 Bayesian PINN (VI) | `8.92e-2` | 0.0768 | 68.7% |
| 🟡 MC Dropout | `9.44e-3` | 0.137 | 83.9% |

Deep Ensembles won decisively on both accuracy and calibration. The Bayesian PINN's mean-field variational posterior produced spatially *uniform* uncertainty -- a known failure mode when independent weight distributions are a poor fit for PDE residual losses.

[`burgers_pinn/COMPARISON.md`](./burgers_pinn/COMPARISON.md)

</details>

<details>
<summary><b>Ocean / Red Sea advection-diffusion</b> (click to expand)</summary>
<br>

The same Deep Ensemble UQ pipeline applied to a linear advection-diffusion equation, modeling ocean pollutant/heat transport.

| Metric | Value |
|---|---|
| ECE | `0.102` |
| 90% Coverage | `76.9%` |

The key insight: ensemble members agree much more closely on this **linear** PDE (a single dominant solution basin), so their disagreement is a *weaker, less informative* uncertainty signal than in the nonlinear Burgers' case -- suggesting **Deep Ensembles are more informative for nonlinear PDEs than linear ones.**

[`ocean_pinn/`](./ocean_pinn/)

</details>

<details>
<summary><b>P4 -- Inverse problem: parameter recovery</b> (click to expand)</summary>
<br>

Recovering the unknown viscosity (nu) from sparse, noisy sensor data -- 9 conditions x 10 ensemble members.

A first attempt produced a **400-700% error** regardless of noise or sensor count -- a red flag for a genuine bug. Root-cause analysis found two compounding issues (loss-term imbalance + a starved gradient signal for nu sharing an optimizer with 200+ network weights). The fix -- Adam warm-up then L-BFGS refinement, auto-balanced loss -- brought error down to:

| Sensors | Noise 0.5% | Noise 1.0% | Noise 2.0% |
|---|---|---|---|
| 20  | 🔴 78.5% | 🔴 183.7% | 🔴 240.8% |
| 50  | 🔴 106.2% | 🟡 93.0% | 🟡 58.5% |
| 100 | 🟡 94.5% | 🟢 28.3% | 🟢 **17.6%** |

CI coverage: 2 / 9 conditions

[`inverse_problem/`](./inverse_problem/)

</details>

<details>
<summary><b>P5 -- Neural Operator vs. per-instance PINN</b> (click to expand)</summary>
<br>

A Fourier Neural Operator (FNO) trained once across 800 initial conditions, vs. a PINN retrained per instance.

| Method | Rel. L2 error | Training | Inference |
|---|---|---|---|
| 🟢 **FNO (winner)** | **6.98%** | 76s (one-time) | ~2.7-5.8 ms |
| 🔴 PINN (per-instance) | 32.75% | 22s **per instance** | ~1 ms |

FNO is ~4.7x more accurate, and becomes cheaper than retraining a PINN after only ~4 new scenarios.

[`neural_operator/`](./neural_operator/)

</details>

<details>
<summary><b>2-D Darcy flow extension</b> (click to expand)</summary>
<br>

Extending the pipeline from 1-D to 2-D using a manufactured solution, verified numerically to `1e-14` precision.

| Metric | Value |
|---|---|
| MSE | `3.26e-10` |
| Relative L2 error | 🟢 `0.004%` |
| Training time | `6.1 min` |

[`darcy_2d/`](./darcy_2d/)

</details>

<details>
<summary><b>P6 -- Validation: failure analysis & ablations</b> (click to expand)</summary>
<br>

**Failure analysis** -- sweeping viscosity to sharpen the shock:

| Shock sharpness | Error |
|---|---|
| 1x (baseline) | 🟢 4.56% |
| 2x | 🟡 20.8% |
| 5x | 🔴 33.3% |
| 10x | 🔴 34.8% |

**Ablation A (ensemble size)** -- MSE improves monotonically (M=3 to 20), but calibration is *not* monotonic (M=10 best calibrated).

**Ablation B (loss weighting)** -- the surprise:

| Scheme | Rel. L2 error |
|---|---|
| 🟢 **Fixed weighting (winner)** | **4.21%** |
| 🟡 Uniform | 6.70% |
| 🔴 Auto-balanced | 51.21% |

Auto-balancing (which fixed the inverse problem) performed **worst** here -- proof that techniques don't transfer automatically.

[`burgers_pinn/`](./burgers_pinn/)

</details>

<details>
<summary><b>P7 -- Interactive dashboard</b> (click to expand)</summary>
<br>

A React/Vite dashboard presenting every phase above with data loaded live from each phase's actual output files -- nothing fabricated.

[`dashboard/`](./dashboard/)

</details>

<br>

## Key findings across phases

1. **Ensemble UQ's value depends on problem structure.** Informative for nonlinear PDEs with multiple solution basins (Burgers'), much less so for linear PDEs (ocean/advection-diffusion).
2. **A systematic bug can look like an intractable problem until it's isolated.** The inverse problem's 400-700% error resembled "this is just hard" -- it was two fixable issues.
3. **Operator learning and per-instance PINNs solve different problems.** FNO's advantage compounds with the number of scenarios needed.
4. **Techniques don't transfer automatically.** Auto-balanced loss weighting fixed the inverse problem and *broke* standard forward training.

## Scope and future work

Left out by deliberate design, not failure:

- 2-D/3-D extensions beyond Darcy flow (e.g., Navier-Stokes)
- Calibration correction for the inverse problem's CI under-coverage (conformal prediction, M>=30)
- Physics-informed regularization for the FNO's training loss

## Running the dashboard

🔗 **Live:** [pinn-uncertainty-quantification-nine.vercel.app](https://pinn-uncertainty-quantification-nine.vercel.app/)

To run locally instead:

```bash
cd dashboard
npm install
npm run dev
```

<div align="center">

## Contact

**Malyika Muskaan**

[![GitHub](https://img.shields.io/badge/GitHub-MalyikaMuskaan-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MalyikaMuskaan)
[![Email](https://img.shields.io/badge/Email-malyikamuskaann%40gmail.com-4fb8ff?style=for-the-badge&logo=gmail&logoColor=white)](mailto:malyikamuskaann@gmail.com)

*Open to discussing this research, collaboration, or PhD opportunities in AI/ML and scientific machine learning.*

</div>
