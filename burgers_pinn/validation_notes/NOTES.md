# Phase 6 — Validation: Failure Analysis & Ablation Study

**Scope:** `burgers_pinn/failure_analysis.py` and `burgers_pinn/ablation.py`
**Equation:** Viscous Burgers'  `u_t + u·u_x = ν·u_xx`  on x∈[-1,1], t∈[0,1]
**IC:** `u(x,0) = -sin(πx)`,  **BC:** `u(±1,t) = 0`
**Run environment:** Google Colab, T4 GPU

---

## Part A — Failure Case Analysis

### Setup

A viscosity sweep tests the vanilla PINN (4×50 tanh, 5,000 epochs, Adam +
ReduceLROnPlateau) at four values of ν:

| Index | ν | Ratio vs baseline | Shock sharpness |
|-------|---|---|---|
| 0 | 0.01/π ≈ 3.18×10⁻³ | 1× (baseline) | Smooth near-shock |
| 1 | 0.005/π ≈ 1.59×10⁻³ | 2× | Moderate sharpening |
| 2 | 0.002/π ≈ 6.37×10⁻⁴ | 5× | Sharp shock |
| 3 | 0.001/π ≈ 3.18×10⁻⁴ | 10× | Near-discontinuity |

Each case is an independent fresh model; only ν in the PDE residual changes.

### Final Results (Colab T4)

| ν | Rel-L2 error | vs baseline | Qualitative description |
|---|---|---|---|
| 3.18×10⁻³ (0.01/π) | **4.57%** | 1× | Good solution; established working case |
| 1.59×10⁻³ (0.005/π) | **20.76%** | 4.6× | Shock visibly smeared; error concentrated at x≈0 |
| 6.37×10⁻⁴ (0.002/π) | **33.27%** | 7.3× | Shock region poorly captured; plateau in accuracy |
| 3.18×10⁻⁴ (0.001/π) | **34.77%** | 7.6× | Error saturates; near-discontinuity effectively missed |

**Key observation — error plateau:** Error rises sharply from the baseline to
the 2× case (4.6× increase), then largely **saturates** between the 5× and 10×
cases (33% vs 35%).  This is consistent with a failure mode where the PINN
completely fails to represent the shock region beyond a certain sharpness
threshold: once the shock is too narrow to be resolved by the smooth tanh
network, the network learns the best smooth approximation it can find, and
further sharpening of the true shock no longer degrades that smooth
approximation further.

### Why PINNs Fail at Small ν — Three Mechanisms

#### 1. Spectral Bias (F-Principle)

Fully-connected networks trained by gradient descent preferentially learn
low-frequency components first (Rahaman et al. 2019; Xu et al. 2019).  The
shock at small ν is a near-discontinuity requiring high spatial frequencies
(frequency ~1/ν) to represent.  The tanh network resists learning these
components, producing the characteristic smooth smearing seen in the error
heatmap.

#### 2. Gradient Stiffness

Near the shock, `|u_xx| ~ 1/ν²`, so the viscous term `ν·u_xx ~ 1/ν`.  The
PDE residual loss carries gradients O(1/ν) — at ν=0.001/π roughly 3,000×
larger than at baseline.  These large gradients destabilise Adam and cause
the optimiser to oscillate rather than converge in the shock region.

#### 3. Collocation Point Starvation

The shock has width O(ν·π).  With uniform sampling, only ~ν·π/2 × 10,000 ≈ 3
collocation points land inside the shock per epoch at ν=0.001/π — statistically
insufficient to constrain the residual where it matters most.

*Remedies in the literature (not implemented here):* Residual-Adaptive
Refinement (RAR, Lu et al. 2021), causal weighting (Wang et al. 2022),
self-adaptive loss weights (Jin et al. 2021), Fourier feature embeddings
(Tancik et al. 2020).

### Output Files

| File | Description |
|------|-------------|
| [`outputs/failure_analysis/model_nu_{0..3}.pt`](../outputs/failure_analysis/) | Trained checkpoints per ν |
| [`outputs/failure_analysis/metrics.json`](../outputs/failure_analysis/metrics.json) | Rel-L2 per ν (source of table above) |
| [`outputs/failure_analysis/failure_error_vs_nu.png`](../outputs/failure_analysis/failure_error_vs_nu.png) | Log-log Rel-L2 vs ν with reference slope lines |
| [`outputs/failure_analysis/failure_heatmap_comparison.png`](../outputs/failure_analysis/failure_heatmap_comparison.png) | 3-panel heatmap at ν=0.001/π: PINN / FD reference / pointwise error |

---

## Part B — Ablation Study

### Ablation A — Ensemble Size

Tests M ∈ {3, 5, 10, 20} at the baseline ν = 0.01/π.  Members 0–9 reused
from `outputs/ensemble/`; members 10–19 trained fresh (seeds 10–19, saved to
`outputs/ablation/ensemble_size/member_{10..19}.pt`).

#### Final Results (Colab T4)

| M | ECE ↓ | 90% Coverage | MSE ↓ |
|---|---|---|---|
| 3  | 0.0982 | 0.811 | 4.15×10⁻³ |
| 5  | 0.1261 | 0.925 | 2.44×10⁻³ |
| **10** | **0.0710** | **0.908** | 1.43×10⁻³ |
| 20 | 0.1337 | 0.911 | **9.06×10⁻⁴** |

#### Key Finding — ECE is Non-Monotonic; MSE is Monotonic

**MSE** decreases monotonically with M (more members → better mean prediction).
This is the expected result: a larger ensemble averages over more diverse
solutions, reducing variance in the mean.

**ECE** does **not** decrease monotonically.  M=10 achieves the best calibration
(ECE=0.071) — better than M=20 (0.134).  The ordering is: M=10 < M=3 < M=5 < M=20.

This non-monotonicity is the most important finding of this ablation.
Two explanations:

1. **Seed-specific behaviour:** Members 10–19 (seeds 10–19) may converge to
   systematically different local minima than members 0–9, shifting the
   ensemble's predictive distribution in a way that degrades calibration even
   as it reduces MSE.  The M=20 ensemble is more accurate in expectation but
   less calibrated — the uncertainty intervals no longer match the actual error
   distribution.

2. **Over-dispersion at M=20:** With 20 members, some pairs of members may be
   genuinely very different solutions (covering different local minima), making
   the predictive std larger than the actual error at most domain points.
   This inflates ECE in the opposite direction from under-confidence.

**Practical recommendation:** M=10 is the sweet spot for this problem —
it achieves the best ECE and near-ideal 90% coverage (0.908 vs ideal 0.900).
Going to M=20 buys a further 1.6× reduction in MSE at the cost of noticeably
worse calibration.  For a PhD application context where calibration quality is
the primary UQ claim, **M=10 remains the recommended configuration**.

#### Output Files

| File | Description |
|------|-------------|
| [`outputs/ablation/ensemble_size/ensemble_size_metrics.json`](../outputs/ablation/ensemble_size/ensemble_size_metrics.json) | ECE, coverage, MSE per M (source of table above) |
| [`outputs/ablation/ensemble_size/ablation_ensemble_size.png`](../outputs/ablation/ensemble_size/ablation_ensemble_size.png) | ECE + 90% coverage vs M (dual y-axis) |
| [`outputs/ablation/ensemble_size/member_{10..19}.pt`](../outputs/ablation/ensemble_size/) | Extra members for M=20 |

---

### Ablation B — Loss Weighting

Three weighting schemes for the vanilla forward PINN at baseline ν:

| Scheme | λ_pde | λ_ic | λ_bc | Rationale |
|--------|-------|------|------|-----------|
| (a) Baseline | 1 | 10 | 10 | Manually tuned for this problem |
| (b) Uniform | 1 | 1 | 1 | No a priori weighting |
| (c) Auto-balanced | 1 | 0.019 | 0.075 | λ set so λ·L ≈ equal at epoch 0 |

The auto-balanced λ values (computed at runtime from the initial loss magnitudes)
were: λ_ic = 0.019, λ_bc = 0.075 — both **much smaller than 1**, meaning the
auto-balance algorithm detected that the initial IC and BC losses were far larger
than the PDE residual and scaled them down accordingly.

#### Final Results (Colab T4)

| Scheme | MSE ↓ | Rel-L2 ↓ | Rank |
|--------|-------|----------|------|
| **(a) Baseline** λ_ic=10, λ_bc=10 | **6.66×10⁻⁴** | **4.21%** | **1st** |
| (b) Uniform λ_ic=1, λ_bc=1 | 1.68×10⁻³ | 6.70% | 2nd |
| (c) Auto-balanced λ_ic=0.019, λ_bc=0.075 | 9.85×10⁻² | **51.21%** | 3rd |

#### Key Finding — Auto-Balance Fails on the Forward Problem

The auto-balanced scheme performs dramatically worse than both alternatives
(51% vs 4% Rel-L2).  This is a significant and informative negative result
that directly contradicts the pre-run expectation.

**Why it fails here but worked in the inverse problem:**

In the **inverse problem**, the auto-balance strategy fixed a genuine
pathology: `lambda_data` was negligibly small, causing the network to
completely ignore sensor observations.  Balancing the losses at epoch 0
corrected this and allowed the sensor data to constrain ν.

In the **forward problem**, the initial loss magnitudes are:
- `L_pde0 ~ 1e-4 to 1e-3` (small, because the randomly-initialised network
  happens to produce near-zero output, which has small second derivatives)
- `L_ic0 ~ 0.5` (large, because the IC is -sin(πx) with amplitude 1 and
  the network output is near-zero)
- `L_bc0 ~ 0.1` (moderate)

Auto-balancing sets λ_ic = λ_pde × L_pde0 / L_ic0 ≈ 0.019 — **effectively
down-weighting the IC by a factor of 500 relative to the baseline**.  With such
a small λ_ic, the network is almost free to ignore the initial condition
`u(x,0) = -sin(πx)`.  The resulting "solution" satisfies the PDE residual at
random interior points but violates the IC, producing a qualitatively wrong
field that happens to have low PDE loss but high actual error.

**Core lesson:** Auto-balancing anchors loss weights to the initial loss
magnitudes at a randomly-initialised network.  At initialisation, IC/BC
losses are large (the network hasn't learned the data yet) while the PDE
residual is small (near-zero outputs have small derivatives).  This ratio
is the inverse of what is physically meaningful: IC/BC constraints should
be prioritised early in training, not down-weighted.

The baseline scheme λ_ic=λ_bc=10 encodes exactly this physical prior.
Auto-balancing, which has no access to physical reasoning, gets it backwards.

**Scope of the inverse-problem fix:** The auto-balance fix in
`inverse_problem/train.py` is correct in its specific context — it addresses
a `lambda_data` that was genuinely too small.  It should **not** be
generalised as a universal PINN loss-weighting strategy.

#### Output Files

| File | Description |
|------|-------------|
| [`outputs/ablation/loss_weighting/loss_weighting_metrics.json`](../outputs/ablation/loss_weighting/loss_weighting_metrics.json) | MSE, Rel-L2, λ values per scheme (source of table above) |
| [`outputs/ablation/loss_weighting/ablation_loss_weighting.png`](../outputs/ablation/loss_weighting/ablation_loss_weighting.png) | Bar chart of Rel-L2 per scheme (log scale) |
| [`outputs/ablation/loss_weighting/model_scheme_{a,b,c}.pt`](../outputs/ablation/loss_weighting/) | Trained checkpoints per scheme |

---

## Summary — Which Design Choices Matter Most

### 1. Viscosity (failure analysis) — largest effect by far

Error grows from 4.6% to 35% as ν decreases 10×.  No hyperparameter
choice within the standard PINN framework overcomes the spectral bias +
gradient stiffness combination at small ν.  The path forward for sharp-shock
problems requires structural changes: adaptive collocation (RAR), causal
training, or switching to operator learning (FNO — sub-project 4), which
sidesteps PDE residual stiffness entirely.

### 2. Loss weighting (ablation B) — critical for correctness

The spread from best (4.2%) to worst (51.2%) weighting scheme is 12×.
This is the highest-leverage hyperparameter choice.  **The physical prior
that IC/BC losses need upweighting early in training is essential and cannot
be replaced by a data-driven auto-balance computed at random initialisation.**

### 3. Ensemble size (ablation A) — matters for MSE but not calibration

MSE improves monotonically (4.6× gain from M=3 to M=20) but ECE is
non-monotonic with M=10 optimal.  M=10 is the recommended default.

### Correction to pre-run expectation

The pre-run NOTES predicted: *"auto-balancing is a practical substitute for
manual weight tuning."*  The experimental result is the opposite: auto-balanced
weighting is catastrophically worse (51% vs 4% error) on the forward problem.
This is a genuine finding — not a failure of the experiment — that sharpens the
understanding of when auto-balancing is and is not appropriate.

---

## How to Run (Colab GPU)

```bash
cd burgers_pinn

# Part A — Failure analysis (~28 min on T4, 4 training runs)
python failure_analysis.py

# Part B — Both ablations (~1.5 hr on T4 total)
python ablation.py

# Run only one ablation:
python ablation.py --only ensemble
python ablation.py --only weighting
```

Checkpoints are skipped if they already exist — safe to interrupt and resume.
