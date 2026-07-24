# Physics-Informed Neural Networks — Research Portfolio

**Context:** PhD application research project (KAUST Computational Science & Engineering)
**Focus:** Physics-Informed Neural Networks (PINNs), Uncertainty Quantification, Operator Learning, Inverse Problems, 2D PDEs, and systematic validation

All training was run on Google Colab (T4 GPU). Code is written and verified locally; outputs are committed after each Colab run.

---

## Project Structure

```
pnn/
├── burgers_pinn/          Phase 1–2: Baseline PINN + UQ comparison
├── ocean_pinn/            Phase 3:   Ocean/climate 1D PDE extension
├── inverse_problem/       Phase 4:   Inverse problem (ν recovery)
├── neural_operator/       Phase 5:   Fourier Neural Operator
├── darcy_2d/              Phase 5b:  2D Darcy flow PINN
├── COMPARISON.md          UQ method comparison summary (Phase 2)
└── README.md              This file
```

---

## Phases at a Glance

| Phase | Sub-project | PDE | Key result |
|-------|-------------|-----|------------|
| 1 | Burgers' baseline PINN | Viscous Burgers' 1D | MSE 7.82×10⁻⁴ vs Crank-Nicolson reference |
| 2 | UQ method comparison | Viscous Burgers' 1D | Deep Ensemble best: ECE 0.083, 90% coverage 0.886 |
| 3 | Ocean/climate PINN | Advection-diffusion 1D | MSE 7.0×10⁻⁵; linear PDE converges 10× faster |
| 4 | Inverse problem | Viscous Burgers' 1D | ν recovered to 17.6% error (100 sensors, 2% noise) after fixing lambda_data bug |
| 5a | Fourier Neural Operator | Viscous Burgers' 1D | FNO: 6.98% rel-L2 vs PINN: 32.75%; break-even at 4 instances |
| 5b | 2D Darcy flow PINN | Darcy flow 2D (elliptic) | MSE 3.26×10⁻¹⁰, rel-L2 0.004% — first 2D result |
| 6 | Failure analysis + ablation | Viscous Burgers' 1D | Shock sharpness dominates failure; loss weighting critical; ECE non-monotonic in M |

---

## Phase 1 — Burgers' Equation Baseline PINN

**Directory:** [`burgers_pinn/`](burgers_pinn/)

Solves the viscous Burgers' equation `u_t + u·u_x = ν·u_xx` (ν = 0.01/π)
with IC `u(x,0) = -sin(πx)` and zero Dirichlet BCs, validated against a
Crank-Nicolson finite-difference reference.

| Item | Value |
|------|-------|
| Architecture | 4 hidden layers × 50 neurons, tanh, Xavier init |
| Training | 15,000 epochs, Adam + ReduceLROnPlateau |
| Collocation | 10,000 points/epoch (resampled) |
| Final MSE | 7.82×10⁻⁴ |
| Reference solver | Crank-Nicolson FD, Nx=512, Nt=2000 |

**Key files:** [`model.py`](burgers_pinn/model.py) · [`train.py`](burgers_pinn/train.py) · [`main.py`](burgers_pinn/main.py)

---

## Phase 2 — Uncertainty Quantification Comparison

**Directory:** [`burgers_pinn/`](burgers_pinn/) · **Summary:** [`COMPARISON.md`](COMPARISON.md)

Three UQ methods applied to the same Burgers' PINN setup, evaluated on a 256×100 grid against the Crank-Nicolson reference.

| Method | MSE ↓ | ECE ↓ | 90% Coverage | Train time | Infer time |
|--------|-------|-------|---|---|---|
| **Deep Ensemble** (M=10) | **7.82×10⁻⁴** | **0.083** | **0.886** | 89 min | 0.1 s |
| Bayesian PINN (VI, 200 MC) | 8.89×10⁻² | 0.077 | 0.687 | **10 min** | 3.0 s |
| MC Dropout (p=0.05, 100 MC) | 9.39×10⁻³ | 0.138 | 0.839 | 35 min | 14.3 s |

**Winner:** Deep Ensemble — best on every accuracy and calibration metric.
**Key files:** [`run_ensemble.py`](burgers_pinn/run_ensemble.py) · [`run_bayesian.py`](burgers_pinn/run_bayesian.py) · [`run_dropout.py`](burgers_pinn/run_dropout.py) · [`compare_methods.py`](burgers_pinn/compare_methods.py)

---

## Phase 3 — Ocean/Climate PDE Extension

**Directory:** [`ocean_pinn/`](ocean_pinn/) · **Notes:** [`ocean_pinn/NOTES.md`](ocean_pinn/NOTES.md)

Extends the Deep Ensemble UQ approach to a 1D advection-diffusion equation modelling pollutant/heat transport in an ocean current: `∂c/∂t + v·∂c/∂x = D·∂²c/∂x²` (v=1, D=0.05, Gaussian IC).

| Metric | Advection-diffusion | Burgers' |
|--------|---------------------|----------|
| Final PDE loss | **3.4×10⁻⁶** | 9.0×10⁻⁴ |
| MSE vs reference | **7.0×10⁻⁵** | 7.8×10⁻⁴ |
| ECE (ensemble) | 0.102 | **0.083** |
| 90% coverage | 0.769 | **0.886** |

**Key finding:** The linear PDE converges ~10× faster, but ensemble calibration
is worse because all 10 members find the same convex minimum — Deep Ensembles
derive calibrated uncertainty from multi-modality, which requires a nonlinear problem.

**Key files:** [`run_ensemble.py`](ocean_pinn/run_ensemble.py) · [`train.py`](ocean_pinn/train.py)

---

## Phase 4 — Inverse Problem (ν Recovery)

**Directory:** [`inverse_problem/`](inverse_problem/) · **Notes:** [`inverse_problem/NOTES.md`](inverse_problem/NOTES.md)

Recovers the unknown viscosity ν from sparse noisy sensor observations using a PINN where `log_nu` is a learnable parameter. 9-condition robustness sweep: 3 noise levels × 3 sensor counts, 10 ensemble members each.

**Bug found and fixed:** A two-part bug caused 400–700% error before the fix:
1. `lambda_data` was negligibly small → network ignored sensor data
2. `log_nu` shared the same Adam LR as network weights → ν oscillated

**Fix:** Auto-balanced `lambda_data` at initialisation + two-stage Adam → L-BFGS schedule.

| Sensors | Best error | Worst error |
|---------|-----------|------------|
| 20 | 78.5% | 240.8% |
| 50 | 58.5% | 106.2% |
| 100 | **17.6%** | 94.5% |

CI calibration: 2/9 conditions captured true ν in 90% CI (ensemble underdispersed at M=10).

**Key files:** [`train.py`](inverse_problem/train.py) · [`robustness.py`](inverse_problem/robustness.py) · [`plot_robustness_analysis.py`](inverse_problem/plot_robustness_analysis.py)

---

## Phase 5a — Fourier Neural Operator

**Directory:** [`neural_operator/`](neural_operator/) · **Notes:** [`neural_operator/NOTES.md`](neural_operator/NOTES.md)

Trains a Fourier Neural Operator (FNO1d) to learn the solution operator
`G: u₀(x) → u(x,t)` for Burgers' equation — a single model that generalises to any IC without retraining.

| Metric | FNO | PINN (per-instance) |
|--------|-----|---------------------|
| Rel-L2 error (mean, 100 test ICs) | **6.98%** | 32.75% |
| Training time | **76 s** (one-time) | 22 s / instance |
| Inference time | ~2.7 ms | 1.0 ms |
| Generalises to new ICs? | **Yes** | No |
| Physics-constrained? | No | **Yes** |
| Parameters | 562,276 | ~10,400 |

**Break-even:** After 4 instances the FNO has repaid its training cost; for ≥5 ICs it is strictly cheaper. At steady-state the FNO is ~8,000× faster per query.

**Key files:** [`model.py`](neural_operator/model.py) · [`train.py`](neural_operator/train.py) · [`evaluate.py`](neural_operator/evaluate.py)

---

## Phase 5b — 2D Darcy Flow PINN

**Directory:** [`darcy_2d/`](darcy_2d/) · **Notes:** [`darcy_2d/NOTES.md`](darcy_2d/NOTES.md)

First **2D, time-independent** (elliptic) problem in the project. Solves steady-state Darcy flow with variable permeability:

```
-∇·(k(x,y) ∇u(x,y)) = f(x,y)   on [0,1]²,   u = 0 on ∂Ω
k(x,y) = 1 + 0.5·sin(πx)·sin(πy)
```

Exact solution `u* = sin(πx)sin(πy)` chosen via Method of Manufactured Solutions; `f` derived analytically and verified to 1×10⁻¹⁴ precision.

| Metric | Value |
|--------|-------|
| MSE vs exact u* (256×256 grid) | **3.26×10⁻¹⁰** |
| Rel-L2 error | **3.63×10⁻⁵ (0.004%)** |
| Training time (T4) | 6.1 min (Adam 3,000 + L-BFGS 2,000 epochs) |
| Parameters | 24,833 |

MSE is ~2,400× lower than the Burgers' ensemble: the elliptic PDE has a convex
residual landscape and the analytical MMS reference has no discretisation error.

**Key files:** [`model.py`](darcy_2d/model.py) · [`train.py`](darcy_2d/train.py) · [`data.py`](darcy_2d/data.py) · [`main.py`](darcy_2d/main.py)

---

## Phase 6 — Validation: Failure Analysis & Ablation Study

**Directory:** [`burgers_pinn/`](burgers_pinn/) · **Notes:** [`burgers_pinn/validation_notes/NOTES.md`](burgers_pinn/validation_notes/NOTES.md)

### Part A — Failure Case Analysis (ν sweep)

Demonstrates PINN accuracy degradation as the shock sharpens (ν decreasing).

| ν | Rel-L2 | vs baseline | Mechanism |
|---|---|---|---|
| 0.01/π ≈ 3.18×10⁻³ | 4.57% | 1× | Baseline — working |
| 0.005/π ≈ 1.59×10⁻³ | 20.76% | 4.6× | Shock smearing begins |
| 0.002/π ≈ 6.37×10⁻⁴ | 33.27% | 7.3× | Shock region effectively missed |
| 0.001/π ≈ 3.18×10⁻⁴ | 34.77% | 7.6× | **Error saturates** — plateau |

Error saturates after 5× sharpening: the PINN learns the best smooth approximation and further sharpening of the true shock no longer changes it. Root causes: spectral bias, gradient stiffness (PDE residual scales as 1/ν), collocation starvation (~3 points in the shock at smallest ν). **FD reference solver** was also fixed here — adaptive grid scaling (Nx, Nt ∝ 1/ν) to prevent overflow at small ν.

### Part B — Ablation Study

**Ablation A — Ensemble size {3, 5, 10, 20}:**

| M | ECE ↓ | 90% Coverage | MSE ↓ |
|---|---|---|---|
| 3 | 0.098 | 0.811 | 4.15×10⁻³ |
| 5 | 0.126 | 0.925 | 2.44×10⁻³ |
| **10** | **0.071** | **0.908** | 1.43×10⁻³ |
| 20 | 0.134 | 0.911 | **9.06×10⁻⁴** |

MSE decreases monotonically; **ECE is non-monotonic** — M=10 is best calibrated. M=20 over-disperses the predictive distribution, inflating ECE.

**Ablation B — Loss weighting {baseline, uniform, auto-balanced}:**

| Scheme | Rel-L2 |
|--------|--------|
| (a) λ_ic=10, λ_bc=10 (baseline) | **4.21%** |
| (b) λ_ic=1, λ_bc=1 (uniform) | 6.70% |
| (c) Auto-balanced (λ_ic=0.019, λ_bc=0.075) | **51.21%** |

**Key negative result:** Auto-balancing — which fixed the inverse problem — catastrophically fails here. At random initialisation, IC/BC losses are large and PDE loss is small, so auto-balance down-weights the IC by 500× relative to the baseline, letting the network ignore the initial condition. **The physical prior (IC/BC need upweighting early) cannot be replaced by a data-driven weight computed at random init.**

**Key files:** [`failure_analysis.py`](burgers_pinn/failure_analysis.py) · [`ablation.py`](burgers_pinn/ablation.py)

---

## Cross-Project Findings

### PINN accuracy vs problem type

| Problem | PDE type | Nonlinear? | Reference | Final rel-L2 |
|---------|----------|-----------|-----------|-------------|
| Burgers' 1D | Parabolic | Yes (u·u_x) | Crank-Nicolson FD | ~5% |
| Advection-diffusion 1D | Parabolic | No | Crank-Nicolson FD | ~3% |
| Darcy 2D | Elliptic | No | Analytical MMS | **0.004%** |
| Burgers' inverse (best) | Parabolic | Yes | — | 17.6% (ν error) |

Linear/elliptic problems converge reliably to near-machine-precision accuracy. Nonlinear parabolic problems plateau around 4–35% depending on shock sharpness.

### When to use each method

| Objective | Recommended approach |
|-----------|---------------------|
| Best forward accuracy + calibrated UQ | Deep Ensemble (M=10) |
| Many ICs, fast amortised inference | Fourier Neural Operator |
| Unknown parameter from sparse data | Inverse PINN + Adam → L-BFGS + auto-balance λ_data |
| 2D steady-state elliptic PDE | Single PINN + L-BFGS refinement |
| Understanding failure modes | See Phase 6 failure analysis |

### Transferability of the auto-balance fix

The `lambda_data` auto-balance from Phase 4 (inverse problem) **does not
transfer** to the standard forward PINN loss weighting (shown by Phase 6
ablation B, 51% vs 4% error). The fix is correct in context — it addresses a
genuine pathology (sensor data being ignored) — but the underlying mechanism
(anchoring weights to initial loss magnitudes at random init) gets the
direction backwards for IC/BC terms in forward problems.

---

## Reproducibility

```bash
# Phase 1–2: Burgers' baseline + UQ
cd burgers_pinn
python main.py                # single model
python run_ensemble.py        # Deep Ensemble (~89 min CPU / ~10 min T4)
python run_bayesian.py        # Bayesian PINN (~10 min)
python run_dropout.py         # MC Dropout (~35 min)
python compare_methods.py     # comparison plots (no training)

# Phase 3: Ocean PINN
cd ocean_pinn
python run_ensemble.py

# Phase 4: Inverse problem
cd inverse_problem
python main.py
python robustness.py          # full 9-condition sweep (~2 hr T4)

# Phase 5a: FNO
cd neural_operator
python data_gen.py            # generate dataset (~10 min CPU)
python train.py               # train FNO (76 s T4)
python evaluate.py            # compare vs PINN

# Phase 5b: 2D Darcy
cd darcy_2d
python main.py                # train + evaluate + plot (~6 min T4)

# Phase 6: Validation
cd burgers_pinn
python failure_analysis.py    # nu sweep (~28 min T4)
python ablation.py            # ensemble size + loss weighting (~1.5 hr T4)
```

All checkpoints are committed under their respective `outputs/` directories.
Scripts skip already-trained checkpoints and can be safely interrupted and resumed.

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.11.0
```

Install: `pip install torch numpy matplotlib scipy`
