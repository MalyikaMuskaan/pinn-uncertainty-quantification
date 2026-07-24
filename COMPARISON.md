# Uncertainty Quantification Comparison — Burgers' PINN

Comparison of three uncertainty quantification (UQ) methods applied to a
Physics-Informed Neural Network solving the viscous Burgers' equation
(ν = 0.01/π, x ∈ [−1, 1], t ∈ [0, 1]).

All results were produced by running `compare_methods.py` against models
trained independently by `run_ensemble.py`, `run_bayesian.py`, and
`run_dropout.py`.

---

## Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | **Deep Ensemble** | 10 independently initialised and trained PINNs; predictive mean and std are the sample mean/std across members at inference. |
| 2 | **Bayesian PINN (VI)** | Single PINN with mean-field variational inference (Bayes by Backprop); uncertainty comes from 200 Monte Carlo forward passes that draw fresh weight samples. |
| 3 | **MC Dropout** | Single PINN trained with dropout (p = 0.05); uncertainty comes from 100 stochastic forward passes with dropout active at inference. |

---

## Results Summary

### Quantitative metrics

| Method | MSE ↓ | ECE ↓ | 90 % Coverage | Std max | Train time | Inference time |
|--------|-------|-------|---------------|---------|-----------|----------------|
| Deep Ensemble (10 members) | **7.82 × 10⁻⁴** | **0.0835** | **0.8855** | 3.57 × 10⁻¹ | 89.4 min | **0.1 s** |
| Bayesian PINN (VI)         | 8.92 × 10⁻²    | 0.0860         | 0.6811        | 3.37 × 10⁻¹ | **10.0 min** | 3.0 s |
| MC Dropout (p = 0.05)      | 9.44 × 10⁻³    | 0.1374         | 0.8389        | 5.41 × 10⁻¹ | 34.5 min | 14.3 s |

*Metrics are computed against a finite-difference (FD) reference solution on a
256 × 100 (x, t) evaluation grid.*

**Key:**
- **MSE** — mean squared error between PINN predictive mean and the FD reference.
- **ECE** — Expected Calibration Error; lower is better (perfectly calibrated = 0).
- **90 % Coverage** — fraction of FD reference points that fall inside the 90 % predictive interval; ideal = 0.90.
- **Std max** — maximum standard deviation across the domain; a rough proxy for the dynamic range of the uncertainty estimate.
- **Train / Inference time** — wall-clock time on CPU.

---

## Per-method Analysis

### 1 — Deep Ensemble

**Best overall accuracy and calibration.**

- MSE is ~12× lower than MC Dropout and ~114× lower than Bayesian PINN,
  confirming that diverse random initialisations converge to genuinely different
  local minima and produce a rich predictive distribution.
- ECE (0.0835) is the lowest of the three, meaning the 90 % intervals actually
  contain ≈ 88.6 % of FD reference values — very close to the nominal level.
- Inference is near-instantaneous (0.1 s) because all 10 forward passes run in a
  single batched tensor operation with no weight resampling.
- **Downside:** training time is by far the longest (89 min), as 10 separate
  full training runs are required.

### 2 — Bayesian PINN (VI)

**Fastest to train; poorest predictive accuracy.**

- Training takes only 10 min (≈ 9× faster than Ensemble), because a single
  variational model is optimised end-to-end with an ELBO objective.
- ECE (0.0860) is competitive, but the 90 % empirical coverage is only 0.681 —
  significantly under-covering. The VI posterior is overconfident: predicted
  intervals are too tight relative to the actual error level.
- MSE is the worst of the three (~8.9 × 10⁻²), indicating that the variational
  approximation struggles to find the sharp shock layer solution as accurately as
  a deterministic ensemble member.
- Inference is moderate (3.0 s for 200 MC passes).

### 3 — MC Dropout

**Good accuracy; poorly calibrated uncertainty.**

- MSE (9.44 × 10⁻³) is 12× higher than Ensemble but 9× lower than Bayesian VI,
  placing it as a middle-ground in accuracy.
- ECE (0.1374) is the worst of the three. The empirical calibration curve shows
  that dropout uncertainty is systematically under-dispersed at lower confidence
  levels and over-dispersed at higher levels — a sign that the dropout rate
  (p = 0.05) was not tuned for calibration.
- 90 % coverage (0.839) is reasonable in absolute terms, but trails the Ensemble
  by ~5 percentage points.
- The highest `std_max` (0.541) indicates sporadic large uncertainty spikes —
  likely near the shock — not a smooth, well-calibrated spread.
- Inference is the slowest (14.3 s) because each of the 100 passes applies a
  different random dropout mask independently (no weight batching).

---

## Visual Outputs

All figures are saved under `burgers_pinn/outputs/comparison/`.

| File | Description |
|------|-------------|
| [`uncertainty_comparison.png`](burgers_pinn/outputs/comparison/uncertainty_comparison.png) | Three-panel log-scale std(u) heatmaps on a **shared** colour scale. The shared axis immediately reveals that Bayesian PINN and Ensemble produce similar uncertainty magnitudes, while MC Dropout concentrates its uncertainty near the shock front. |
| [`calibration_comparison.png`](burgers_pinn/outputs/comparison/calibration_comparison.png) | All three calibration curves (nominal confidence vs. empirical coverage) overlaid on one axes with the perfect-calibration diagonal for reference. |
| [`comparison_table.csv`](burgers_pinn/outputs/comparison/comparison_table.csv) | Machine-readable table of all quantitative metrics. |

---

## Recommendations

| Objective | Recommended method |
|-----------|--------------------|
| Highest accuracy | Deep Ensemble |
| Best calibration | Deep Ensemble |
| Fastest training (budget-constrained) | Bayesian PINN |
| Minimal model complexity / single checkpoint | MC Dropout |

**For production use on Burgers-type PDE problems, the Deep Ensemble is the
clear winner on every accuracy and calibration metric.** The 89-minute training
cost is the only barrier; it can be reduced by lowering the number of members
(5 members typically retains ~90 % of the accuracy gain) or by using GPU
hardware (not available in this run).

If training budget is the primary constraint and a rough uncertainty estimate is
acceptable, **Bayesian PINN** offers the best accuracy-per-training-minute trade-
off. Its ECE is competitive, though the under-covering 90 % interval requires a
post-hoc temperature or scale calibration step before use in safety-critical
settings.

**MC Dropout** requires the least code change (add `nn.Dropout` layers and keep
them active at inference), but its ECE is poorest here and its inference is the
slowest of the three — making it the least attractive option for this problem.

---

## Reproducibility

```bash
cd burgers_pinn

# (Re-)train all three methods
python run_ensemble.py   # ~89 min on CPU
python run_bayesian.py   # ~10 min on CPU
python run_dropout.py    # ~35 min on CPU

# Regenerate comparison artefacts
python compare_methods.py
```

All trained checkpoints are committed under `burgers_pinn/outputs/`.
The comparison script reads the checkpoints and metric JSON files directly;
it does **not** re-train any model.

---

*Generated by `compare_methods.py` — results reflect the trained model state at
the time of this run.*
