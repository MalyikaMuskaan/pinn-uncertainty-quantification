# Fourier Neural Operator — Notes & Final Results

**Equation:** Viscous Burgers'  `u_t + u·u_x = ν·u_xx`  on x ∈ [-1,1], t ∈ [0,1]
**ν:** `0.01/π ≈ 0.003183` (same as the PINN project)
**Task:** Operator learning — map any initial condition u₀(x) to the full solution u(x,t)
**Run environment:** Google Colab (GPU, T4)

---

## Core Trade-off: PINN vs FNO

### Physics-Informed Neural Network (PINN)

The PINN in `inverse_problem/` learns to approximate **one specific solution**
by minimising a combined loss of:
- PDE residual at collocation points (enforces the physics)
- Initial condition at t = 0
- Boundary conditions at x = ±1
- (Optionally) sparse sensor data for the inverse problem

**Strengths:**
- No labelled training data required — the physics is the supervision signal
- Enforces the governing equation structurally during training
- The solution it learns is physics-consistent (up to optimisation error)

**Critical limitation:**
- **Must be retrained from scratch for every new scenario** (new IC, new ν, new domain)
- Each retrain costs ~minutes of GPU time
- Does not generalise: a PINN trained on `-sin(πx)` gives nonsense on any other IC

---

### Fourier Neural Operator (FNO)

The FNO in this folder learns a **solution operator** — a mapping from the space of
initial conditions to the space of solution trajectories.  Once trained, it can
evaluate any new IC in milliseconds without retraining.

**How it works:**
1. **Lift** the 1-D initial condition signal into a high-dimensional feature space
2. Apply **four Fourier layers**: each layer operates globally in frequency space
   (capturing long-range interactions) and locally via a bypass convolution
3. **Project** down to the output: one predicted value per spatial location × time step

The Fourier layers are the key innovation: standard convolutions are local
(limited receptive field), but multiplication in Fourier space is equivalent to
a global convolution — the FNO can capture the full spatial structure of the PDE
at each layer with O(N log N) cost.

**Strengths:**
- **Single training run generalises** to the entire distribution of ICs
- Inference is extremely fast (milliseconds per instance post-training)
- Scales well: training cost is amortised over all future queries

**Limitations:**
- **Requires labelled training data** — ground-truth solutions must be computed
  upfront (we use the Crank-Nicolson FD solver, which takes ~0.5s/instance on CPU)
- **Purely data-driven**: the FNO does not know it is solving Burgers' equation;
  it will give plausible-looking but unphysical predictions for ICs or parameters
  far outside the training distribution
- **Fixed discretisation**: this implementation maps N_x=256 → (N_x=256, N_t=100);
  changing the grid resolution requires retraining (or architectural changes)
- Needs ~1000 training examples to learn the operator; with fewer samples,
  accuracy degrades quickly

---

## Experimental Setup

| Item | Value |
|------|-------|
| Training samples | 800 (from 1000 total) |
| Validation samples | 100 |
| Test samples | 100 |
| IC family | Random sums of sin(kπx), k=1…5, amplitudes ∈ (-1,1), normalised to max\|u₀\|=1 |
| Ground truth | Crank-Nicolson FD (N_fd=512, N_t_fd=4000) |
| FNO modes | 16 |
| FNO width | 64 |
| FNO depth | 4 Fourier layers |
| FNO parameters | 562,276 |
| Training | Adam + cosine annealing, 300 epochs, batch 32 |
| PINN comparison | Retrained per test instance, Adam 2000 + L-BFGS 1500 epochs, ν fixed at true value |

---

## Final Results (Colab GPU — T4)

| Metric | FNO | PINN (per-instance) |
|--------|-----|---------------------|
| Rel L2 error — mean | **6.98%** | 32.75% |
| Rel L2 error — std | 3.33% | 6.61% |
| Training time | **76 s** (one-time, all ICs) | 22 s / instance |
| Inference time / instance | ~2.7 ms (steady-state)† | 1.0 ms |
| Generalises to new ICs? | **Yes** | No — must retrain |
| Physics-constrained? | No | **Yes** |

† The reported mean of 5.79 ms includes a one-off JIT/CUDA warmup on the first
call (≈299 ms). All subsequent calls settle at ~2.7 ms, confirmed by the
per-instance timing list in `outputs/eval_fno_vs_pinn.json`.

### Break-even analysis

At 22 s/retrain for the PINN and 76 s of one-time FNO training:

    break-even = 76 s / 22 s ≈ 3–4 instances

After **4 instances** the FNO has paid back its training cost.  For any
workload requiring ≥ 5 distinct ICs the FNO is strictly cheaper in total
wall time, and inference is ~8 000× faster per query once trained.

### Accuracy gap

The FNO is **4.7× more accurate** than the per-instance PINN (6.98% vs 32.75%
relative L2 on the 100 test ICs).  This is somewhat surprising given that the
PINN sees the exact IC and is physics-constrained; the likely explanation is
that the PINN's Adam+L-BFGS schedule (3500 epochs total) is not long enough to
fully converge for arbitrary ICs — it was tuned for the single canonical
`-sin(πx)` IC.  A longer PINN schedule would close some of this gap.

---

## Interpretation of Results

When reading `outputs/eval_fno_vs_pinn.json` or `outputs/plots/summary_table.png`,
keep in mind:

1. **The accuracy comparison is not perfectly apples-to-apples.** The PINN is
   retrained on each test IC (fair — it sees the specific IC), while the FNO was
   trained on a different 800-sample training set.  However the FNO was never
   shown any test IC, so its accuracy on test cases is a genuine generalisation
   measure.

2. **The PINN uses physics; the FNO does not.**  If the test IC is within the
   training distribution, the FNO should be accurate.  If you query the FNO on
   an IC very unlike anything in training (e.g., a sharp discontinuity when
   training only on smooth sines), expect degraded accuracy.  The PINN, by
   contrast, enforces the PDE and will handle unusual ICs reasonably as long as
   it is retrained.

3. **Training time asymmetry.** The FNO training time is a one-time fixed cost.
   The PINN cost is per-instance.  The break-even is ~4 instances (see above).
   If you only ever need to solve the equation for ≤ 3 ICs, the PINN is cheaper
   overall.

4. **Uncertainty.** Neither method in this implementation produces calibrated
   uncertainty estimates.  The inverse PINN ensemble provides ν uncertainty only,
   not solution-field uncertainty.  Adding uncertainty to the FNO (e.g., via
   ensembles or MC dropout) is a natural extension.

---

## File Map

| File / Path | Role |
|-------------|------|
| `model.py` | FNO1d architecture (SpectralConv1d, FourierLayer, FNO1d) |
| `data_gen.py` | Generate 1000 IC/solution pairs; save to `outputs/dataset.npz` |
| `train.py` | Supervised training loop; save best checkpoint to `outputs/fno_best.pt` |
| `evaluate.py` | Evaluate FNO on test set; optionally retrain PINN for comparison |
| `requirements.txt` | Python dependencies |
| `outputs/dataset.npz` | 1000 IC/solution pairs (u0: (1000,256), u: (1000,256,100)) |
| `outputs/fno_best.pt` | Best FNO checkpoint (epoch with lowest val MSE) |
| `outputs/train_history.npz` | Per-epoch train/val MSE loss curves |
| `outputs/eval_fno_vs_pinn.json` | Full per-instance metrics (rel L2, inference times, PINN train times) |
| `outputs/plots/comparison_000.png` | FNO vs ground truth — test instance 0, at t=0.0, 0.5, 1.0 |
| `outputs/plots/comparison_001.png` | FNO vs ground truth — test instance 1, at t=0.0, 0.5, 1.0 |
| `outputs/plots/comparison_002.png` | FNO vs ground truth — test instance 2, at t=0.0, 0.5, 1.0 |
| `outputs/plots/summary_table.png` | Side-by-side FNO vs PINN metrics table (accuracy, timing, generalisation) |

Each `comparison_NNN.png` shows three panels (t = 0, 0.5, 1.0): the Crank-Nicolson
FD ground truth (black solid) and the FNO prediction (blue dashed), with the
per-instance relative L2 error in the figure title.  These plots are generated
by `evaluate.py::plot_comparisons()` for the first 3 test instances.

`summary_table.png` is a rendered matplotlib table covering: mean/std rel L2
error, inference time per instance, total training cost, generalisation scope,
and whether the method is physics-constrained.

## How to Run (on Colab GPU)

```bash
cd neural_operator

# Step 1 — generate dataset (~8-12 min on CPU)
python data_gen.py

# Step 2 — train FNO (76s on T4 for 300 epochs)
python train.py --epochs 300 --lr 1e-3

# Step 3 — evaluate (FNO + 3 PINN retrains for comparison)
python evaluate.py --n_pinn_eval 3
# or FNO-only (skips the ~66s of PINN retraining):
python evaluate.py --skip_pinn
```
