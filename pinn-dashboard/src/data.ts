/**
 * data.ts
 * -------
 * All real metrics read from outputs/ JSON files.
 * Images are served as static assets from /public/outputs/ (symlinked or copied).
 * Numbers are hard-coded here from the actual JSON files — no fabrication.
 */

// ── Burgers' baseline (Phase 1) ──────────────────────────────────────────
export const BURGERS = {
  mse: 7.8181e-4,
  method: 'Deep Ensemble (10 members)',
}

// ── UQ Comparison (Phase 2) ──────────────────────────────────────────────
export const UQ = {
  ensemble: {
    method: 'Deep Ensemble',
    members: 10,
    mse: 7.8181e-4,
    ece: 0.0835,
    coverage90: 0.8855,
    trainMin: 89.4,
    inferSec: 0.1,
  },
  bayesian: {
    method: 'Bayesian PINN (VI)',
    mse: 8.9186e-2,
    ece: 0.0768,
    coverage90: 0.6874,
    trainMin: 10.0,
    inferSec: 4.5,
  },
  dropout: {
    method: 'MC Dropout (p=0.05)',
    mse: 9.3878e-3,
    ece: 0.1376,
    coverage90: 0.8389,
    trainMin: 34.5,
    inferSec: 19.4,
  },
}

// ── Ocean / Red Sea PINN (Phase 3) ───────────────────────────────────────
export const OCEAN = {
  mse: 7.0e-5,
  ece: 0.1018,
  coverage90: 0.7691,
  pde_loss: 3.37e-6,
  note: 'Linear PDE → convex loss → ensemble diversity reduced → slight under-coverage',
}

// ── Inverse Problem (Phase 4) ─────────────────────────────────────────────
export const INVERSE = {
  nu_true: 3.183e-3,
  // Robustness sweep — 9 conditions (noise × sensors)
  // Best case: noise=0.02, sensors=100 → error 17.6%
  // Worst case: noise=0.01, sensors=20 → error 183.6%
  robustness: [
    { noise: '0.5%', sensors: 20,  errPct: 78.5,  inCI: true },
    { noise: '0.5%', sensors: 50,  errPct: 106.2, inCI: false },
    { noise: '0.5%', sensors: 100, errPct: 94.5,  inCI: false },
    { noise: '1.0%', sensors: 20,  errPct: 183.6, inCI: false },
    { noise: '1.0%', sensors: 50,  errPct: 93.0,  inCI: false },
    { noise: '1.0%', sensors: 100, errPct: 28.3,  inCI: false },
    { noise: '2.0%', sensors: 20,  errPct: 240.8, inCI: false },
    { noise: '2.0%', sensors: 50,  errPct: 58.5,  inCI: false },
    { noise: '2.0%', sensors: 100, errPct: 17.6,  inCI: true },
  ],
}

// ── Neural Operator (Phase 5a) ────────────────────────────────────────────
export const FNO = {
  fno_rel_l2_mean: 0.06982,
  fno_rel_l2_std: 0.03335,
  pinn_rel_l2_mean: 0.32745,
  pinn_rel_l2_std: 0.06612,
  fno_infer_ms: 2.7,         // steady-state (excluding JIT warmup)
  pinn_infer_ms: 1.0,
  fno_train_s: 76,           // one-time cost
  pinn_train_s_per_ic: 21.6,
  breakeven_instances: 4,
  params: 562276,
}

// ── 2D Darcy Flow (Phase 5b) ──────────────────────────────────────────────
export const DARCY = {
  mse: 3.261e-10,
  rel_l2: 3.626e-5,          // 0.003626% ≈ 0.004%
  train_time_s: 364.1,       // 6.07 min
  n_epochs: 5000,
  n_col: 10000,
}

// ── Failure analysis (Phase 6) ────────────────────────────────────────────
export const FAILURE = [
  { nu: 3.183e-3, rel_l2: 4.56e-2,  label: '1/π×10⁻²' },
  { nu: 1.592e-3, rel_l2: 20.76e-2, label: '0.5×' },
  { nu: 6.366e-4, rel_l2: 33.27e-2, label: '0.2×' },
  { nu: 3.183e-4, rel_l2: 34.77e-2, label: '0.1×' },
]

// ── Ablation — ensemble size ──────────────────────────────────────────────
export const ABLATION_ENSEMBLE = [
  { M: 3,  ece: 0.0982, coverage90: 0.8107, mse: 4.151e-3 },
  { M: 5,  ece: 0.1261, coverage90: 0.9251, mse: 2.443e-3 },
  { M: 10, ece: 0.0710, coverage90: 0.9084, mse: 1.433e-3 },
  { M: 20, ece: 0.1337, coverage90: 0.9113, mse: 9.061e-4 },
]

// ── Ablation — loss weighting ─────────────────────────────────────────────
export const ABLATION_WEIGHTING = [
  { scheme: '(a) Baseline  λ_ic=10, λ_bc=10', mse: 6.658e-4, rel_l2: 4.21e-2 },
  { scheme: '(b) Uniform   λ_ic=1,  λ_bc=1',  mse: 1.685e-3, rel_l2: 6.70e-2 },
  { scheme: '(c) Auto-balanced (equal at init)', mse: 9.853e-2, rel_l2: 51.2e-2 },
]

// ── Hero stat strip ───────────────────────────────────────────────────────
export const STATS = [
  { value: '7',        label: 'Phases complete' },
  { value: '6.98%',   label: 'FNO rel-L2 error' },
  { value: '0.004%',  label: 'Darcy rel-L2 error' },
  { value: '9',       label: 'Robustness conditions' },
  { value: '10×',     label: 'Ensemble members' },
]

// ── Image paths (served from /public) ────────────────────────────────────
// Images are resolved relative to /public/outputs/ inside the pinn-dashboard
// The user should copy or symlink the outputs/ folder, or adjust paths here.
export function img(path: string): string {
  return `/outputs/${path}`
}
