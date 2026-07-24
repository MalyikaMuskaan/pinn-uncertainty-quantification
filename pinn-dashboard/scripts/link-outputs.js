/**
 * scripts/link-outputs.js
 * -----------------------
 * Makes all project PNG outputs available to the Vite dev server at /outputs/...
 *
 * NOTE: You no longer need to run this script!
 * The Vite config (vite.config.ts) now includes a custom middleware plugin that
 * intercepts GET /outputs/<path> requests and streams the file directly from
 * the project root — no symlinks, no copies, no script needed.
 *
 * This script is kept as a FALLBACK for production builds, which don't use
 * the dev-server middleware.  For `npm run build` + `npm run preview`, run:
 *
 *   node scripts/link-outputs.js
 *
 * It will attempt a directory junction (Windows) or symlink (macOS/Linux),
 * falling back to copying if that fails.  It prints a full diagnostic report.
 */

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const dashboardRoot = path.resolve(__dirname, '..')
const projectRoot   = path.resolve(dashboardRoot, '..')
const publicOutputs = path.resolve(dashboardRoot, 'public', 'outputs')

// ── All image paths the app requests (relative to projectRoot) ─────────────
const EXPECTED_IMAGES = [
  'burgers_pinn/outputs/heatmap.png',
  'burgers_pinn/outputs/time_slices.png',
  'burgers_pinn/outputs/loss_history.png',
  'burgers_pinn/outputs/ensemble/ensemble_mean_heatmap.png',
  'burgers_pinn/outputs/ensemble/ensemble_std_heatmap.png',
  'burgers_pinn/outputs/ensemble/ensemble_time_slices.png',
  'burgers_pinn/outputs/ensemble/ensemble_calibration.png',
  'burgers_pinn/outputs/bayesian/bayesian_mean_heatmap.png',
  'burgers_pinn/outputs/bayesian/bayesian_std_heatmap.png',
  'burgers_pinn/outputs/bayesian/bayesian_time_slices.png',
  'burgers_pinn/outputs/bayesian/bayesian_calibration.png',
  'burgers_pinn/outputs/dropout/dropout_mean_heatmap.png',
  'burgers_pinn/outputs/dropout/dropout_std_heatmap.png',
  'burgers_pinn/outputs/dropout/dropout_time_slices.png',
  'burgers_pinn/outputs/dropout/dropout_calibration.png',
  'burgers_pinn/outputs/comparison/calibration_comparison.png',
  'burgers_pinn/outputs/comparison/uncertainty_comparison.png',
  'burgers_pinn/outputs/failure_analysis/failure_error_vs_nu.png',
  'burgers_pinn/outputs/failure_analysis/failure_heatmap_comparison.png',
  'burgers_pinn/outputs/ablation/ensemble_size/ablation_ensemble_size.png',
  'burgers_pinn/outputs/ablation/loss_weighting/ablation_loss_weighting.png',
  'ocean_pinn/outputs/heatmap.png',
  'ocean_pinn/outputs/time_slices.png',
  'ocean_pinn/outputs/loss_history.png',
  'ocean_pinn/outputs/ensemble/ensemble_mean_heatmap.png',
  'ocean_pinn/outputs/ensemble/ensemble_std_heatmap.png',
  'ocean_pinn/outputs/ensemble/ensemble_time_slices.png',
  'ocean_pinn/outputs/ensemble/ensemble_calibration.png',
  'inverse_problem/outputs/nu_convergence.png',
  'inverse_problem/outputs/solution_comparison.png',
  'inverse_problem/outputs/robustness/robustness_error_vs_sensors.png',
  'inverse_problem/outputs/robustness/robustness_nu_estimates.png',
  'neural_operator/outputs/plots/comparison_000.png',
  'neural_operator/outputs/plots/comparison_001.png',
  'neural_operator/outputs/plots/comparison_002.png',
  'neural_operator/outputs/plots/summary_table.png',
  'darcy_2d/outputs/solution_comparison.png',
  'darcy_2d/outputs/loss_history.png',
  'darcy_2d/outputs/pde_residual_map.png',
]

// ── Diagnostic pass — check which source files actually exist ──────────────
console.log('\n📂 Scanning source files in project root…\n')
const found = []
const missing = []

for (const rel of EXPECTED_IMAGES) {
  const srcPath = path.join(projectRoot, rel)
  if (fs.existsSync(srcPath)) {
    found.push(rel)
  } else {
    missing.push(rel)
  }
}

console.log(`  ✓ Found:   ${found.length} / ${EXPECTED_IMAGES.length}`)
if (missing.length > 0) {
  console.log(`  ✗ Missing: ${missing.length}`)
  for (const m of missing) console.log(`      ✗  ${m}`)
}
console.log('')

// ── Try junction / symlink first ───────────────────────────────────────────
if (fs.existsSync(publicOutputs)) {
  const stat = fs.lstatSync(publicOutputs)
  if (stat.isSymbolicLink() || (stat.isDirectory() && !stat.isFile())) {
    console.log('✓ public/outputs already exists — skipping link step.')
    printSummary()
    process.exit(0)
  }
}

let linked = false
try {
  // 'junction' works on Windows without elevated privileges
  fs.symlinkSync(projectRoot, publicOutputs, 'junction')
  console.log(`✓ Junction created: public/outputs → ${projectRoot}`)
  linked = true
} catch (e) {
  console.warn(`  ⚠ Junction failed (${e.message}) — falling back to file copy…`)
}

// ── Copy fallback ──────────────────────────────────────────────────────────
if (!linked) {
  let copied = 0
  let failed = 0
  for (const rel of found) {
    const src  = path.join(projectRoot, rel)
    const dest = path.join(publicOutputs, rel)
    try {
      fs.mkdirSync(path.dirname(dest), { recursive: true })
      fs.copyFileSync(src, dest)
      copied++
    } catch (err) {
      console.error(`  ✗ Failed to copy ${rel}: ${err.message}`)
      failed++
    }
  }
  console.log(`\n  Copied ${copied} files (${failed} failed)`)
}

printSummary()

function printSummary() {
  console.log('\n─────────────────────────────────────────────────')
  console.log('  DIAGNOSTIC SUMMARY')
  console.log('─────────────────────────────────────────────────')
  console.log(`  Project root:   ${projectRoot}`)
  console.log(`  public/outputs: ${publicOutputs}`)
  console.log(`  Source found:   ${found.length} / ${EXPECTED_IMAGES.length}`)
  if (missing.length > 0) {
    console.log(`\n  These files are missing from project outputs/`)
    console.log(`  (they will show the ⚠ placeholder in the dashboard):`)
    for (const m of missing) console.log(`    ✗  ${m}`)
  } else {
    console.log(`  All expected source images exist ✓`)
  }
  console.log('─────────────────────────────────────────────────\n')
}
