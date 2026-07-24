# PINN + UQ — Research Portfolio Dashboard

Dark, cinematic "liquid glass" research portfolio built with **React + Vite + TypeScript + Tailwind CSS + Framer Motion + Three.js**.

---

## Prerequisites

- Node.js ≥ 18  
- npm ≥ 9

---

## Setup

### 1. Install dependencies

```bash
cd pinn-dashboard
npm install
```

### 2. Link the output images

The dashboard loads plot images from `/outputs/…` relative to the Vite dev server.
The easiest way to make this work is to create a symlink (or copy) inside `public/`:

**Option A — symlink (recommended, zero disk cost)**

```bash
# From inside pinn-dashboard/
# Windows PowerShell (run as Administrator or with Developer Mode enabled):
New-Item -ItemType SymbolicLink -Path ".\public\outputs" -Target "..\outputs_for_dashboard"
```

Or use the helper script already provided:

```bash
node scripts/link-outputs.js
```

**Option B — run the copy script**

```bash
node scripts/copy-outputs.js
```

**Option C — manual structure**

Create `pinn-dashboard/public/outputs/` and populate it with:

```
public/outputs/
├── burgers_pinn/outputs/
│   ├── heatmap.png
│   ├── time_slices.png
│   ├── loss_history.png
│   ├── ensemble/
│   │   ├── ensemble_mean_heatmap.png
│   │   ├── ensemble_std_heatmap.png
│   │   ├── ensemble_time_slices.png
│   │   └── ensemble_calibration.png
│   ├── bayesian/
│   │   ├── bayesian_mean_heatmap.png
│   │   ├── bayesian_calibration.png
│   │   └── …
│   ├── dropout/
│   │   └── dropout_calibration.png
│   ├── comparison/
│   │   ├── calibration_comparison.png
│   │   └── uncertainty_comparison.png
│   ├── failure_analysis/
│   │   ├── failure_error_vs_nu.png
│   │   └── failure_heatmap_comparison.png
│   └── ablation/
│       ├── ensemble_size/ablation_ensemble_size.png
│       └── loss_weighting/ablation_loss_weighting.png
├── ocean_pinn/outputs/
│   ├── heatmap.png
│   └── ensemble/…
├── inverse_problem/outputs/
│   ├── nu_convergence.png
│   └── robustness/robustness_error_vs_sensors.png
├── neural_operator/outputs/plots/
│   ├── comparison_000.png … comparison_002.png
│   └── summary_table.png
└── darcy_2d/outputs/
    ├── solution_comparison.png
    ├── loss_history.png
    └── pde_residual_map.png
```

> Images not found render gracefully as a dashed "not yet generated" placeholder — the dashboard will not crash.

---

### 3. Run the dev server

```bash
# From inside pinn-dashboard/
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

### 4. Production build

```bash
npm run build
npm run preview   # serves the dist/ folder locally
```

---

## Project structure

```
pinn-dashboard/
├── index.html
├── package.json
├── tailwind.config.js
├── vite.config.ts
├── tsconfig.json
├── public/
│   ├── favicon.svg
│   └── outputs/          ← symlink or copy of your outputs/
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── index.css          ← global styles, .glass, .shiny-text, etc.
    ├── data.ts            ← all real metrics hard-coded from JSON files
    └── components/
        ├── OceanBackground.tsx   ← Three.js animated canvas
        ├── Nav.tsx
        ├── Hero.tsx
        ├── Section.tsx           ← scroll-triggered fade-in wrapper
        ├── UI.tsx                ← MetricCard, PlotImage, SectionDivider
        ├── SectionBurgers.tsx
        ├── SectionUQ.tsx
        ├── SectionOcean.tsx
        ├── SectionInverse.tsx
        ├── SectionNeuralOperator.tsx
        ├── SectionDarcy.tsx
        ├── SectionAblations.tsx
        └── Footer.tsx
```

---

## Design system

| Token | Value |
|---|---|
| Background | `#050403` (near-black, warm red-black) |
| Accent | `#ff8f6b` (red-orange, Red Sea tint) |
| Glass card | `rgba(255,255,255,0.025)` + `backdrop-filter: blur(6px)` + border gradient mask |
| Heading font | Instrument Serif (Google Fonts) |
| Body font | Inter |
| Animations | Framer Motion scroll-triggered `fadeUp`, shiny text gradient loop |
| 3-D background | Three.js: drifting particles, undulating floor, pulsing core light |
