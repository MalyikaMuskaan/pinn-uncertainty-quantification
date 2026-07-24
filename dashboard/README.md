# Deep PINNs Dashboard

Interactive Streamlit dashboard for the PINN + Uncertainty Quantification
research portfolio. Covers all 7 phases: Burgers' baseline, UQ comparison,
Ocean PDE, inverse problem, FNO operator learning, 2D Darcy, and validation.

## Requirements

```
pip install streamlit plotly pandas numpy Pillow
```

Or install from the bundled requirements file:

```
pip install -r dashboard/requirements.txt
```

## Run

From the **project root** (`d:/pnn/`):

```bash
streamlit run dashboard/app.py
```

> ⚠️ Must be run from the project root so that relative paths to
> `burgers_pinn/outputs/`, `darcy_2d/outputs/`, etc. resolve correctly.

The app opens at `http://localhost:8501` in your browser.

## Navigation

Use the left sidebar to switch between sections:

| Section | Content |
|---------|---------|
| 🏠 Home / Overview | Stat grid across all phases |
| 📈 Burgers' PINN | Live inference slider + saved heatmaps |
| 🎯 UQ Comparison | Calibration diagrams + metrics cards |
| 🌊 Ocean PDE | Advection-diffusion solution + uncertainty |
| 🔍 Inverse Problem | ν recovery robustness sweep table |
| ⚡ Neural Operator | FNO vs PINN error distribution + break-even |
| 🟥 2D Darcy Flow | 3-panel solution heatmap + loss history |
| 🔬 Failure & Ablation | ν sweep degradation + ensemble/weighting ablations |
| ℹ️ About & Methods | Cross-project findings + references |

## Background

The animated Three.js Red Sea background (caustic light rays + floating
particles) is injected via `st.components.v1.html()` with `pointer-events:none`
so it never interferes with Streamlit widgets. The canvas is `position:fixed`
at `z-index:0`; all Streamlit content sits at `z-index:1` above it.

## File structure

```
dashboard/
├── app.py                  Main entry point
├── background.py           Three.js Red Sea HTML generator
├── styles.py               CSS injector (glass-card theme)
├── data.py                 Cached data loaders for all output files
├── requirements.txt        Python dependencies
├── README.md               This file
└── pages/
    ├── home.py             Overview + stat grid
    ├── burgers.py          Burgers' solution viewer
    ├── uq_comparison.py    UQ method comparison
    ├── ocean.py            Ocean PDE viewer
    ├── inverse.py          Inverse problem
    ├── neural_operator.py  FNO vs PINN
    ├── darcy.py            2D Darcy flow
    ├── validation.py       Failure analysis + ablation
    └── about.py            Methodology + references
```

## Missing data

If a training output file is not present, the dashboard displays a styled
"Data not yet generated" placeholder rather than crashing. Run the relevant
training script to populate the outputs.
