"""
data.py
-------
Centralised, cached data loaders for all project output files.

All loaders use @st.cache_data so Streamlit only reads each file once per
session.  Every function returns None (never raises) if the file is missing,
so individual pages can display a friendly "data not yet generated" message.

ROOT is resolved relative to this file's location: dashboard/ lives one level
below the project root (d:/pnn/), so ROOT = Path(__file__).parent.parent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Project root (one directory above dashboard/) ─────────────────────────
ROOT = Path(__file__).parent.parent


# ── Generic helpers ────────────────────────────────────────────────────────

def _p(*parts: str) -> Path:
    """Build an absolute path relative to ROOT."""
    return ROOT.joinpath(*parts)


def _load_json(path: Path) -> Any | None:
    """Load a JSON file; return None if missing or malformed."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_image(path: Path) -> Image.Image | None:
    """Load a PIL Image; return None if missing."""
    try:
        return Image.open(path)
    except (FileNotFoundError, Exception):
        return None


def _load_csv(path: Path) -> pd.DataFrame | None:
    """Load a CSV as a DataFrame; return None if missing."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, Exception):
        return None


# ── Phase 1 — Burgers' baseline ────────────────────────────────────────────

@st.cache_data
def load_burgers_heatmap() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "heatmap.png"))

@st.cache_data
def load_burgers_time_slices() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "time_slices.png"))

@st.cache_data
def load_burgers_loss_history() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "loss_history.png"))


# ── Phase 2 — UQ comparison ────────────────────────────────────────────────

@st.cache_data
def load_ensemble_metrics() -> dict | None:
    return _load_json(_p("burgers_pinn", "outputs", "ensemble", "ensemble_metrics.json"))

@st.cache_data
def load_bayesian_metrics() -> dict | None:
    return _load_json(_p("burgers_pinn", "outputs", "bayesian", "bayesian_metrics.json"))

@st.cache_data
def load_dropout_metrics() -> dict | None:
    return _load_json(_p("burgers_pinn", "outputs", "dropout", "dropout_metrics.json"))

@st.cache_data
def load_comparison_table() -> pd.DataFrame | None:
    return _load_csv(_p("burgers_pinn", "outputs", "comparison", "comparison_table.csv"))

@st.cache_data
def load_ensemble_calibration_img() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "ensemble", "ensemble_calibration.png"))

@st.cache_data
def load_bayesian_calibration_img() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "bayesian", "bayesian_calibration.png"))

@st.cache_data
def load_dropout_calibration_img() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "dropout", "dropout_calibration.png"))

@st.cache_data
def load_uncertainty_comparison_img() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "comparison", "uncertainty_comparison.png"))

@st.cache_data
def load_calibration_comparison_img() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "comparison", "calibration_comparison.png"))

@st.cache_data
def load_ensemble_mean_heatmap() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "ensemble", "ensemble_mean_heatmap.png"))

@st.cache_data
def load_ensemble_std_heatmap() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "ensemble", "ensemble_std_heatmap.png"))

@st.cache_data
def load_ensemble_time_slices_img() -> Image.Image | None:
    return _load_image(_p("burgers_pinn", "outputs", "ensemble", "ensemble_time_slices.png"))


# ── Phase 3 — Ocean PINN ───────────────────────────────────────────────────

@st.cache_data
def load_ocean_heatmap() -> Image.Image | None:
    return _load_image(_p("ocean_pinn", "outputs", "heatmap.png"))

@st.cache_data
def load_ocean_ensemble_mean() -> Image.Image | None:
    return _load_image(_p("ocean_pinn", "outputs", "ensemble", "ensemble_mean_heatmap.png"))

@st.cache_data
def load_ocean_ensemble_std() -> Image.Image | None:
    return _load_image(_p("ocean_pinn", "outputs", "ensemble", "ensemble_std_heatmap.png"))

@st.cache_data
def load_ocean_ensemble_time_slices() -> Image.Image | None:
    return _load_image(_p("ocean_pinn", "outputs", "ensemble", "ensemble_time_slices.png"))

@st.cache_data
def load_ocean_calibration() -> Image.Image | None:
    return _load_image(_p("ocean_pinn", "outputs", "ensemble", "ensemble_calibration.png"))


# ── Phase 4 — Inverse problem ──────────────────────────────────────────────

@st.cache_data
def load_inverse_metrics() -> dict | None:
    return _load_json(_p("inverse_problem", "outputs", "metrics.json"))

@st.cache_data
def load_nu_convergence_img() -> Image.Image | None:
    return _load_image(_p("inverse_problem", "outputs", "nu_convergence.png"))

@st.cache_data
def load_robustness_summary() -> list | None:
    # Prefer the richer robustness/summary.json (9-condition sweep)
    data = _load_json(_p("inverse_problem", "outputs", "robustness", "summary.json"))
    if data is not None:
        return data
    # Fall back to the embedded robustness array inside metrics.json
    m = load_inverse_metrics()
    if m and "robustness" in m:
        return m["robustness"]
    return None

@st.cache_data
def load_robustness_error_img() -> Image.Image | None:
    return _load_image(
        _p("inverse_problem", "outputs", "robustness", "robustness_error_vs_sensors.png"))

@st.cache_data
def load_robustness_nu_img() -> Image.Image | None:
    return _load_image(
        _p("inverse_problem", "outputs", "robustness", "robustness_nu_estimates.png"))


# ── Phase 5a — Neural Operator ─────────────────────────────────────────────

@st.cache_data
def load_fno_metrics() -> dict | None:
    return _load_json(_p("neural_operator", "outputs", "eval_fno_vs_pinn.json"))

@st.cache_data
def load_fno_comparison_imgs() -> list[Image.Image | None]:
    return [
        _load_image(_p("neural_operator", "outputs", "plots", f"comparison_{i:03d}.png"))
        for i in range(3)
    ]

@st.cache_data
def load_fno_summary_table_img() -> Image.Image | None:
    return _load_image(_p("neural_operator", "outputs", "plots", "summary_table.png"))


# ── Phase 5b — 2D Darcy ────────────────────────────────────────────────────

@st.cache_data
def load_darcy_metrics() -> dict | None:
    return _load_json(_p("darcy_2d", "outputs", "metrics.json"))

@st.cache_data
def load_darcy_solution_img() -> Image.Image | None:
    return _load_image(_p("darcy_2d", "outputs", "solution_comparison.png"))

@st.cache_data
def load_darcy_loss_img() -> Image.Image | None:
    return _load_image(_p("darcy_2d", "outputs", "loss_history.png"))

@st.cache_data
def load_darcy_residual_img() -> Image.Image | None:
    return _load_image(_p("darcy_2d", "outputs", "pde_residual_map.png"))


# ── Phase 6 — Failure analysis & ablation ─────────────────────────────────

@st.cache_data
def load_failure_metrics() -> list | None:
    return _load_json(
        _p("burgers_pinn", "outputs", "failure_analysis", "metrics.json"))

@st.cache_data
def load_failure_error_img() -> Image.Image | None:
    return _load_image(
        _p("burgers_pinn", "outputs", "failure_analysis", "failure_error_vs_nu.png"))

@st.cache_data
def load_failure_heatmap_img() -> Image.Image | None:
    return _load_image(
        _p("burgers_pinn", "outputs", "failure_analysis", "failure_heatmap_comparison.png"))

@st.cache_data
def load_ablation_ensemble_metrics() -> list | None:
    return _load_json(
        _p("burgers_pinn", "outputs", "ablation", "ensemble_size",
           "ensemble_size_metrics.json"))

@st.cache_data
def load_ablation_ensemble_img() -> Image.Image | None:
    return _load_image(
        _p("burgers_pinn", "outputs", "ablation", "ensemble_size",
           "ablation_ensemble_size.png"))

@st.cache_data
def load_ablation_weighting_metrics() -> list | None:
    return _load_json(
        _p("burgers_pinn", "outputs", "ablation", "loss_weighting",
           "loss_weighting_metrics.json"))

@st.cache_data
def load_ablation_weighting_img() -> Image.Image | None:
    return _load_image(
        _p("burgers_pinn", "outputs", "ablation", "loss_weighting",
           "ablation_loss_weighting.png"))
