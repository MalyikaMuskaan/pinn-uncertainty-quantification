"""pages/home.py — Project overview and cross-phase stat grid."""
import streamlit as st
import pandas as pd
from data import (
    load_ensemble_metrics, load_bayesian_metrics, load_dropout_metrics,
    load_fno_metrics, load_darcy_metrics, load_failure_metrics,
    load_ablation_ensemble_metrics, load_ablation_weighting_metrics,
)
from styles import stat_badge, phase_pill, missing


def render() -> None:
    st.markdown(
        '<h1 style="text-align:center;margin-bottom:4px;">🌊 Deep PINNs</h1>'
        '<p style="text-align:center;color:#b09098;font-size:1.05rem;margin-bottom:30px;">'
        'Physics-Informed Neural Networks &amp; Uncertainty Quantification '
        '— KAUST PhD Application Research Portfolio'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Summary stat row ──────────────────────────────────────────────────
    ens  = load_ensemble_metrics()
    fno  = load_fno_metrics()
    drcy = load_darcy_metrics()
    fail = load_failure_metrics()
    abls = load_ablation_ensemble_metrics()

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        val = f"{ens['mse']:.2e}" if ens else "—"
        st.markdown(stat_badge(val, "Ensemble MSE"), unsafe_allow_html=True)
    with col2:
        val = f"{ens['ece']:.3f}" if ens else "—"
        st.markdown(stat_badge(val, "Ensemble ECE"), unsafe_allow_html=True)
    with col3:
        val = f"{fno['fno']['rel_l2_mean']*100:.1f}%" if fno else "—"
        st.markdown(stat_badge(val, "FNO Rel-L2"), unsafe_allow_html=True)
    with col4:
        val = f"{drcy['rel_l2']*100:.4f}%" if drcy else "—"
        st.markdown(stat_badge(val, "Darcy Rel-L2"), unsafe_allow_html=True)
    with col5:
        val = f"{fail[0]['rel_l2']*100:.1f}%" if fail else "—"
        st.markdown(stat_badge(val, "PINN (baseline ν)"), unsafe_allow_html=True)
    with col6:
        m10 = next((r for r in abls if r["M"] == 10), None) if abls else None
        val = f"{m10['ece']:.3f}" if m10 else "—"
        st.markdown(stat_badge(val, "Ens ECE (M=10)"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Phase cards ───────────────────────────────────────────────────────
    phases = [
        ("1", "Burgers' PINN Baseline",
         "Viscous Burgers' 1D solved with a 4×50 tanh PINN validated against Crank-Nicolson FD.",
         f"MSE {ens['mse']:.2e}" if ens else "—"),
        ("2", "UQ Method Comparison",
         "Deep Ensemble vs Bayesian PINN vs MC Dropout — ECE, coverage, MSE on the same problem.",
         f"Ensemble ECE {ens['ece']:.3f}" if ens else "—"),
        ("3", "Ocean / Climate PDE",
         "1D advection-diffusion: linear PDE converges 10× faster, but calibration is harder.",
         "MSE 7.0×10⁻⁵"),
        ("4", "Inverse Problem",
         "Recover unknown ν from sparse noisy sensors. Fixed lambda_data bug: error 400→17%.",
         "Best error 17.6%"),
        ("5a", "Fourier Neural Operator",
         "FNO learns the solution operator over all ICs. 4× faster than per-instance PINN retrain.",
         f"FNO {fno['fno']['rel_l2_mean']*100:.1f}% vs PINN {fno['pinn']['rel_l2_mean']*100:.1f}%" if fno else "—"),
        ("5b", "2D Darcy Flow",
         "First 2D elliptic PINN. MMS exact solution, Adam + L-BFGS, sub-machine-epsilon accuracy.",
         f"Rel-L2 {drcy['rel_l2']*100:.4f}%" if drcy else "—"),
        ("6", "Failure Analysis & Ablation",
         "ν sweep shows PINN fails at sharp shocks. ECE non-monotonic in M. Auto-balance fails forward PINNs.",
         f"Baseline {fail[0]['rel_l2']*100:.1f}% → worst {fail[-1]['rel_l2']*100:.1f}%" if fail else "—"),
    ]

    for pill_label, title, desc, metric in phases:
        st.markdown(
            f'<div class="card">'
            f'{phase_pill("Phase " + pill_label)}'
            f'<strong style="color:#f0e8ec;font-size:1.05rem;"> {title}</strong>'
            f'<p style="color:#b09098;margin:6px 0 4px;">{desc}</p>'
            f'<span style="color:#ff3355;font-size:0.9rem;font-weight:600;">{metric}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Navigation hint ───────────────────────────────────────────────────
    st.markdown(
        '<div class="callout">👈 Use the sidebar to navigate between phases.</div>',
        unsafe_allow_html=True,
    )
