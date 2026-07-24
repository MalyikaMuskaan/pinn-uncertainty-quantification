"""pages/validation.py — Phase 6: failure analysis + ablation study."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from data import (
    load_failure_metrics, load_failure_error_img, load_failure_heatmap_img,
    load_ablation_ensemble_metrics, load_ablation_ensemble_img,
    load_ablation_weighting_metrics, load_ablation_weighting_img,
)
from styles import missing, phase_pill


def render() -> None:
    st.markdown(
        f'{phase_pill("Phase 6")}'
        '<h2>Validation — Failure Analysis &amp; Ablation Study</h2>',
        unsafe_allow_html=True,
    )

    tab_fail, tab_ens, tab_wt = st.tabs(
        ["🔴 Failure Analysis (ν sweep)",
         "📊 Ablation A: Ensemble Size",
         "⚖️ Ablation B: Loss Weighting"]
    )

    # ══ Tab 1 — Failure analysis ══════════════════════════════════════════
    with tab_fail:
        st.markdown(
            '<div class="card">'
            'Vanilla PINN (4×50 tanh, 5,000 epochs) tested at 4 viscosity values. '
            'Smaller ν → sharper shock → harder problem. '
            '<b>Failure modes:</b> spectral bias, gradient stiffness (PDE residual '
            'scales as 1/ν), collocation starvation (~3 pts in shock at ν=0.001/π).'
            '</div>',
            unsafe_allow_html=True,
        )

        fail = load_failure_metrics()
        if fail:
            nu_labels = [
                "0.01/π\n(baseline)",
                "0.005/π\n(2×)",
                "0.002/π\n(5×)",
                "0.001/π\n(10×)",
            ]
            errs  = [r["rel_l2"] * 100 for r in fail]
            nus   = [r["nu"] for r in fail]
            colors = ["#34d399", "#fbbf24", "#f97316", "#ff3355"]

            col_a, col_b = st.columns(2)
            with col_a:
                fig = go.Figure(go.Bar(
                    x=nu_labels, y=errs,
                    marker_color=colors,
                    text=[f"{e:.1f}%" for e in errs],
                    textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,0,18,0.6)",
                    title=dict(text="Rel-L2 error vs viscosity ν",
                               font=dict(color="#ff3355")),
                    xaxis_title="ν", yaxis_title="Rel-L2 error (%)",
                    height=340, margin=dict(t=50, b=60, l=50, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                # Log-log plot (reproduces failure_error_vs_nu.png interactively)
                fig2 = go.Figure(go.Scatter(
                    x=nus, y=[e / 100 for e in errs],
                    mode="lines+markers",
                    line=dict(color="#ff3355", width=2.5),
                    marker=dict(size=10, color=colors,
                                line=dict(color="#ffffff", width=1)),
                    text=[f"ν={n:.2e}<br>{e:.1f}%" for n, e in zip(nus, errs)],
                    hoverinfo="text",
                ))
                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,0,18,0.6)",
                    title=dict(text="Log-log: error vs ν",
                               font=dict(color="#ff3355")),
                    xaxis=dict(type="log", title="ν (log)", autorange="reversed"),
                    yaxis=dict(type="log", title="Rel-L2 (log)"),
                    height=340, margin=dict(t=50, b=50, l=60, r=20),
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown(
                '<div class="callout">'
                '<b>Error plateau:</b> Error jumps sharply from 4.6% → 20.8% at 2× '
                'sharpening, then saturates at ~33–35% for 5× and 10×. Beyond a '
                'threshold sharpness the PINN learns the best smooth approximation '
                'it can, and further shock sharpening no longer changes it.'
                '</div>',
                unsafe_allow_html=True,
            )

        else:
            missing("failure_analysis/metrics.json not found")

        c1, c2 = st.columns(2)
        with c1:
            img = load_failure_error_img()
            if img:
                st.image(img, caption="Log-log error vs ν (saved plot)",
                         use_container_width=True)
            else:
                missing("failure_error_vs_nu.png missing")
        with c2:
            img2 = load_failure_heatmap_img()
            if img2:
                st.image(img2,
                         caption="PINN vs FD at ν=0.001/π — shock entirely missed",
                         use_container_width=True)
            else:
                missing("failure_heatmap_comparison.png missing")

    # ══ Tab 2 — Ablation A: Ensemble size ════════════════════════════════
    with tab_ens:
        st.markdown(
            '<div class="card">'
            'M ∈ {3, 5, 10, 20} at baseline ν. Members 0–9 reused from Phase 2; '
            'members 10–19 trained fresh (seeds 10–19).'
            '</div>',
            unsafe_allow_html=True,
        )

        abl_ens = load_ablation_ensemble_metrics()
        if abl_ens:
            Ms        = [r["M"]           for r in abl_ens]
            eces      = [r["ece"]         for r in abl_ens]
            coverages = [r["coverage_90"] for r in abl_ens]
            mses      = [r["mse"]         for r in abl_ens]

            col_a, col_b = st.columns(2)
            with col_a:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=Ms, y=eces, name="ECE ↓",
                    mode="lines+markers",
                    line=dict(color="#ff3355", width=2.5),
                    marker=dict(size=10),
                ))
                fig.add_trace(go.Scatter(
                    x=Ms, y=coverages, name="90% Coverage ↑",
                    mode="lines+markers",
                    line=dict(color="#4da6ff", width=2.5, dash="dot"),
                    marker=dict(size=10),
                    yaxis="y2",
                ))
                fig.add_hline(y=0.9, line_dash="dash", line_color="#aaaaaa",
                              yref="y2",
                              annotation_text="ideal 0.90",
                              annotation_font_color="#aaaaaa")
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,0,18,0.6)",
                    title=dict(text="ECE and 90% Coverage vs M",
                               font=dict(color="#ff3355")),
                    xaxis=dict(title="M", tickvals=Ms),
                    yaxis=dict(title="ECE", color="#ff3355"),
                    yaxis2=dict(title="Coverage", overlaying="y",
                                side="right", range=[0, 1.05], color="#4da6ff"),
                    legend=dict(font=dict(color="#f0e8ec")),
                    height=340, margin=dict(t=50, b=40, l=50, r=60),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig2 = go.Figure(go.Bar(
                    x=Ms, y=mses,
                    marker_color=["#34d399" if m == min(mses) else "#ff3355"
                                  for m in mses],
                    text=[f"{v:.2e}" for v in mses],
                    textposition="outside",
                ))
                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,0,18,0.6)",
                    title=dict(text="MSE vs M (monotonically decreasing)",
                               font=dict(color="#ff3355")),
                    xaxis=dict(title="M", tickvals=Ms),
                    yaxis=dict(title="MSE"),
                    height=340, margin=dict(t=50, b=40, l=60, r=20),
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Summary table
            df = pd.DataFrame({
                "M": Ms,
                "ECE ↓": [f"{v:.4f}" for v in eces],
                "90% Coverage": [f"{v:.4f}" for v in coverages],
                "MSE ↓": [f"{v:.4e}" for v in mses],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown(
                '<div class="callout">'
                '<b>Key finding — ECE is non-monotonic:</b> M=10 achieves the best '
                'calibration (ECE=0.071), better than M=20 (0.134). MSE decreases '
                'monotonically. M=20 over-disperses, inflating ECE. '
                '<b>Recommendation: M=10 is the sweet spot.</b>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            missing("ensemble_size_metrics.json not found")

        img = load_ablation_ensemble_img()
        if img:
            st.image(img, caption="ECE + 90% coverage vs M (saved plot)",
                     use_container_width=True)
        else:
            missing("ablation_ensemble_size.png missing")

    # ══ Tab 3 — Ablation B: Loss weighting ═══════════════════════════════
    with tab_wt:
        st.markdown(
            '<div class="card">'
            'Three loss weight configurations for the vanilla PINN at baseline ν.'
            '</div>',
            unsafe_allow_html=True,
        )

        abl_wt = load_ablation_weighting_metrics()
        if abl_wt:
            labels  = ["(a) Baseline\nλ_ic=10, λ_bc=10",
                       "(b) Uniform\nλ_ic=1, λ_bc=1",
                       "(c) Auto-balanced\nλ_ic=0.019, λ_bc=0.075"]
            rl2s   = [r["rel_l2"] * 100 for r in abl_wt]
            mses   = [r["mse"]          for r in abl_wt]
            colors = ["#34d399", "#fbbf24", "#ff3355"]

            col_a, col_b = st.columns(2)
            with col_a:
                fig = go.Figure(go.Bar(
                    y=labels, x=rl2s,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:.2f}%" for v in rl2s],
                    textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,0,18,0.6)",
                    title=dict(text="Rel-L2 error per weighting scheme",
                               font=dict(color="#ff3355")),
                    xaxis=dict(type="log", title="Rel-L2 (%) — log scale"),
                    height=300, margin=dict(t=50, b=40, l=30, r=60),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                scheme_labels = ["(a) Baseline", "(b) Uniform", "(c) Auto-balanced"]
                df = pd.DataFrame({
                    "Scheme": scheme_labels,
                    "λ_ic": [r.get("lambda_ic", "auto") for r in abl_wt],
                    "λ_bc": [r.get("lambda_bc", "auto") for r in abl_wt],
                    "MSE":  [f"{r['mse']:.4e}" for r in abl_wt],
                    "Rel-L2": [f"{r['rel_l2']*100:.2f}%" for r in abl_wt],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown(
                '<div class="callout">'
                '<b>Negative result — auto-balance fails on the forward problem '
                '(51% vs 4% error).</b> At random init, IC/BC losses are large '
                '(~0.5) while PDE loss is small (~1e-3). Auto-balance sets '
                'λ_ic≈0.019 — effectively down-weighting the IC by 500× vs '
                'baseline. The PINN ignores the initial condition and produces '
                'a qualitatively wrong field.<br><br>'
                '<b>Lesson:</b> Auto-balancing is context-dependent. It fixed '
                'the inverse problem\'s lambda_data pathology, but <em>the '
                'physical prior that IC/BC need upweighting cannot be replaced '
                'by a data-driven weight computed at random init</em>.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            missing("loss_weighting_metrics.json not found")

        img = load_ablation_weighting_img()
        if img:
            st.image(img, caption="MSE per weighting scheme (saved plot)",
                     use_container_width=True)
        else:
            missing("ablation_loss_weighting.png missing")
