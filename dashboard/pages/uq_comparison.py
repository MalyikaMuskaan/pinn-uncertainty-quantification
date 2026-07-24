"""pages/uq_comparison.py — UQ method comparison: metrics cards + calibration plots."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from data import (
    load_ensemble_metrics, load_bayesian_metrics, load_dropout_metrics,
    load_comparison_table,
    load_ensemble_calibration_img, load_bayesian_calibration_img,
    load_dropout_calibration_img,
    load_uncertainty_comparison_img, load_calibration_comparison_img,
    load_ensemble_time_slices_img,
)
from styles import missing, phase_pill, stat_badge


def _metric_card(method: str, data: dict | None, color: str) -> None:
    if data is None:
        missing(f"{method} metrics not found.")
        return
    mse = data.get("mse", float("nan"))
    ece = data.get("ece", float("nan"))
    cov = data.get("coverage_90", float("nan"))
    tt  = data.get("train_time_s", float("nan"))
    it  = data.get("inference_time_s", float("nan"))
    st.markdown(
        f'<div class="card" style="border-color:{color}55;">'
        f'<b style="color:{color};">{method}</b>'
        f'<table style="width:100%;margin-top:8px;font-size:0.88rem;color:#f0e8ec;">'
        f'<tr><td>MSE</td><td style="text-align:right;color:{color};">'
        f'<b>{mse:.3e}</b></td></tr>'
        f'<tr><td>ECE</td><td style="text-align:right;">{ece:.4f}</td></tr>'
        f'<tr><td>90% Coverage</td><td style="text-align:right;">'
        f'{cov:.4f}</td></tr>'
        f'<tr><td>Train time</td><td style="text-align:right;">'
        f'{tt/60:.1f} min</td></tr>'
        f'<tr><td>Infer time</td><td style="text-align:right;">'
        f'{it:.1f} s</td></tr>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


def render() -> None:
    st.markdown(
        f'{phase_pill("Phase 2")}'
        '<h2>Uncertainty Quantification Comparison</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card">Three UQ methods applied to the same Burgers\' PINN setup, '
        'evaluated on a 256×100 grid vs Crank-Nicolson reference.</div>',
        unsafe_allow_html=True,
    )

    ens_m = load_ensemble_metrics()
    bay_m = load_bayesian_metrics()
    drp_m = load_dropout_metrics()

    # ── Metrics cards ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        _metric_card("Deep Ensemble (M=10)", ens_m, "#ff3355")
    with col2:
        _metric_card("Bayesian PINN (VI)", bay_m, "#4da6ff")
    with col3:
        _metric_card("MC Dropout (p=0.05)", drp_m, "#ffa040")

    # ── Interactive bar chart ──────────────────────────────────────────────
    if ens_m and bay_m and drp_m:
        methods  = ["Deep Ensemble", "Bayesian PINN", "MC Dropout"]
        mse_vals = [ens_m["mse"], bay_m["mse"], drp_m["mse"]]
        ece_vals = [ens_m["ece"], bay_m["ece"], drp_m["ece"]]
        cov_vals = [ens_m["coverage_90"], bay_m["coverage_90"], drp_m["coverage_90"]]
        colors   = ["#ff3355", "#4da6ff", "#ffa040"]

        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure(go.Bar(
                x=methods, y=mse_vals,
                marker_color=colors,
                text=[f"{v:.2e}" for v in mse_vals],
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,0,18,0.6)",
                title=dict(text="MSE  (lower = better)", font=dict(color="#ff3355")),
                yaxis=dict(type="log", title="MSE (log)"),
                height=300, margin=dict(t=50, b=30, l=40, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="ECE", x=methods, y=ece_vals,
                marker_color=colors, opacity=0.9,
                text=[f"{v:.3f}" for v in ece_vals], textposition="outside",
            ))
            fig2.add_trace(go.Scatter(
                name="90% Coverage", x=methods, y=cov_vals,
                mode="markers+lines",
                marker=dict(size=12, color="#ffcc00"),
                line=dict(color="#ffcc00", width=2, dash="dot"),
                yaxis="y2",
            ))
            fig2.add_hline(y=0.9, line_dash="dash", line_color="#aaaaaa",
                           annotation_text="ideal 90%",
                           annotation_font_color="#aaaaaa",
                           yref="y2")
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,0,18,0.6)",
                title=dict(text="ECE (bars) + 90% Coverage (line)",
                           font=dict(color="#ff3355")),
                yaxis=dict(title="ECE"),
                yaxis2=dict(title="Coverage", overlaying="y", side="right",
                            range=[0, 1.05]),
                height=300, margin=dict(t=50, b=30, l=40, r=60),
                legend=dict(font=dict(color="#f0e8ec")),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Calibration plots ─────────────────────────────────────────────────
    st.markdown("### Calibration Reliability Diagrams")
    cal_imgs = [
        ("Deep Ensemble", load_ensemble_calibration_img()),
        ("Bayesian PINN", load_bayesian_calibration_img()),
        ("MC Dropout",    load_dropout_calibration_img()),
    ]
    cols = st.columns(3)
    for col, (name, img) in zip(cols, cal_imgs):
        with col:
            if img:
                st.image(img, caption=name, use_container_width=True)
            else:
                missing(f"{name} calibration plot missing")

    st.markdown("### Combined Comparison Plots")
    c1, c2 = st.columns(2)
    with c1:
        img = load_uncertainty_comparison_img()
        if img:
            st.image(img, caption="Uncertainty std heatmaps (shared scale)",
                     use_container_width=True)
        else:
            missing("uncertainty_comparison.png missing")
    with c2:
        img2 = load_calibration_comparison_img()
        if img2:
            st.image(img2, caption="Calibration curves — all 3 methods",
                     use_container_width=True)
        else:
            missing("calibration_comparison.png missing")

    st.markdown("### Ensemble time slices (mean ± 2σ vs FD reference)")
    img3 = load_ensemble_time_slices_img()
    if img3:
        st.image(img3, caption="Ensemble mean ± 2σ vs Crank-Nicolson",
                 use_container_width=True)
    else:
        missing("ensemble_time_slices.png missing")

    # ── Full metrics table ────────────────────────────────────────────────
    st.markdown("### Full metrics table")
    df = load_comparison_table()
    if df is not None:
        st.dataframe(df.style.highlight_min(
            subset=["MSE", "ECE"], color="rgba(200,30,60,0.3)"
        ), use_container_width=True)
    else:
        missing("comparison_table.csv missing")
