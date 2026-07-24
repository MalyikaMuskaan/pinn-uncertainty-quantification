"""pages/inverse.py — Inverse problem: ν recovery from sparse noisy sensors."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from data import (
    load_inverse_metrics, load_nu_convergence_img,
    load_robustness_summary,
    load_robustness_error_img, load_robustness_nu_img,
)
from styles import missing, phase_pill

NU_TRUE = 0.01 / np.pi


def render() -> None:
    st.markdown(
        f'{phase_pill("Phase 4")}'
        '<h2>Inverse Problem — Viscosity Recovery</h2>',
        unsafe_allow_html=True,
    )

    # ── Bug box ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="callout">'
        '<b>🐛 Systematic bug found and fixed.</b> Initial error was 400–700% '
        'across all conditions. Root cause: (1) <code>lambda_data</code> was '
        'negligibly small → network ignored sensor data; (2) <code>log_nu</code> '
        'shared the same Adam LR as weights → ν oscillated wildly.<br>'
        '<b>Fix:</b> Auto-balanced <code>lambda_data</code> at init + two-stage '
        'Adam → L-BFGS schedule. Error reduced 3–10× to 17–240%.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Setup card ────────────────────────────────────────────────────────
    m = load_inverse_metrics()
    nu_true = m["nu_true"] if m else NU_TRUE
    st.markdown(
        f'<div class="card">'
        f'<b>Goal:</b> Recover ν from sparse noisy sensor observations '
        f'(ν_true = {nu_true:.5f} = 0.01/π).<br>'
        f'<b>Method:</b> PINN with learnable <code>log_nu</code> parameter, '
        f'Adam warm-up (2,000 steps) + L-BFGS refinement (1,500 steps).<br>'
        f'<b>Sweep:</b> 3 noise levels × 3 sensor counts = 9 conditions, '
        f'10 ensemble members each.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Robustness table ──────────────────────────────────────────────────
    summary = load_robustness_summary()
    if summary:
        rows = []
        for cond in summary:
            rows.append({
                "Noise": f"{cond['noise_frac']*100:.1f}%",
                "Sensors": cond["n_sensors"],
                "ν mean": f"{cond['nu_mean']:.5f}",
                "ν std": f"{cond['nu_std']:.5f}",
                "90% CI": f"[{cond['ci_90_lo']:.4f}, {cond['ci_90_hi']:.4f}]",
                "Error %": f"{cond['mean_err_pct']:.1f}%",
                "True in CI": "✓" if cond["true_in_ci"] else "✗",
            })
        df = pd.DataFrame(rows)
        st.markdown("### 9-Condition Robustness Sweep")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Interactive bar chart of error % by condition
        labels = [f"{r['Noise']}/{r['Sensors']}" for r in rows]
        errs   = [cond["mean_err_pct"] for cond in summary]
        colors = ["#34d399" if cond["true_in_ci"] else "#ff3355"
                  for cond in summary]

        fig = go.Figure(go.Bar(
            x=labels, y=errs,
            marker_color=colors,
            text=[f"{e:.1f}%" for e in errs],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_color="#ffffff", line_width=0.5)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,0,18,0.6)",
            title=dict(text="ν Recovery Error % by condition "
                       "(green = true ν inside 90% CI)",
                       font=dict(color="#ff3355")),
            xaxis_title="Noise / Sensors",
            yaxis_title="Error %",
            height=380,
            margin=dict(t=60, b=60, l=50, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        missing("robustness/summary.json not found")

    st.markdown("---")

    # ── Plot images ───────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        img = load_robustness_error_img()
        if img:
            st.image(img, caption="Error % vs sensor count",
                     use_container_width=True)
        else:
            missing("robustness_error_vs_sensors.png missing")
    with c2:
        img2 = load_robustness_nu_img()
        if img2:
            st.image(img2, caption="ν estimates ± 90% CI vs true ν",
                     use_container_width=True)
        else:
            missing("robustness_nu_estimates.png missing")

    img3 = load_nu_convergence_img()
    if img3:
        st.image(img3, caption="ν convergence during training",
                 use_container_width=True)
    else:
        missing("nu_convergence.png missing")

    st.markdown(
        '<div class="callout">'
        '<b>Calibration:</b> Only 2/9 conditions captured true ν in the 90% CI '
        '(expected ~8/9). The ensemble is severely underdispersed at M=10. '
        'Fix: increase to M≥30, add bootstrap sensor sampling per member, '
        'apply conformal prediction recalibration.'
        '</div>',
        unsafe_allow_html=True,
    )
