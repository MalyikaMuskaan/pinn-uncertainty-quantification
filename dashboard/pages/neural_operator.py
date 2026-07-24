"""pages/neural_operator.py — FNO vs per-instance PINN comparison."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from data import (
    load_fno_metrics, load_fno_comparison_imgs, load_fno_summary_table_img,
)
from styles import missing, phase_pill


def render() -> None:
    st.markdown(
        f'{phase_pill("Phase 5a")}'
        '<h2>Fourier Neural Operator — Solution Operator Learning</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card">'
        'The FNO learns the <em>solution operator</em> G: u₀(x) → u(x,t) for '
        'Burgers\' equation — one trained model that generalises to any IC without '
        'retraining. Trained on 800 IC/solution pairs (Crank-Nicolson ground truth), '
        'evaluated on 100 held-out ICs.<br><br>'
        '<b>Architecture:</b> Lift → 4 Fourier layers (16 modes, width 64) → Project. '
        '562,276 parameters. Trained in 76 s on Colab T4.'
        '</div>',
        unsafe_allow_html=True,
    )

    m = load_fno_metrics()

    if m:
        fno_mean  = m["fno"]["rel_l2_mean"] * 100
        fno_std   = m["fno"]["rel_l2_std"]  * 100
        pinn_mean = m["pinn"]["rel_l2_mean"] * 100
        pinn_std  = m["pinn"]["rel_l2_std"]  * 100
        fno_infer = m["fno"]["inference_time_mean"] * 1000       # ms
        pinn_infer= m["pinn"]["inference_time_mean"] * 1000
        pinn_train= m["pinn"]["train_time_mean"]

        # ── Summary metrics ──────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        pairs = [
            (f"{fno_mean:.2f}%", "FNO Rel-L2"),
            (f"{pinn_mean:.2f}%", "PINN Rel-L2"),
            (f"{fno_infer:.1f} ms", "FNO infer/IC"),
            (f"76 s", "FNO train (one-time)"),
        ]
        from styles import stat_badge
        for col, (val, lbl) in zip([col1, col2, col3, col4], pairs):
            with col:
                st.markdown(stat_badge(val, lbl), unsafe_allow_html=True)

        # ── Error distribution ────────────────────────────────────────────
        st.markdown("### Rel-L2 error distribution (100 test ICs)")
        fno_all  = np.array(m["fno"]["rel_l2_all"])  * 100
        pinn_all = np.array(m["pinn"]["rel_l2_all"]) * 100

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=fno_all, name="FNO",
            nbinsx=20, marker_color="#ff3355", opacity=0.75,
        ))
        fig.add_trace(go.Histogram(
            x=pinn_all, name="PINN",
            nbinsx=10, marker_color="#4da6ff", opacity=0.75,
        ))
        fig.add_vline(x=fno_mean,  line_dash="dash", line_color="#ff3355",
                      annotation_text=f"FNO mean {fno_mean:.2f}%",
                      annotation_font_color="#ff3355")
        fig.add_vline(x=pinn_mean, line_dash="dash", line_color="#4da6ff",
                      annotation_text=f"PINN mean {pinn_mean:.2f}%",
                      annotation_font_color="#4da6ff",
                      annotation_position="top left")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,0,18,0.6)",
            barmode="overlay",
            xaxis_title="Rel-L2 error (%)",
            yaxis_title="Count",
            title=dict(text="Error distribution — FNO vs PINN",
                       font=dict(color="#ff3355")),
            legend=dict(font=dict(color="#f0e8ec")),
            height=320,
            margin=dict(t=50, b=40, l=50, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Break-even analysis ───────────────────────────────────────────
        st.markdown("### Break-even analysis")
        n_instances = np.arange(1, 21)
        fno_cost    = 76 + fno_infer / 1000 * n_instances
        pinn_cost   = pinn_train * n_instances

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=n_instances, y=fno_cost,
            name="FNO (76s one-time + inference)",
            line=dict(color="#ff3355", width=2.5),
        ))
        fig2.add_trace(go.Scatter(
            x=n_instances, y=pinn_cost,
            name=f"PINN ({pinn_train:.0f}s × N)",
            line=dict(color="#4da6ff", width=2.5, dash="dot"),
        ))
        # Break-even
        be = int(np.ceil(76 / (pinn_train - fno_infer / 1000)))
        fig2.add_vline(x=be, line_dash="dash", line_color="#ffcc00",
                       annotation_text=f"Break-even N={be}",
                       annotation_font_color="#ffcc00")
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,0,18,0.6)",
            xaxis_title="Number of IC instances",
            yaxis_title="Total wall time (s)",
            title=dict(text="Cumulative cost: FNO vs per-instance PINN retraining",
                       font=dict(color="#ff3355")),
            legend=dict(font=dict(color="#f0e8ec")),
            height=320,
            margin=dict(t=50, b=40, l=50, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        missing("eval_fno_vs_pinn.json not found")

    st.markdown("---")
    st.markdown("### Sample predictions — FNO vs ground truth")
    imgs = load_fno_comparison_imgs()
    cols = st.columns(3)
    for i, (col, img) in enumerate(zip(cols, imgs)):
        with col:
            if img:
                st.image(img, caption=f"Test IC {i}", use_container_width=True)
            else:
                missing(f"comparison_{i:03d}.png missing")

    img_tbl = load_fno_summary_table_img()
    if img_tbl:
        st.image(img_tbl, caption="FNO vs PINN summary table",
                 use_container_width=True)
    else:
        missing("summary_table.png missing")
