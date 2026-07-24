"""pages/about.py — Methodology summary and project-wide findings."""
import streamlit as st
from styles import phase_pill


def render() -> None:
    st.markdown(
        '<h2>About — Methodology &amp; Cross-Project Findings</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card">'
        '<b>Context:</b> PhD application research project for KAUST '
        'Computational Science &amp; Engineering. All training was performed on '
        'Google Colab (T4 GPU). The project systematically builds from a '
        'single-equation baseline to multi-method UQ, inverse problems, operator '
        'learning, 2D PDEs, and rigorous validation.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── PINN accuracy by problem type ─────────────────────────────────────
    st.markdown("### PINN Accuracy vs Problem Type")
    st.markdown(
        '<div class="card">'
        '<table style="width:100%;font-size:0.88rem;color:#f0e8ec;">'
        '<tr style="color:#ff3355;">'
        '<th>Problem</th><th>PDE type</th><th>Nonlinear?</th>'
        '<th>Reference</th><th>Final accuracy</th></tr>'
        '<tr><td>Burgers\' 1D</td><td>Parabolic</td><td>Yes (u·u_x)</td>'
        '<td>Crank-Nicolson FD</td><td>MSE 7.8×10⁻⁴ (~5%)</td></tr>'
        '<tr><td>Advection-diffusion 1D</td><td>Parabolic</td><td>No</td>'
        '<td>FD reference</td><td>MSE 7.0×10⁻⁵ (~3%)</td></tr>'
        '<tr><td>Darcy 2D</td><td>Elliptic</td><td>No</td>'
        '<td>Analytical MMS</td><td><b style="color:#34d399;">MSE 3.3×10⁻¹⁰ (0.004%)</b></td></tr>'
        '<tr><td>Burgers\' inverse</td><td>Parabolic</td><td>Yes</td>'
        '<td>—</td><td>ν error 17.6% (best condition)</td></tr>'
        '<tr><td>Burgers\' (ν=0.001/π)</td><td>Parabolic</td><td>Yes</td>'
        '<td>FD</td><td><b style="color:#ff3355;">35% (failure case)</b></td></tr>'
        '</table></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### When to Use Each Method")
    st.markdown(
        '<div class="card">'
        '<table style="width:100%;font-size:0.88rem;color:#f0e8ec;">'
        '<tr style="color:#ff3355;"><th>Objective</th><th>Recommended approach</th></tr>'
        '<tr><td>Best forward accuracy + calibrated UQ</td>'
        '<td>Deep Ensemble (M=10)</td></tr>'
        '<tr><td>Many ICs, fast amortised inference</td>'
        '<td>Fourier Neural Operator</td></tr>'
        '<tr><td>Unknown parameter from sparse sensor data</td>'
        '<td>Inverse PINN + Adam warm-up + L-BFGS + auto-balanced λ_data</td></tr>'
        '<tr><td>2D steady-state elliptic PDE</td>'
        '<td>Single PINN + L-BFGS refinement</td></tr>'
        '<tr><td>Sharp-shock low-ν problems</td>'
        '<td>FNO (avoids PDE residual stiffness) or adaptive collocation (RAR)</td></tr>'
        '</table></div>',
        unsafe_allow_html=True,
    )

    # ── Key cross-project lessons ──────────────────────────────────────────
    st.markdown("### Key Lessons Across All Phases")

    lessons = [
        ("Deep Ensemble UQ quality depends on problem nonlinearity",
         "All 10 members converge to the same minimum on the linear advection-diffusion "
         "equation (ECE 0.102 vs 0.083 on Burgers'). Ensemble diversity — the source of "
         "calibrated uncertainty — requires multi-modal loss landscapes."),
        ("The auto-balance fix is context-dependent",
         "lambda_data auto-balancing fixed the inverse problem (400% → 17% error) but "
         "catastrophically failed on the forward PINN (51% vs 4% error). At random init, "
         "IC/BC losses are large and PDE loss is small — auto-balance down-weights IC by "
         "500×, letting the network ignore the initial condition. The physical prior "
         "(IC/BC need upweighting) cannot be replaced by a data-driven heuristic."),
        ("ECE is non-monotonic in ensemble size",
         "MSE decreases monotonically M=3→20, but ECE has a minimum at M=10 (0.071). "
         "M=20 over-disperses the predictive distribution (ECE 0.134). For calibration, "
         "M=10 is optimal for this problem."),
        ("FNO breaks even after just 4 instances",
         "The FNO's 76-second one-time training cost is repaid after only 4 instances. "
         "For any workload ≥5 ICs the FNO is strictly cheaper, and at steady state "
         "is ~8,000× faster per query than retraining a PINN."),
        ("Elliptic PINNs converge to near-machine-epsilon accuracy",
         "Darcy 2D achieves rel-L2 0.004% — 2,400× lower than the Burgers' ensemble. "
         "The convex residual landscape of linear elliptic PDEs makes L-BFGS extremely "
         "effective. The analytical MMS reference removes all discretisation error."),
        ("PINN failure at small ν is sharp and plateaus",
         "Error jumps 4.6× from baseline → 2× sharpening, then saturates at ~35% "
         "for 5× and 10×. Once the shock is too narrow for the smooth tanh network, "
         "further sharpening doesn't change the learned approximation."),
    ]

    for title, body in lessons:
        st.markdown(
            f'<div class="card">'
            f'<b style="color:#ff3355;">{title}</b>'
            f'<p style="color:#d0c8cc;margin-top:6px;font-size:0.9rem;">{body}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### References")
    st.markdown(
        '<div class="card" style="font-size:0.85rem;color:#b09098;">'
        '• Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). '
        'Physics-informed neural networks. <em>JCP</em>.<br>'
        '• Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). '
        'Simple and scalable predictive uncertainty estimation using deep ensembles. '
        '<em>NeurIPS</em>.<br>'
        '• Li, Z., Kovachki, N., et al. (2020). '
        'Fourier Neural Operator for Parametric PDEs. <em>ICLR 2021</em>.<br>'
        '• Rahaman, N., et al. (2019). '
        'On the spectral bias of neural networks. <em>ICML</em>.<br>'
        '• Lu, L., et al. (2021). '
        'DeepXDE: A deep learning library for solving PDEs. <em>SIAM Review</em>.<br>'
        '• Wang, S., et al. (2022). '
        'Respecting causality for training physics-informed neural networks. '
        '<em>CMAME</em>.'
        '</div>',
        unsafe_allow_html=True,
    )
