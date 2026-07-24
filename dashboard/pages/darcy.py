"""pages/darcy.py — 2D Darcy flow PINN results."""
import streamlit as st
from data import (
    load_darcy_metrics, load_darcy_solution_img,
    load_darcy_loss_img, load_darcy_residual_img,
)
from styles import missing, phase_pill, stat_badge


def render() -> None:
    st.markdown(
        f'{phase_pill("Phase 5b")}'
        '<h2>2D Darcy Flow PINN — First 2D Result</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card">'
        '<b>PDE:</b> −∇·(k(x,y) ∇u(x,y)) = f(x,y) &nbsp; on [0,1]² &nbsp;|&nbsp; '
        'u = 0 on ∂Ω<br>'
        '<b>Permeability:</b> k(x,y) = 1 + 0.5·sin(πx)·sin(πy)<br>'
        '<b>Exact solution (MMS):</b> u*(x,y) = sin(πx)·sin(πy) — '
        'zero on all 4 edges, peak 1 at (0.5, 0.5)<br>'
        '<b>Training:</b> 5×64 tanh PINN, Adam 3,000 + L-BFGS 2,000 epochs, '
        'auto-balanced λ_bc'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    m = load_darcy_metrics()
    if m:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(stat_badge(f"{m['mse']:.2e}", "MSE"), unsafe_allow_html=True)
        with col2:
            st.markdown(stat_badge(f"{m['rel_l2']*100:.4f}%", "Rel-L2"),
                        unsafe_allow_html=True)
        with col3:
            st.markdown(stat_badge(f"{m['train_time']/60:.1f} min", "Train time (T4)"),
                        unsafe_allow_html=True)
        with col4:
            st.markdown(stat_badge(f"{m['n_epochs']:,}", "Total epochs"),
                        unsafe_allow_html=True)

        st.markdown(
            '<div class="callout">'
            f'MSE of <b>{m["mse"]:.2e}</b> is ~2,400× lower than the Burgers\' '
            'ensemble MSE (7.82×10⁻⁴). Three reasons: (1) linear PDE → convex '
            'residual landscape; (2) analytical MMS reference has no discretisation '
            'error; (3) L-BFGS is highly effective on smooth elliptic problems.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        missing("darcy_2d/outputs/metrics.json not found")

    st.markdown("---")

    # ── Solution comparison (main result) ─────────────────────────────────
    img = load_darcy_solution_img()
    if img:
        st.image(img,
                 caption="3-panel: PINN prediction | Exact u* = sin(πx)sin(πy) | "
                         "Pointwise error |û − u*|",
                 use_container_width=True)
    else:
        missing("solution_comparison.png not found")

    col1, col2 = st.columns(2)
    with col1:
        img2 = load_darcy_loss_img()
        if img2:
            st.image(img2, caption="Training loss history — Adam → L-BFGS transition",
                     use_container_width=True)
        else:
            missing("loss_history.png not found")
    with col2:
        img3 = load_darcy_residual_img()
        if img3:
            st.image(img3, caption="PDE residual |R(x,y)| at 128×128 — "
                     "uniformly near-zero, no hotspots",
                     use_container_width=True)
        else:
            missing("pde_residual_map.png not found")

    # ── Differences vs 1D PINNs ───────────────────────────────────────────
    st.markdown("### How 2D Darcy differs from 1D PINNs")
    st.markdown(
        '<div class="card">'
        '<table style="width:100%;font-size:0.88rem;color:#f0e8ec;">'
        '<tr style="color:#ff3355;">'
        '<th>Aspect</th><th>1D Burgers / Ocean</th><th>2D Darcy (this)</th></tr>'
        '<tr><td>Inputs</td><td>(x, t)</td><td>(x, y)</td></tr>'
        '<tr><td>Derivatives</td><td>u_t, u_x, u_xx</td><td>u_x, u_y, u_xx, u_yy</td></tr>'
        '<tr><td>PDE type</td><td>Parabolic (time-dep.)</td><td>Elliptic (steady-state)</td></tr>'
        '<tr><td>Boundary data</td><td>IC at t=0 + BC at x=±1</td><td>Dirichlet on all 4 edges</td></tr>'
        '<tr><td>Reference</td><td>Crank-Nicolson FD</td><td>Analytical MMS</td></tr>'
        '<tr><td>Architecture</td><td>4×50 tanh</td><td>5×64 tanh</td></tr>'
        '</table></div>',
        unsafe_allow_html=True,
    )
