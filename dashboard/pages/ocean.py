"""pages/ocean.py — Ocean/climate 1D advection-diffusion PDE viewer."""
import streamlit as st
from data import (
    load_ocean_heatmap, load_ocean_ensemble_mean, load_ocean_ensemble_std,
    load_ocean_ensemble_time_slices, load_ocean_calibration,
)
from styles import missing, phase_pill


def render() -> None:
    st.markdown(
        f'{phase_pill("Phase 3")}'
        '<h2>Ocean / Climate PDE — Advection-Diffusion</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card">'
        '<b>PDE:</b> ∂c/∂t + v·∂c/∂x = D·∂²c/∂x² &nbsp;|&nbsp;'
        ' v = 1.0, D = 0.05 &nbsp;|&nbsp; x∈[0,10] km, t∈[0,5] &nbsp;|&nbsp;'
        ' IC: Gaussian pulse at x = 2 km'
        '<br><br>'
        '<b>Key finding:</b> The linear PDE converges ~10× faster than Burgers\' '
        '(PDE loss 3.4×10⁻⁶ vs 9.0×10⁻⁴). However, Deep Ensemble calibration '
        'is worse (ECE 0.102 vs 0.083) because all 10 members converge to the '
        '<em>same</em> convex minimum — ensemble diversity requires multi-modality.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Comparison table vs Burgers' ──────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            '<div class="card">'
            '<b style="color:#ff3355;">Convergence comparison</b>'
            '<table style="width:100%;margin-top:8px;font-size:0.85rem;color:#f0e8ec;">'
            '<tr><th>Metric</th><th>Adv-diff</th><th>Burgers\'</th></tr>'
            '<tr><td>PDE loss</td><td style="color:#ff3355;"><b>3.4×10⁻⁶</b></td><td>9.0×10⁻⁴</td></tr>'
            '<tr><td>MSE vs ref</td><td style="color:#ff3355;"><b>7.0×10⁻⁵</b></td><td>7.8×10⁻⁴</td></tr>'
            '</table></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            '<div class="card">'
            '<b style="color:#4da6ff;">Calibration comparison</b>'
            '<table style="width:100%;margin-top:8px;font-size:0.85rem;color:#f0e8ec;">'
            '<tr><th>Metric</th><th>Adv-diff</th><th>Burgers\'</th></tr>'
            '<tr><td>ECE</td><td>0.102</td><td style="color:#4da6ff;"><b>0.083</b></td></tr>'
            '<tr><td>90% Coverage</td><td>0.769</td><td style="color:#4da6ff;"><b>0.886</b></td></tr>'
            '</table></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Heatmap images ────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        img = load_ocean_heatmap()
        if img:
            st.image(img, caption="Advection-diffusion solution c(x,t)",
                     use_container_width=True)
        else:
            missing("ocean_pinn/outputs/heatmap.png not found")

        img2 = load_ocean_ensemble_std()
        if img2:
            st.image(img2, caption="Ensemble uncertainty std(c) — diagonal stripe "
                     "following the characteristic x = x₀ + v·t",
                     use_container_width=True)
        else:
            missing("ensemble_std_heatmap.png not found")

    with c2:
        img3 = load_ocean_ensemble_mean()
        if img3:
            st.image(img3, caption="Ensemble mean prediction",
                     use_container_width=True)
        else:
            missing("ensemble_mean_heatmap.png not found")

        img4 = load_ocean_calibration()
        if img4:
            st.image(img4, caption="Calibration diagram — ocean ensemble",
                     use_container_width=True)
        else:
            missing("ocean calibration plot not found")

    img5 = load_ocean_ensemble_time_slices()
    if img5:
        st.image(img5, caption="Ensemble mean ± 2σ time slices vs reference",
                 use_container_width=True)
    else:
        missing("ensemble_time_slices.png not found")

    st.markdown(
        '<div class="callout">'
        '<b>Physical interpretation:</b> Uncertainty is highest at the '
        '<em>leading edge and flanks</em> of the pollutant plume, not at the '
        'known source. For ocean monitoring, sensors should be placed at the '
        'advancing front — the region the ensemble is most uncertain about.'
        '</div>',
        unsafe_allow_html=True,
    )
