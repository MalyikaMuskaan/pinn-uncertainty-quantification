"""
app.py
------
Main entry point for the Deep PINNs interactive dashboard.

Run with:
    cd d:/pnn
    streamlit run dashboard/app.py

Architecture
------------
- Left sidebar: radio-button navigation between 9 sections
- Full-page Three.js Red Sea background injected once via st.components.v1.html
- Global CSS (glass-card theme) injected once via st.markdown
- Each page is a separate module in dashboard/pages/ with a render() function
- All data is loaded via dashboard/data.py (st.cache_data — loaded once per session)

The Three.js canvas is position:fixed with pointer-events:none so it stays
behind all Streamlit widgets. A z-index:1 wrapper on .block-container ensures
all Streamlit content renders above it.
"""

import streamlit as st
import streamlit.components.v1 as components

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Deep PINNs Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import helpers AFTER set_page_config ──────────────────────────────────
from styles     import inject_css
from background import get_background_html
from pages      import (
    home, burgers, uq_comparison, ocean,
    inverse, neural_operator, darcy, validation, about,
)

# ── Inject global CSS ──────────────────────────────────────────────────────
inject_css()

# ── Inject Three.js background (position:fixed, pointer-events:none) ───────
# Height is set to 100vh; the component iframe itself is 0-height so it
# doesn't push content down. We give the component a fixed small height
# and hide the iframe border — the canvas escapes into the parent document
# via position:fixed in the injected HTML.
components.html(
    get_background_html(height=100),
    height=0,
    scrolling=False,
)

# Ensure Streamlit's .block-container sits above the canvas
st.markdown(
    """
    <style>
    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 1.5rem;
    }
    /* Remove default white background from the main content area */
    [data-testid="stMain"] {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:12px 0 8px;">'
        '<span style="font-size:2rem;">🌊</span><br>'
        '<b style="font-size:1.1rem;color:#ff3355;">Deep PINNs</b><br>'
        '<span style="font-size:0.75rem;color:#a09098;">'
        'KAUST PhD Research Portfolio'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    PAGE_OPTIONS = [
        "🏠  Home / Overview",
        "📈  Burgers' PINN",
        "🎯  UQ Comparison",
        "🌊  Ocean PDE",
        "🔍  Inverse Problem",
        "⚡  Neural Operator (FNO)",
        "🟥  2D Darcy Flow",
        "🔬  Failure & Ablation",
        "ℹ️  About & Methods",
    ]

    page = st.radio("Navigate", PAGE_OPTIONS, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#706070;padding:4px;">'
        'Streamlit 1.59 · Plotly 6 · Three.js r134<br>'
        'All data loaded from <code>outputs/</code> folders'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Page routing ───────────────────────────────────────────────────────────
if   "Home"         in page:  home.render()
elif "Burgers"      in page:  burgers.render()
elif "UQ"           in page:  uq_comparison.render()
elif "Ocean"        in page:  ocean.render()
elif "Inverse"      in page:  inverse.render()
elif "Neural"       in page:  neural_operator.render()
elif "Darcy"        in page:  darcy.render()
elif "Failure"      in page:  validation.render()
elif "About"        in page:  about.render()
