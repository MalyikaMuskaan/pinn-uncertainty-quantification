"""pages/burgers.py — Burgers' PINN solution viewer with time slider."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from data import (
    load_burgers_heatmap, load_burgers_time_slices, load_burgers_loss_history,
    load_ensemble_mean_heatmap, load_ensemble_std_heatmap,
)
from styles import missing, phase_pill

# Try to load the checkpoint for live inference; gracefully skip if unavailable
def _try_load_model():
    """Attempt to load the trained BurgersPINN checkpoint for live eval."""
    try:
        import sys, pathlib, torch
        root = pathlib.Path(__file__).parent.parent.parent
        sys.path.insert(0, str(root / "burgers_pinn"))
        from model import BurgersPINN
        ckpt_path = root / "burgers_pinn" / "outputs" / "burgers_pinn.pt"
        if not ckpt_path.exists():
            return None, None
        model = BurgersPINN(n_hidden=4, n_neurons=50)
        model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
        model.eval()
        return model, torch
    except Exception:
        return None, None


@st.cache_resource
def _get_model():
    return _try_load_model()


@st.cache_data
def _evaluate_model_grid(n_x: int = 100, n_t: int = 80):
    """Evaluate the PINN on an n_x × n_t grid; return arrays."""
    model, torch = _get_model()
    if model is None:
        return None, None, None
    import numpy as np
    x_np = np.linspace(-1, 1, n_x, dtype=np.float32)
    t_np = np.linspace( 0, 1, n_t, dtype=np.float32)
    X, T = np.meshgrid(x_np, t_np)
    x_flat = torch.tensor(X.reshape(-1, 1), requires_grad=False)
    t_flat = torch.tensor(T.reshape(-1, 1), requires_grad=False)
    with torch.no_grad():
        u_flat = model(x_flat, t_flat).numpy().reshape(n_t, n_x)
    return x_np, t_np, u_flat


def render() -> None:
    st.markdown(
        f'{phase_pill("Phase 1")}'
        '<h2>Burgers\' PINN — Solution Viewer</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card">'
        '<b>PDE:</b> u<sub>t</sub> + u·u<sub>x</sub> = ν·u<sub>xx</sub> &nbsp;|&nbsp; '
        'ν = 0.01/π &nbsp;|&nbsp; x∈[−1,1], t∈[0,1] &nbsp;|&nbsp; '
        'IC: u(x,0) = −sin(πx) &nbsp;|&nbsp; BC: u(±1,t) = 0'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_live, tab_saved = st.tabs(["🎛 Live inference (checkpoint)", "🖼 Saved outputs"])

    # ── Tab 1: live interactive heatmap ──────────────────────────────────
    with tab_live:
        x_np, t_np, u_grid = _evaluate_model_grid()

        if u_grid is None:
            missing("BurgersPINN checkpoint not found. "
                    "Run burgers_pinn/main.py to generate outputs/burgers_pinn.pt.")
        else:
            n_t, n_x = u_grid.shape
            t_idx = st.slider(
                "Time step  t", 0, n_t - 1, n_t // 2,
                help="Scrub through the solution in time",
            )
            t_val = float(t_np[t_idx])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_np, y=u_grid[t_idx],
                mode="lines",
                line=dict(color="#ff3355", width=2.5),
                name=f"PINN  t={t_val:.2f}",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,0,18,0.6)",
                xaxis_title="x",
                yaxis_title="u(x, t)",
                title=dict(text=f"u(x, t={t_val:.2f})  — PINN prediction",
                           font=dict(color="#ff3355")),
                yaxis=dict(range=[-1.2, 1.2]),
                height=350,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Full heatmap
            fig2 = go.Figure(go.Heatmap(
                z=u_grid, x=x_np, y=t_np,
                colorscale="RdBu_r",
                zmid=0,
                colorbar=dict(title="u(x,t)", tickfont=dict(color="#f0e8ec")),
            ))
            fig2.add_shape(type="line",
                           x0=-1, x1=1, y0=t_val, y1=t_val,
                           line=dict(color="#ffcc00", width=1.5, dash="dot"))
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,0,18,0.6)",
                xaxis_title="x", yaxis_title="t",
                title=dict(text="Full solution heatmap (yellow line = current t)",
                           font=dict(color="#ff3355")),
                height=320,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: saved PNG outputs ──────────────────────────────────────────
    with tab_saved:
        col1, col2 = st.columns(2)
        with col1:
            img = load_burgers_heatmap()
            if img:
                st.image(img, caption="Solution heatmap u(x,t)", use_container_width=True)
            else:
                missing("heatmap.png not found")
            img2 = load_ensemble_mean_heatmap()
            if img2:
                st.image(img2, caption="Ensemble mean heatmap", use_container_width=True)
            else:
                missing("ensemble_mean_heatmap.png not found")
        with col2:
            img3 = load_burgers_time_slices()
            if img3:
                st.image(img3, caption="PINN vs FD reference — time slices",
                         use_container_width=True)
            else:
                missing("time_slices.png not found")
            img4 = load_ensemble_std_heatmap()
            if img4:
                st.image(img4, caption="Ensemble uncertainty std(u)", use_container_width=True)
            else:
                missing("ensemble_std_heatmap.png not found")

        img5 = load_burgers_loss_history()
        if img5:
            st.image(img5, caption="Training loss history", use_container_width=True)
        else:
            missing("loss_history.png not found")
