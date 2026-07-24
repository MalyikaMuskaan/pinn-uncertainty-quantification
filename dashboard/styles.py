"""
styles.py
---------
Injects global CSS into the Streamlit app.

Design language
---------------
- Glass-morphism cards: semi-transparent dark surface with a red-tinted border,
  so they float visibly above the Three.js background without blocking it.
- All interactive Streamlit widgets automatically inherit the card context.
- Red Sea colour palette:
    --card-bg      : rgba(10, 0, 18, 0.72)  deep navy with slight transparency
    --card-border  : rgba(200, 30, 60, 0.35) muted crimson
    --accent       : #ff3355                 bright red (headings, badges)
    --accent-soft  : #cc2244
    --text         : #f0e8ec                 warm off-white
    --muted        : #a09098
- Sidebar and top-bar are also darkened to match the ocean theme.
"""

import streamlit as st


_CSS = """
/* ── Reset / root ─────────────────────────────────────────────────── */
:root {
  --card-bg:     rgba(10, 0, 18, 0.78);
  --card-border: rgba(200, 30, 60, 0.40);
  --accent:      #ff3355;
  --accent-soft: #cc2244;
  --text:        #f0e8ec;
  --muted:       #b09098;
  --font:        'Segoe UI', system-ui, sans-serif;
}

/* ── App background: transparent so Three.js shows through ────────── */
.stApp {
  background: transparent !important;
}
/* Block-level wrappers Streamlit generates */
section[data-testid="stMain"] > div,
.block-container {
  background: transparent !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: rgba(8, 0, 16, 0.88) !important;
  border-right: 1px solid var(--card-border);
}
section[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
section[data-testid="stSidebar"] .stRadio label {
  cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover span {
  color: var(--accent) !important;
}

/* ── Glass card helper ────────────────────────────────────────────── */
/* Usage in Python: st.markdown('<div class="card">...</div>', unsafe_allow_html=True) */
.card {
  background:    var(--card-bg);
  border:        1px solid var(--card-border);
  border-radius: 10px;
  padding:       18px 22px;
  margin-bottom: 14px;
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
}

/* ── Stat badge ───────────────────────────────────────────────────── */
.stat-badge {
  display: inline-block;
  background: rgba(200, 30, 60, 0.18);
  border: 1px solid var(--card-border);
  border-radius: 8px;
  padding: 10px 16px;
  text-align: center;
  min-width: 120px;
}
.stat-badge .val {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--accent);
  display: block;
  line-height: 1.2;
}
.stat-badge .lbl {
  font-size: 0.72rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 4px;
  display: block;
}

/* ── Phase pill ───────────────────────────────────────────────────── */
.phase-pill {
  display: inline-block;
  background: rgba(180, 20, 50, 0.28);
  border: 1px solid var(--card-border);
  border-radius: 20px;
  padding: 3px 12px;
  font-size: 0.75rem;
  color: var(--accent);
  font-weight: 600;
  letter-spacing: 0.04em;
  margin-right: 6px;
}

/* ── Alert / callout ──────────────────────────────────────────────── */
.callout {
  border-left: 3px solid var(--accent);
  background: rgba(180, 20, 50, 0.12);
  border-radius: 0 6px 6px 0;
  padding: 10px 14px;
  margin: 10px 0;
  color: var(--text);
  font-size: 0.88rem;
}

/* ── Headings ──────────────────────────────────────────────────────── */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  color: var(--text) !important;
  font-family: var(--font);
}
h1 { font-size: 2rem !important; }
h2 { color: var(--accent) !important; font-size: 1.35rem !important; }
h3 { font-size: 1.1rem !important; }

/* ── Body text ─────────────────────────────────────────────────────── */
p, li, span, label, .stMarkdown {
  color: var(--text) !important;
  font-family: var(--font);
}

/* ── Streamlit table ───────────────────────────────────────────────── */
.stDataFrame, .stTable {
  background: var(--card-bg) !important;
  border-radius: 8px;
  border: 1px solid var(--card-border);
}
.stDataFrame thead th {
  background: rgba(180, 20, 50, 0.25) !important;
  color: var(--accent) !important;
}
.stDataFrame tbody tr:hover td {
  background: rgba(200, 30, 60, 0.1) !important;
}

/* ── Slider / widgets ─────────────────────────────────────────────── */
.stSlider > div > div > div > div {
  background: var(--accent) !important;
}
.stSlider [data-testid="stThumbValue"] {
  color: var(--accent) !important;
}

/* ── Plotly container ─────────────────────────────────────────────── */
.stPlotlyChart {
  background: transparent !important;
}
.js-plotly-plot .plotly .bg {
  fill: transparent !important;
}

/* ── Image caption ────────────────────────────────────────────────── */
.stImage > div > figcaption {
  color: var(--muted) !important;
  font-size: 0.8rem;
}

/* ── Missing-data message ─────────────────────────────────────────── */
.missing {
  border: 1px dashed rgba(200, 30, 60, 0.4);
  border-radius: 8px;
  padding: 18px;
  text-align: center;
  color: var(--muted);
  font-size: 0.9rem;
  background: rgba(10, 0, 18, 0.5);
}

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(200, 30, 60, 0.4);
  border-radius: 3px;
}

/* ── st.tabs ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(10, 0, 18, 0.6) !important;
  border-radius: 8px 8px 0 0;
  border-bottom: 1px solid var(--card-border);
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}

/* ── Section divider ─────────────────────────────────────────────── */
hr { border-color: rgba(200, 30, 60, 0.25) !important; }
"""


def inject_css() -> None:
    """Call once at app startup to apply the Red Sea theme."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def card(content_html: str) -> None:
    """Render `content_html` inside a glass card div."""
    st.markdown(f'<div class="card">{content_html}</div>',
                unsafe_allow_html=True)


def missing(message: str = "Data not yet generated — run the training script first.") -> None:
    """Show a styled placeholder for missing output files."""
    st.markdown(f'<div class="missing">⚓ {message}</div>',
                unsafe_allow_html=True)


def phase_pill(label: str) -> str:
    """Return HTML for a phase pill badge (inline use in markdown)."""
    return f'<span class="phase-pill">{label}</span>'


def stat_badge(value: str, label: str) -> str:
    """Return HTML for a stat badge (inline use in markdown)."""
    return (f'<div class="stat-badge">'
            f'<span class="val">{value}</span>'
            f'<span class="lbl">{label}</span>'
            f'</div>')
