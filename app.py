"""
app.py
======
EEG Emotion AI — Streamlit Dashboard

Tabs:
  🏠 Home / Overview
  📊 Model Comparison  (shows saved metrics)
  🔮 DEAP Demo         (run inference on DEAP or synthetic data)
  📁 Custom EEG        (upload .npy / .csv / .dat)
  ⚡ Real-time Stream  (simulated live stream)

Run: streamlit run app.py
"""

import os
import time
import pickle
import numpy as np
import streamlit as st

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="EEG Emotion AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ───────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"], section.main {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(46, 160, 67, 0.4);
}
[data-testid="stTabs"] [role="tab"] {
    background: #161b22 !important;
    color: #8b949e !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid #30363d !important;
    font-weight: 600;
    padding: 8px 16px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1f6feb !important;
    color: #ffffff !important;
    border-color: #1f6feb !important;
}
[data-testid="stProgress"] > div > div { background-color: #1f6feb !important; }
[data-testid="stExpander"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
[data-testid="stAlert"] { border-radius: 8px !important; }
.stCodeBlock { background-color: #161b22 !important; }
[data-testid="stDataFrame"] { background: #161b22 !important; }
h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ── Lazy imports ─────────────────────────────────
@st.cache_resource
def get_imports():
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    return pd, go, px

pd, go, px = get_imports()

# ──────────────────────────────────────────────
# HARDCODED MODEL METRICS (from your training run)
# ──────────────────────────────────────────────
SAVED_METRICS = {
    "valence": pd.DataFrame([
        {"Model": "XGBoost",      "RMSE": 1.6727, "R2": 0.3761, "MAE": 1.3627, "Train Time (s)": 29.9,   "Best": "★"},
        {"Model": "SVM",          "RMSE": 1.6882, "R2": 0.3645, "MAE": 1.2881, "Train Time (s)": 1930.1, "Best": ""},
        {"Model": "RandomForest", "RMSE": 1.8660, "R2": 0.2236, "MAE": 1.5681, "Train Time (s)": 6.1,    "Best": ""},
    ]),
    "arousal": pd.DataFrame([
        {"Model": "XGBoost",      "RMSE": 1.5181, "R2": 0.4180, "MAE": 1.2221, "Train Time (s)": 29.1,   "Best": "★"},
        {"Model": "SVM",          "RMSE": 1.5367, "R2": 0.4036, "MAE": 1.1624, "Train Time (s)": 1968.7, "Best": ""},
        {"Model": "RandomForest", "RMSE": 1.6941, "R2": 0.2751, "MAE": 1.3926, "Train Time (s)": 5.8,    "Best": ""},
    ]),
    "emotion": pd.DataFrame([
        {"Model": "SVM",          "Accuracy": 0.5584, "F1": 0.5609, "Train Time (s)": 18.3, "Best": "★"},
        {"Model": "XGBoost",      "Accuracy": 0.5555, "F1": 0.5546, "Train Time (s)": 0.7,  "Best": ""},
        {"Model": "RandomForest", "Accuracy": 0.5539, "F1": 0.5553, "Train Time (s)": 0.6,  "Best": ""},
    ]),
    "focus": pd.DataFrame([
        {"Model": "XGBoost",      "RMSE": 3.5631, "R2": 0.9634, "MAE": 2.5873, "Train Time (s)": 2.8,   "Best": "★"},
        {"Model": "RandomForest", "RMSE": 4.6307, "R2": 0.9381, "MAE": 3.3841, "Train Time (s)": 5.1,   "Best": ""},
        {"Model": "SVM",          "RMSE": 3.9261, "R2": 0.9555, "MAE": 2.5027, "Train Time (s)": 496.4, "Best": ""},
    ]),
}

BEST_MODELS = {
    "valence": "XGBoost",
    "arousal": "XGBoost",
    "emotion": "SVM",
    "focus":   "XGBoost",
}

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
def init_state():
    defaults = {
        "predictor":      None,
        "models_loaded":  False,
        "last_result":    None,
        "stream_results": [],
        "load_error":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ──────────────────────────────────────────────
# AUTO-LOAD MODELS ON STARTUP
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor_cached():
    """Load saved models once, cache across reruns."""
    try:
        from predict import EEGPredictor
        p = EEGPredictor().load()
        return p, None
    except Exception as e:
        import traceback
        return None, traceback.format_exc()

if not st.session_state.models_loaded:
    predictor, err = load_predictor_cached()
    st.session_state.predictor     = predictor
    st.session_state.models_loaded = True
    st.session_state.load_error    = err

# ──────────────────────────────────────────────
# PLOTLY DARK THEME
# ──────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    margin=dict(l=40, r=20, t=40, b=40),
)

def _dark_layout(**overrides) -> dict:
    layout = {k: (dict(v) if isinstance(v, dict) else v)
              for k, v in DARK_LAYOUT.items()}
    for key, val in overrides.items():
        if key in ("xaxis", "yaxis") and isinstance(val, dict):
            layout[key] = {**layout.get(key, {}), **val}
        else:
            layout[key] = val
    return layout

def dark_fig(fig):
    fig.update_layout(**DARK_LAYOUT)
    return fig

# ──────────────────────────────────────────────
# VISUALIZATION HELPERS
# ──────────────────────────────────────────────

def plot_eeg_signal(eeg: np.ndarray, fs: int = 128,
                    n_ch: int = 8, title: str = "EEG Signal") -> go.Figure:
    n_channels, n_samples = eeg.shape
    n_ch = min(n_ch, n_channels)
    t    = np.linspace(0, n_samples / fs, n_samples)
    colors = ["#58a6ff", "#3fb950", "#f78166", "#ffa657",
              "#d2a8ff", "#79c0ff", "#56d364", "#ff7b72"]
    fig    = go.Figure()
    offset = 0
    for ch in range(n_ch):
        fig.add_trace(go.Scatter(
            x=t, y=eeg[ch] + offset,
            mode="lines", name=f"Ch {ch+1}",
            line=dict(color=colors[ch % len(colors)], width=1.2),
        ))
        offset += 3
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e6edf3")),
        xaxis_title="Time (s)", yaxis_title="Amplitude (offset)",
        showlegend=True, height=350, **DARK_LAYOUT,
    )
    return fig


def plot_band_powers(band_powers: dict) -> go.Figure:
    bands  = list(band_powers.keys())
    powers = list(band_powers.values())
    colors = ["#d2a8ff", "#79c0ff", "#58a6ff", "#3fb950", "#ffa657"]
    max_p  = max(powers) if max(powers) > 0 else 1
    norm_p = [p / max_p * 100 for p in powers]
    freq_labels = {
        "delta": "δ Delta (0.5–4 Hz)", "theta": "θ Theta (4–8 Hz)",
        "alpha": "α Alpha (8–13 Hz)",  "beta":  "β Beta (13–30 Hz)",
        "gamma": "γ Gamma (30–45 Hz)",
    }
    fig = go.Figure(go.Bar(
        x=norm_p, y=[freq_labels.get(b, b) for b in bands],
        orientation="h", marker=dict(color=colors[:len(bands)]),
        text=[f"{p:.1f}%" for p in norm_p], textposition="auto",
    ))
    fig.update_layout(
        title=dict(text="Brainwave Band Powers", font=dict(size=14, color="#e6edf3")),
        xaxis_title="Relative Power (%)", height=280, **DARK_LAYOUT,
    )
    return fig


def plot_valence_arousal(v: float, a: float,
                          history_v: list = None,
                          history_a: list = None) -> go.Figure:
    fig = go.Figure()
    quad_colors = {
        "Happy\n(High V, High A)": (5, 9, 5, 9, "rgba(255,215,0,0.08)"),
        "Calm\n(High V, Low A)":   (5, 9, 1, 5, "rgba(144,238,144,0.08)"),
        "Stress\n(Low V, High A)": (1, 5, 5, 9, "rgba(255,99,71,0.08)"),
        "Sad\n(Low V, Low A)":     (1, 5, 1, 5, "rgba(135,206,235,0.08)"),
    }
    for label, (x0, x1, y0, y1, clr) in quad_colors.items():
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=clr, line=dict(width=0), layer="below")
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=label,
                           showarrow=False, font=dict(size=9, color="#8b949e"))
    fig.add_hline(y=5, line=dict(color="#30363d", dash="dash", width=1))
    fig.add_vline(x=5, line=dict(color="#30363d", dash="dash", width=1))
    if history_v and history_a and len(history_v) > 1:
        fig.add_trace(go.Scatter(x=history_v, y=history_a, mode="lines",
                                  line=dict(color="#30363d", width=1),
                                  name="History", opacity=0.4))
    fig.add_trace(go.Scatter(x=[v], y=[a], mode="markers+text",
                              marker=dict(size=18, color="#f78166",
                                          line=dict(color="#ff7b72", width=2)),
                              text=["You"], textposition="top center",
                              name="Current"))
    fig.update_layout(**_dark_layout(
        title=dict(text="Valence–Arousal Circumplex", font=dict(size=14, color="#e6edf3")),
        xaxis=dict(title="Valence (1–9)", range=[1, 9]),
        yaxis=dict(title="Arousal (1–9)", range=[1, 9]),
        height=320, showlegend=False,
    ))
    return fig


def plot_model_comparison(cdfs: dict) -> dict:
    figs = {}
    for task, df in cdfs.items():
        primary = "RMSE" if task in ("valence", "arousal", "focus") else "F1"
        if primary not in df.columns:
            continue
        colors = ["#3fb950" if b == "★" else "#58a6ff" for b in df["Best"]]
        fig = go.Figure(go.Bar(
            x=df["Model"], y=df[primary],
            marker_color=colors,
            text=[f"{v:.4f}" for v in df[primary]],
            textposition="auto",
        ))
        fig.update_layout(
            title=dict(text=f"{task.title()} — {primary}", font=dict(size=13, color="#e6edf3")),
            yaxis_title=primary, height=280, **DARK_LAYOUT,
        )
        figs[task] = fig
    return figs


def plot_timeseries(values: list, label: str, color: str = "#58a6ff",
                    y_range: tuple = None) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=values, mode="lines+markers",
        line=dict(color=color, width=2), marker=dict(size=5), name=label,
    ))
    y_extra = dict(range=y_range) if y_range else {}
    fig.update_layout(**_dark_layout(
        title=dict(text=label, font=dict(size=13, color="#e6edf3")),
        xaxis_title="Window #", yaxis_title=label,
        height=220,
        yaxis=y_extra,
    ))
    return fig


def _fmt(val, fmt=".2f", fallback="?"):
    """Safely format a value; return fallback string if val is not numeric."""
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return fallback


def render_prediction_cards(result: dict):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        v     = result.get("valence", {})
        v_val = v.get("value", v.get("mean", None))
        st.metric("💚 Valence", f"{_fmt(v_val)} / 9",
                  delta=v.get("interpretation", ""))
    with c2:
        a     = result.get("arousal", {})
        a_val = a.get("value", a.get("mean", None))
        st.metric("⚡ Arousal", f"{_fmt(a_val)} / 9",
                  delta=a.get("interpretation", ""))
    with c3:
        e     = result.get("emotion", {})
        label = e.get("label") or e.get("dominant") or ""
        if not label or label in ("?", "Unknown"):
            probs = e.get("mean_probs") or e.get("all_probs") or {}
            label = max(probs, key=probs.get) if probs else "Unknown"
        emoji = e.get("emoji", "") or ""
        if not emoji:
            _emap = {"Happy": "😄", "Calm": "😌", "Stress": "😤", "Sad": "😢"}
            emoji = _emap.get(label, "")
        conf   = e.get("confidence", None)
        conf_s = f"Conf: {_fmt(conf, '.0%', '')}" if isinstance(conf, (int, float)) else ""
        color  = e.get("color", "#888") or "#888"
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;
                    border-radius:12px;padding:14px 18px;">
            <div style="color:#8b949e;font-size:0.85rem;">🎭 Emotion</div>
            <div style="font-size:1.8rem;font-weight:700;color:{color};">
                {emoji} {label}
            </div>
            <div style="color:#3fb950;font-size:0.8rem;">{conf_s}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        f_dict    = result.get("focus", {})
        score_raw = f_dict.get("score") if f_dict.get("score") is not None else f_dict.get("mean")
        score     = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0
        cat       = f_dict.get("category", "")
        if not cat:
            cat = "High 🎯" if score >= 70 else "Medium 🔆" if score >= 40 else "Low 😴"
        pct       = max(0, min(100, int(score)))
        bar_color = "#3fb950" if score >= 70 else "#ffa657" if score >= 40 else "#f78166"
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;
                    border-radius:12px;padding:14px 18px;">
            <div style="color:#8b949e;font-size:0.85rem;">🎯 Focus Score</div>
            <div style="font-size:1.8rem;font-weight:700;color:#58a6ff;">
                {score:.1f}<span style="font-size:1rem;color:#8b949e;">/100</span>
            </div>
            <div style="background:#21262d;border-radius:6px;height:8px;margin:6px 0;">
                <div style="width:{pct}%;background:{bar_color};height:100%;
                            border-radius:6px;"></div>
            </div>
            <div style="color:#8b949e;font-size:0.8rem;">{cat}</div>
        </div>""", unsafe_allow_html=True)


def render_emotion_probs(result: dict):
    e     = result.get("emotion", {})
    probs = e.get("all_probs", {}) or e.get("mean_probs", {})
    if not probs:
        return
    st.markdown("**🎭 Emotion Probability Breakdown**")
    colors_map = {"Happy": "#FFD700", "Calm": "#90EE90",
                  "Stress": "#FF6347", "Sad": "#87CEEB"}
    emojis     = {"Happy": "😄", "Calm": "😌", "Stress": "😤", "Sad": "😢"}
    for emo, prob in probs.items():
        bar_w = int(prob * 100)
        col   = colors_map.get(emo, "#888")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin:4px 0;">
            <div style="width:80px;color:#e6edf3;font-size:0.9rem;">
                {emojis.get(emo,"")} {emo}
            </div>
            <div style="flex:1;background:#21262d;border-radius:6px;height:14px;">
                <div style="width:{bar_w}%;background:{col};height:100%;
                            border-radius:6px;"></div>
            </div>
            <div style="width:45px;text-align:right;color:#8b949e;font-size:0.8rem;">
                {prob:.1%}
            </div>
        </div>""", unsafe_allow_html=True)


def render_explanation(result: dict):
    explanations = result.get("explanation", [])
    if not explanations:
        return
    st.markdown("**💡 Why this prediction?**")
    for exp in explanations:
        st.markdown(f"""
        <div style="background:#161b22;border-left:3px solid #1f6feb;
                    padding:8px 14px;margin:5px 0;border-radius:0 6px 6px 0;
                    font-size:0.88rem;color:#e6edf3;">{exp}</div>
        """, unsafe_allow_html=True)


def render_timeseries(result: dict):
    if "valence" not in result or "timeseries" not in result.get("valence", {}):
        return
    st.markdown("### 📈 Temporal Analysis")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.plotly_chart(
            plot_timeseries(result["valence"]["timeseries"],
                            "Valence Over Time", "#3fb950", (1, 9)),
            use_container_width=True)
    with tc2:
        st.plotly_chart(
            plot_timeseries(result["arousal"]["timeseries"],
                            "Arousal Over Time", "#ffa657", (1, 9)),
            use_container_width=True)
    with tc3:
        if "focus" in result and "timeseries" in result["focus"]:
            st.plotly_chart(
                plot_timeseries(result["focus"]["timeseries"],
                                "Focus Over Time", "#58a6ff", (0, 100)),
                use_container_width=True)


def render_full_result(result: dict, fs: int = 128):
    """Shared helper: render all result panels after a prediction."""
    render_prediction_cards(result)
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if "raw_eeg" in result:
            st.plotly_chart(
                plot_eeg_signal(result["raw_eeg"], fs=fs,
                                title="EEG Signal (First Window)"),
                use_container_width=True)
        render_emotion_probs(result)
    with col_b:
        if "band_powers" in result:
            st.plotly_chart(plot_band_powers(result["band_powers"]),
                            use_container_width=True)
        v_val = result.get("valence", {}).get("mean",
                result.get("valence", {}).get("value", 5))
        a_val = result.get("arousal", {}).get("mean",
                result.get("arousal", {}).get("value", 5))
        st.plotly_chart(plot_valence_arousal(v_val, a_val),
                        use_container_width=True)
    st.markdown("---")
    render_explanation(result)
    render_timeseries(result)


# ──────────────────────────────────────────────
# HELPER — count trials in a DEAP .dat file
# ──────────────────────────────────────────────
def count_dat_trials(filepath: str) -> int:
    """Return the number of trials in a DEAP .dat file (default 40 on failure)."""
    try:
        import pickle as _pkl
        with open(filepath, "rb") as f:
            subject = _pkl.load(f, encoding="latin1")
        return int(subject["data"].shape[0])
    except Exception:
        return 40   # DEAP standard fallback


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 EEG Emotion AI")
    st.markdown("*Multi-Task Neural Decoding*")
    st.divider()

    if st.session_state.load_error:
        st.error("❌ Models failed to load")
        with st.expander("Error details"):
            st.code(st.session_state.load_error)
    elif st.session_state.predictor:
        st.success("✅ Models loaded & ready")
    else:
        st.warning("⚠️ No models found in `models/`")

    st.divider()
    st.markdown("### ⚙️ Demo Configuration")
    deap_folder   = st.text_input("DEAP Dataset Folder", value="./data",
                                  help="Path containing s01.dat … files")
    use_synthetic = st.checkbox("Use Synthetic Data (demo)",
                                value=not os.path.isdir(deap_folder),
                                help="Generate synthetic EEG if DEAP not available")
    st.divider()
    st.markdown("### 📊 Tasks")
    st.markdown("- 💚 **Valence** (regression)\n- ⚡ **Arousal** (regression)"
                "\n- 🎭 **Emotion** (4-class)\n- 🎯 **Focus** (0–100)")
    st.divider()
    st.markdown("### 🏆 Best Models")
    for task, model in BEST_MODELS.items():
        icons = {"valence": "💚", "arousal": "⚡", "emotion": "🎭", "focus": "🎯"}
        st.markdown(f"{icons.get(task,'')} **{task.title()}** → `{model}`")
    st.divider()
    st.markdown("*Built by Kiran Nambiar (24BYB1050) and Bhuvanesh Jatla (24BYB1113)*")


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "📊 Model Comparison",
    "🔮 DEAP Demo",
    "📁 Custom EEG",
    "⚡ Real-time Stream",
])

# ──────────────────────────────────────────────
# TAB 1 — HOME
# ──────────────────────────────────────────────
with tab1:
    st.markdown("# 🧠 EEG Emotion AI")
    st.markdown("### *Multi-Task Brain State Decoding — Two-Stage ML Pipeline*")
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### 🎯 What This System Does")
        st.markdown(
            "This end-to-end AI system decodes **emotional and cognitive states** directly "
            "from raw EEG (brainwave) signals using a scientifically grounded two-stage "
            "machine learning pipeline."
        )
        st.markdown("#### 🔁 Two-Stage Architecture")

        # ── Architecture diagram image ──────────────────────────
        # Look for the image next to app.py (the most natural location)
        _app_dir    = os.path.dirname(os.path.abspath(__file__))
        _arch_paths = [
            os.path.join(_app_dir, "Architecture_Diagram.png"),   # same folder as app.py
            "Architecture_Diagram.png",                            # cwd fallback
        ]
        _arch_img = next((p for p in _arch_paths if os.path.isfile(p)), None)

        if _arch_img:
            st.image(_arch_img, use_container_width=True)
        else:
            # Graceful fallback: show the text diagram so the app never crashes
            st.warning(
                "⚠️ `Architecture_Diagram.png` not found next to `app.py`. "
                "Place the image in the same folder to display it here."
            )
            st.code(
                "EEG Signal (32 channels × 768 samples)\n"
                "      ↓ [Preprocessing: notch + bandpass + z-score + windowing]\n"
                "      ↓ [Feature Extraction: PSD + Band Powers + Statistics + FAA]\n"
                "      ↓\n"
                "┌─────────────────── STAGE 1 ──────────────────────┐\n"
                "│  Valence Model (XGBoost ★) → score 1–9           │\n"
                "│  Arousal Model (XGBoost ★) → score 1–9           │\n"
                "└──────────────────────────────────────────────────┘\n"
                "      ↓ (predicted V + A as extra features)\n"
                "┌─────────────────── STAGE 2 ──────────────────────┐\n"
                "│  Emotion Model (SVM ★) → Happy / Calm / Stress / Sad │\n"
                "│  Focus Model   (XGBoost ★) → Score 0–100         │\n"
                "└──────────────────────────────────────────────────┘"
            )

    with col2:
        st.markdown("""
        #### 📈 Dataset
        - **DEAP Dataset** (primary)
        - 32 subjects × 40 trials
        - 32 EEG channels @ 128 Hz

        #### 🌊 Brainwave Bands
        | Band | Hz | Role |
        |------|----|------|
        | δ Delta | 0.5–4 | Sleep |
        | θ Theta | 4–8 | Memory |
        | α Alpha | 8–13 | Relaxation |
        | β Beta | 13–30 | Focus |
        | γ Gamma | 30–45 | Cognition |
        """)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📡 EEG Channels", "32")
    c2.metric("🔬 Sampling Rate", "128 Hz")
    c3.metric("🪟 Window Size", "6 sec")
    c4.metric("🤖 Models/Task", "3")

    st.divider()
    if st.session_state.predictor:
        st.success("✅ **Saved models are loaded** — head to any tab to run inference or explore results.")
    elif st.session_state.load_error:
        st.error(
            "❌ **Could not load models.** Make sure `models/` folder exists and contains "
            "trained `.pkl` files. Run `python train.py --synthetic` to train them first."
        )
    else:
        st.warning("⚠️ **No models found.** Run `python train.py --synthetic` in your terminal to train models first.")


# ──────────────────────────────────────────────
# TAB 2 — MODEL COMPARISON
# ──────────────────────────────────────────────
with tab2:
    st.markdown("## 📊 Model Comparison")
    st.markdown("Performance of all three models per task — best model highlighted in green.")
    st.divider()

    st.markdown("### 📈 Performance Charts")
    figs = plot_model_comparison(SAVED_METRICS)

    col1, col2 = st.columns(2)
    for i, (task, fig) in enumerate(figs.items()):
        (col1 if i % 2 == 0 else col2).plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 📋 Detailed Metrics")
    icons = {"valence": "💚", "arousal": "⚡", "emotion": "🎭", "focus": "🎯"}

    for task, df in SAVED_METRICS.items():
        with st.expander(f"{icons.get(task,'')} {task.title()} — Best: **{BEST_MODELS[task]}**",
                         expanded=True):
            try:
                if "RMSE" in df.columns:
                    styled = df.style.highlight_min(subset=["RMSE"], color="#1a3a1a")
                elif "F1" in df.columns:
                    styled = df.style.highlight_max(subset=["F1"], color="#1a3a1a")
                else:
                    styled = df.style
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(df, use_container_width=True)

    st.divider()
    st.markdown("### 🏆 Final System Configuration")
    for task, model in BEST_MODELS.items():
        df  = SAVED_METRICS[task]
        row = df[df["Model"] == model]
        if not row.empty:
            if "RMSE" in row.columns:
                metric_str = f"RMSE = {row['RMSE'].values[0]:.4f},  R² = {row['R2'].values[0]:.4f}"
            elif "F1" in row.columns:
                metric_str = f"F1 = {row['F1'].values[0]:.4f},  Accuracy = {row['Accuracy'].values[0]:.4f}"
            else:
                metric_str = ""
            st.markdown(f"- {icons.get(task,'')} **{task.title()}** → `{model}` ({metric_str})")


# ──────────────────────────────────────────────
# TAB 3 — DEAP DEMO
# ──────────────────────────────────────────────
with tab3:
    st.markdown("## 🔮 DEAP Demo Prediction")
    st.markdown("Run inference on a DEAP trial or synthetic EEG to see the full pipeline in action.")
    st.divider()

    if not st.session_state.predictor:
        st.error("❌ Models not loaded. Check the sidebar for details.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            if use_synthetic:
                st.info("ℹ️ Using **synthetic** DEAP-like data (no real DEAP path set).")
                deap_file_input = None
                trial_idx = st.slider("Trial Index", 0, 39, 0, key="deap_trial_slider")
            else:
                dat_files = []
                if os.path.isdir(deap_folder):
                    dat_files = sorted([f for f in os.listdir(deap_folder)
                                        if f.endswith(".dat")])
                if dat_files:
                    selected_file   = st.selectbox("Select DEAP Subject File", dat_files)
                    deap_file_input = os.path.join(deap_folder, selected_file)
                else:
                    st.warning(f"No `.dat` files found in `{deap_folder}`. Switching to synthetic.")
                    use_synthetic   = True
                    deap_file_input = None
                trial_idx = st.slider("Trial Index", 0, 39, 0, key="deap_trial_slider2")
        with col2:
            run_deap = st.button("▶ Run Prediction", use_container_width=True)

        if run_deap:
            with st.spinner("Running inference..."):
                try:
                    pred = st.session_state.predictor
                    if use_synthetic or deap_file_input is None:
                        from preprocessing import generate_synthetic_deap
                        X, yv, ya = generate_synthetic_deap(n_subjects=1,
                                                             n_trials_per_subject=40)
                        start   = trial_idx * 10
                        X_trial = X[start: start + 10]
                        result  = pred.predict_batch(X_trial, aggregate=True)
                        result["source"]       = f"Synthetic trial {trial_idx}"
                        result["ground_truth"] = {
                            "valence": round(float(yv[start]), 3),
                            "arousal": round(float(ya[start]), 3),
                        }
                        result["raw_eeg"] = X_trial[0]
                    else:
                        result = pred.predict_from_deap(deap_file_input, trial_idx)

                    st.session_state.last_result = result

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if st.session_state.last_result:
            result = st.session_state.last_result
            st.divider()
            st.markdown(f"**Source:** `{result.get('source', 'DEAP')}`")
            if "ground_truth" in result:
                gt = result["ground_truth"]
                st.markdown(f"**Ground Truth** — Valence: `{gt['valence']}` | Arousal: `{gt['arousal']}`")
            st.markdown("---")
            render_full_result(result)


# ──────────────────────────────────────────────
# TAB 4 — CUSTOM EEG
# ──────────────────────────────────────────────
with tab4:
    st.markdown("## 📁 Custom EEG Inference")
    st.markdown("Upload your own EEG file and run inference with the saved models.")
    st.divider()

    if not st.session_state.predictor:
        st.error("❌ Models not loaded. Check the sidebar for details.")
    else:
        st.markdown("""
        **Supported formats:**
        - `.npy` — NumPy array `(channels, samples)` or `(samples, channels)`
        - `.csv` — CSV with rows=samples, columns=channels (no header)
        - `.dat` — DEAP pickle format — a **Trial Index** selector will appear after upload
        """)

        uploaded  = st.file_uploader("Upload EEG File", type=["npy", "csv", "dat"],
                                      key="custom_eeg_uploader")
        custom_fs = st.number_input("Sampling Rate (Hz)", min_value=64,
                                     max_value=2048, value=128)

        if uploaded is not None:
            import tempfile

            suffix = "." + uploaded.name.split(".")[-1].lower()
            is_dat = (suffix == ".dat")

            # Save uploaded bytes to a persistent temp file so we can
            # (a) inspect trial count for .dat files, and
            # (b) pass a real filepath to the predictor.
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            # ── .dat-specific: trial selector ────────────────────
            custom_trial_idx = 0
            if is_dat:
                n_trials = count_dat_trials(tmp_path)
                st.info(f"ℹ️ DEAP `.dat` file detected — **{n_trials} trial(s)** found.")
                custom_trial_idx = st.slider(
                    "Select Trial Index",
                    min_value=0,
                    max_value=max(0, n_trials - 1),
                    value=0,
                    key="custom_dat_trial_slider",
                    help="Each trial is one 60-second recording segment (0-indexed).",
                )

            if st.button("🔍 Analyze Custom EEG", key="analyze_custom_btn"):
                with st.spinner("Analyzing…"):
                    try:
                        pred = st.session_state.predictor

                        if is_dat:
                            # Reuse the exact same DEAP pipeline used in Tab 3,
                            # but on the user-supplied file and chosen trial.
                            result = pred.predict_from_deap(
                                tmp_path, trial_idx=custom_trial_idx
                            )
                            result["source"] = (
                                f"Custom .dat — `{uploaded.name}` — trial {custom_trial_idx}"
                            )
                        else:
                            result = pred.predict_from_file(tmp_path, fs=int(custom_fs))
                            result.setdefault("source", f"Custom — `{uploaded.name}`")

                        st.divider()
                        st.markdown(f"**Source:** `{result.get('source', uploaded.name)}`")

                        if "ground_truth" in result:
                            gt = result["ground_truth"]
                            st.markdown(
                                f"**Ground Truth (from file)** — "
                                f"Valence: `{gt['valence']}` | Arousal: `{gt['arousal']}`"
                            )

                        st.markdown("---")
                        render_full_result(result, fs=int(custom_fs))

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                    finally:
                        # Clean up temp file after analysis is done
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

        else:
            st.info("💡 No file yet? Use the **DEAP Demo** tab, or generate a synthetic demo below.")
            if st.button("🎲 Generate & Analyze Demo EEG", key="demo_eeg_btn"):
                with st.spinner("Generating synthetic EEG…"):
                    try:
                        from preprocessing import generate_synthetic_deap
                        X, _, _ = generate_synthetic_deap(n_subjects=1,
                                                           n_trials_per_subject=1)
                        result  = st.session_state.predictor.predict_batch(
                            X[:15], aggregate=True)
                        result["raw_eeg"] = X[0]
                        result["source"]  = "Synthetic demo"
                        st.divider()
                        st.success("✅ Demo EEG generated and analyzed!")
                        render_full_result(result)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())


# ──────────────────────────────────────────────
# TAB 5 — REAL-TIME STREAM
# ──────────────────────────────────────────────
with tab5:
    st.markdown("## ⚡ Simulated Real-Time EEG Stream")
    st.markdown("Simulates a continuous EEG stream and updates predictions in real-time.")
    st.divider()

    if not st.session_state.predictor:
        st.error("❌ Models not loaded. Check the sidebar for details.")
    else:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        n_chunks = col_ctrl1.slider("Number of Chunks", 5, 30, 10)
        delay_ms = col_ctrl2.slider("Update Delay (ms)", 100, 1000, 300)
        start_rt = col_ctrl3.button("▶ Start Stream", use_container_width=True)

        if start_rt:
            st.markdown("---")
            st.markdown("### 📡 Live Predictions")

            ph_status    = st.empty()
            ph_cards     = st.empty()
            ph_cols      = st.columns(3)
            ph_val_chart = ph_cols[0].empty()
            ph_aro_chart = ph_cols[1].empty()
            ph_foc_chart = ph_cols[2].empty()
            ph_va_circ   = st.empty()

            val_hist, aro_hist, foc_hist = [], [], []

            from preprocessing import generate_synthetic_deap
            X_stream, _, _ = generate_synthetic_deap(
                n_subjects=1,
                n_trials_per_subject=max(1, n_chunks // 10 + 1),
                duration_sec=max(n_chunks * 4, 60),
            )

            for i in range(min(n_chunks, len(X_stream))):
                ph_status.markdown(f"**📡 Processing chunk {i+1}/{n_chunks}…**")

                chunk  = X_stream[i]
                result = st.session_state.predictor.predict_realtime_chunk(chunk)

                v = result.get("valence", {}).get("value", 5)
                a = result.get("arousal", {}).get("value", 5)
                f = result.get("focus",   {}).get("score", 50)

                val_hist.append(v)
                aro_hist.append(a)
                foc_hist.append(f)

                with ph_cards.container():
                    render_prediction_cards(result)

                ph_val_chart.plotly_chart(
                    dark_fig(plot_timeseries(val_hist, "Valence", "#3fb950", (1, 9))),
                    use_container_width=True)
                ph_aro_chart.plotly_chart(
                    dark_fig(plot_timeseries(aro_hist, "Arousal", "#ffa657", (1, 9))),
                    use_container_width=True)
                ph_foc_chart.plotly_chart(
                    dark_fig(plot_timeseries(foc_hist, "Focus", "#58a6ff", (0, 100))),
                    use_container_width=True)
                ph_va_circ.plotly_chart(
                    plot_valence_arousal(v, a, val_hist[:-1], aro_hist[:-1]),
                    use_container_width=True)

                time.sleep(delay_ms / 1000)

            ph_status.success(f"✅ Stream complete — {n_chunks} chunks processed")
            st.session_state.stream_results = {
                "val_hist": val_hist,
                "aro_hist": aro_hist,
                "foc_hist": foc_hist,
            }