"""
app.py  –  DepthGuard: 3D Spatial Occupancy Monitor
=====================================================
Streamlit dashboard for real-time depth-based spatial occupancy detection.

Run with:
    streamlit run app.py
"""

import time
import logging
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd

# ---- local utils ----------------------------------------------------------
from utils.depth import estimate_depth
from utils.occupancy import detect_occupancy
from utils.visualization import (
    depth_to_colormap,
    draw_occupancy_overlay,
    draw_heatmap_overlay,
)

# ---- logging setup --------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/depthguard.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("DepthGuard")

# ===========================================================================
# Page config & custom CSS
# ===========================================================================
st.set_page_config(
    page_title="DepthGuard – Spatial Occupancy AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

/* ---- Global ---------------------------------------------------------- */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #0a0e1a;
    color: #c8d6e5;
}

/* ---- Sidebar: wider + proper padding --------------------------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1425 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2d42;
    min-width: 300px !important;
    width: 300px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.2rem !important;
}

/* ---- Sidebar text legibility ---------------------------------------- */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #d0dfef !important;
    font-size: 0.92rem !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-size: 1rem !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.4rem !important;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e3050;
}

/* ---- Sidebar section box --------------------------------------------- */
.sidebar-section {
    background: #111827;
    border: 1px solid #1e2d42;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 12px;
}
.sidebar-section-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #00d4ff !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    display: block;
}

/* ---- Slider label fix ------------------------------------------------ */
.stSlider > label {
    color: #b0c4d8 !important;
    font-size: 0.88rem !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: #5a7a9a !important;
}

/* ---- Selectbox -------------------------------------------------------- */
.stSelectbox > label { color: #b0c4d8 !important; font-size: 0.88rem !important; }

/* ---- Radio buttons --------------------------------------------------- */
.stRadio > label { color: #b0c4d8 !important; font-size: 0.88rem !important; }
.stRadio div[role="radiogroup"] label { color: #d0dff0 !important; font-size: 0.9rem !important; }

/* ---- Alert badges ----------------------------------------------------- */
.badge-safe     { display:block; text-align:center; background:#0a3020; color:#32e68a; border:1px solid #32e68a; padding:10px 18px; border-radius:6px; font-weight:700; font-size:1.2rem; margin-bottom:10px; }
.badge-warning  { display:block; text-align:center; background:#2a1e00; color:#ffaa00; border:1px solid #ffaa00; padding:10px 18px; border-radius:6px; font-weight:700; font-size:1.2rem; margin-bottom:10px; }
.badge-critical { display:block; text-align:center; background:#2a0008; color:#ff3355; border:1px solid #ff3355; padding:10px 18px; border-radius:6px; font-weight:700; font-size:1.2rem; margin-bottom:10px; animation:pulse 0.8s infinite alternate; }

@keyframes pulse { from { box-shadow:0 0 4px #ff3355; } to { box-shadow:0 0 18px #ff3355; } }

/* ---- Metric cards ----------------------------------------------------- */
.metric-card {
    background:#111827;
    border:1px solid #1e2d42;
    border-radius:8px;
    padding:12px 16px;
    margin-bottom:8px;
}
.metric-label { font-size:0.72rem; color:#7a90a8; text-transform:uppercase; letter-spacing:1px; }
.metric-value { font-family:'Share Tech Mono', monospace; font-size:1.5rem; color:#e0eaf5; }

/* ---- Title ------------------------------------------------------------ */
.dg-title {
    font-family:'Exo 2', sans-serif;
    font-weight:800;
    font-size:2.2rem;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing:3px;
}
.dg-subtitle { color:#5a7a9a; font-size:0.85rem; letter-spacing:3px; text-transform:uppercase; }

/* ---- Main content area ----------------------------------------------- */
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session state
# ===========================================================================
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []   # list of dicts
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "fps_history" not in st.session_state:
    st.session_state.fps_history = []


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    # ── Logo / title ────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:10px 0 16px 0;'>
        <div style='font-size:2rem;'>🛡️</div>
        <div style='font-size:1.2rem; font-weight:800; color:#00d4ff;
                    letter-spacing:2px; font-family:Exo 2,sans-serif;'>
            DEPTH GUARD
        </div>
        <div style='font-size:0.7rem; color:#4a6a8a; letter-spacing:2px;
                    text-transform:uppercase; margin-top:2px;'>
            Spatial Occupancy AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 1. Input Source ─────────────────────────────────────────────────
    st.markdown("""<div class='sidebar-section'>
        <span class='sidebar-section-title'>📷  Input Source</span>
    </div>""", unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose input:",
        ["📁 Upload Video", "🖼️ Upload Image", "📸 Webcam (live)"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 2. Depth Model ──────────────────────────────────────────────────
    st.markdown("""<div class='sidebar-section'>
        <span class='sidebar-section-title'>🧠  Depth Model</span>
    </div>""", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "MiDaS variant",
        ["MiDaS_small  ⚡ Fast", "DPT_Hybrid  ⚖️ Balanced", "DPT_Large  🎯 Best"],
        index=0,
    )
    # Strip the display label to get actual model name
    model_choice = model_choice.split("  ")[0]

    st.caption("⚡ MiDaS_small → best for CPU demo  |  🎯 DPT_Large → best accuracy on GPU")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 3. Detection Thresholds ─────────────────────────────────────────
    st.markdown("""<div class='sidebar-section'>
        <span class='sidebar-section-title'>⚠️  Detection Thresholds</span>
    </div>""", unsafe_allow_html=True)

    st.caption("Higher value = object must be closer to trigger alert")

    warn_thresh = st.slider(
        "🟡  Warning level",
        min_value=0.30, max_value=0.90, value=0.65, step=0.01,
    )
    critical_thresh = st.slider(
        "🔴  Critical level",
        min_value=warn_thresh + 0.05, max_value=0.99, value=0.82, step=0.01,
    )
    critical_pct = st.slider(
        "🔴  Critical if area >  (% of frame)",
        min_value=1, max_value=50, value=15, step=1,
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 4. Visualisation ────────────────────────────────────────────────
    st.markdown("""<div class='sidebar-section'>
        <span class='sidebar-section-title'>🎨  Visualisation</span>
    </div>""", unsafe_allow_html=True)

    show_heatmap   = st.checkbox("🌡️  Proximity heatmap overlay", value=False)
    show_depth_map = st.checkbox("🗺️  Show depth map panel",      value=True)

    colourmap_name = st.selectbox(
        "Depth colour scheme",
        ["INFERNO", "JET", "TURBO", "MAGMA", "PLASMA"],
        index=0,
    )
    COLOURMAP = getattr(cv2, f"COLORMAP_{colourmap_name}")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 5. Performance ──────────────────────────────────────────────────
    st.markdown("""<div class='sidebar-section'>
        <span class='sidebar-section-title'>⚙️  Performance</span>
    </div>""", unsafe_allow_html=True)

    max_dim = st.slider(
        "Max frame size (px)",
        320, 720, 480, 32,
    )
    st.caption("👆 Lower = faster. 480px is good for demo.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; font-size:0.7rem; color:#2a4060;
                border-top:1px solid #1a2a3a; padding-top:10px;'>
        MiDaS · PyTorch · OpenCV · Streamlit
    </div>""", unsafe_allow_html=True)


# ===========================================================================
# Header
# ===========================================================================
st.markdown('<p class="dg-title">🛡️ DEPTH GUARD</p>', unsafe_allow_html=True)
st.markdown('<p class="dg-subtitle">3D Spatial Occupancy · Industrial Safety AI · Edge Vision</p>',
            unsafe_allow_html=True)
st.markdown("---")


# ===========================================================================
# Helper: process a single frame
# ===========================================================================
def process_frame(frame_bgr: np.ndarray, model_type: str):
    """
    Run depth estimation + occupancy detection on one BGR frame.
    Returns (annotated_frame, depth_colour, heatmap_frame, result, elapsed_ms).
    """
    # Resize for speed
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

    t0 = time.time()
    depth_map = estimate_depth(frame_bgr, model_type=model_type)
    elapsed_ms = (time.time() - t0) * 1000

    result = detect_occupancy(
        depth_map,
        warn_threshold=warn_thresh,
        critical_threshold=critical_thresh,
        critical_area_pct=critical_pct / 100,
    )

    annotated = draw_occupancy_overlay(frame_bgr, result)

    if show_heatmap:
        annotated = draw_heatmap_overlay(annotated, depth_map, alpha=0.35)

    depth_colour = depth_to_colormap(depth_map, COLOURMAP)

    return annotated, depth_colour, result, elapsed_ms


# ===========================================================================
# Helper: render metrics panel
# ===========================================================================
def render_metrics(result, elapsed_ms, frame_count):
    # Alert badge
    badge_classes = {
        "SAFE":     "badge-safe",
        "WARNING":  "badge-warning",
        "CRITICAL": "badge-critical",
    }
    icons = {"SAFE": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}
    badge_cls = badge_classes[result.alert_level]
    icon = icons[result.alert_level]

    st.markdown(
        f'<div class="{badge_cls}">{icon} {result.alert_level}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Occupied Area</div>
            <div class="metric-value">{result.occupied_ratio*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Closest Object</div>
            <div class="metric-value">{result.min_depth*100:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Inference Time</div>
            <div class="metric-value">{elapsed_ms:.0f} ms</div>
        </div>""", unsafe_allow_html=True)

        fps_est = 1000 / elapsed_ms if elapsed_ms > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Est. FPS</div>
            <div class="metric-value">{fps_est:.1f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Frames Processed</div>
        <div class="metric-value">{frame_count}</div>
    </div>""", unsafe_allow_html=True)

    # Detected blobs
    if result.bounding_boxes:
        st.info(f"🔍 {len(result.bounding_boxes)} occupied region(s) detected")


# ===========================================================================
# Helper: log alert
# ===========================================================================
def log_alert(result, frame_count):
    if result.alert_level != "SAFE":
        entry = {
            "Frame": frame_count,
            "Alert": result.alert_level,
            "Occ%": f"{result.occupied_ratio*100:.1f}",
            "Closest": f"{result.min_depth*100:.0f}%",
        }
        st.session_state.alert_history.append(entry)
        log.warning("Frame %d | %s | Occ %.1f%%", frame_count,
                    result.alert_level, result.occupied_ratio * 100)


# ===========================================================================
# ---- MODE: Upload Image ---------------------------------------------------
# ===========================================================================
if input_mode == "🖼️ Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        with st.spinner("Running depth estimation …"):
            annotated, depth_colour, result, elapsed_ms = process_frame(frame_bgr, model_choice)

        st.session_state.frame_count += 1
        log_alert(result, st.session_state.frame_count)

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown("#### 📸 Annotated Frame")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col2:
            if show_depth_map:
                st.markdown("#### 🌈 Depth Map")
                st.image(cv2.cvtColor(depth_colour, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col3:
            st.markdown("#### 📊 Metrics")
            render_metrics(result, elapsed_ms, st.session_state.frame_count)


# ===========================================================================
# ---- MODE: Upload Video ---------------------------------------------------
# ===========================================================================
elif input_mode == "📁 Upload Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        # Save to temp file
        tmp_path = f"/tmp/depthguard_input_{int(time.time())}.mp4"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25

        st.info(f"Video: {total_frames} frames @ {fps_src:.1f} fps")

        # Frame skip slider
        frame_skip = st.slider("Process every Nth frame", 1, 10, 3,
                               help="Higher = faster but less smooth.")

        if st.button("▶️ Run Analysis"):
            progress = st.progress(0)
            col1, col2, col3 = st.columns([2, 2, 1])
            frame_ph    = col1.empty()
            depth_ph    = col2.empty()
            metrics_ph  = col3.empty()
            alert_ph    = st.empty()

            frame_idx = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                annotated, depth_colour, result, elapsed_ms = process_frame(frame_bgr, model_choice)
                st.session_state.frame_count += 1
                log_alert(result, st.session_state.frame_count)

                frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               caption="Annotated", use_column_width=True)
                if show_depth_map:
                    depth_ph.image(cv2.cvtColor(depth_colour, cv2.COLOR_BGR2RGB),
                                   caption="Depth Map", use_column_width=True)
                with metrics_ph.container():
                    render_metrics(result, elapsed_ms, st.session_state.frame_count)

                progress.progress(min(frame_idx / max(total_frames, 1), 1.0))

            cap.release()
            st.success("✅ Video analysis complete!")


# ===========================================================================
# ---- MODE: Webcam ---------------------------------------------------
# ===========================================================================
elif input_mode == "📸 Webcam (live)":
    st.warning(
        "⚠️ **Webcam mode** requires Streamlit to run **locally**. "
        "Click **Start** below to begin streaming from your default camera."
    )
    col_start, col_stop = st.columns(2)
    start_btn = col_start.button("▶️ Start Webcam")
    stop_btn  = col_stop.button("⏹ Stop")

    col1, col2, col3 = st.columns([2, 2, 1])
    frame_ph   = col1.empty()
    depth_ph   = col2.empty()
    metrics_ph = col3.empty()

    if start_btn:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Check that a camera is connected.")
        else:
            st.session_state["webcam_running"] = True
            while st.session_state.get("webcam_running", False):
                ret, frame_bgr = cap.read()
                if not ret:
                    st.error("Camera read failed.")
                    break

                annotated, depth_colour, result, elapsed_ms = process_frame(
                    frame_bgr, model_choice
                )
                st.session_state.frame_count += 1
                log_alert(result, st.session_state.frame_count)

                frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               caption="Live Feed", use_column_width=True)
                if show_depth_map:
                    depth_ph.image(cv2.cvtColor(depth_colour, cv2.COLOR_BGR2RGB),
                                   caption="Depth Map", use_column_width=True)
                with metrics_ph.container():
                    render_metrics(result, elapsed_ms, st.session_state.frame_count)

            cap.release()

    if stop_btn:
        st.session_state["webcam_running"] = False
        st.info("Webcam stopped.")


# ===========================================================================
# Alert log panel
# ===========================================================================
st.markdown("---")
st.markdown("### 🗂️ Alert Log")

if st.session_state.alert_history:
    df = pd.DataFrame(st.session_state.alert_history)
    # Colour rows
    def color_alert(val):
        colors = {"WARNING": "background-color:#2a1e00; color:#ffaa00",
                  "CRITICAL": "background-color:#2a0008; color:#ff3355"}
        return colors.get(val, "")

    styled = df.style.applymap(color_alert, subset=["Alert"])
    st.dataframe(styled, use_container_width=True)

    if st.button("🗑️ Clear log"):
        st.session_state.alert_history = []
        st.experimental_rerun()
else:
    st.info("No alerts triggered yet. Run analysis to populate this log.")


# ===========================================================================
# Footer
# ===========================================================================
st.markdown("---")
st.markdown(
    "<center style='color:#2a4060; font-size:0.8rem;'>"
    "DepthGuard · 3D Spatial Occupancy AI · Powered by MiDaS + PyTorch + Streamlit"
    "</center>",
    unsafe_allow_html=True,
)