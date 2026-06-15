"""VoiceShield: Audio Deepfake Detector -- Streamlit dashboard.

A dark, Aceternity/shadcn-inspired UI for the CNN-LSTM deepfake audio
detector. Run with:

    streamlit run app/app.py

Sections
--------
1. Hero header with animated grid background.
2. Drag-and-drop audio upload + native audio player.
3. Inference readout: classification badge + animated confidence
   gauges + LFCC feature heatmap.
4. Model analytics overlay: ROC/EER curve, confusion matrix, and the full
   verification metrics report produced by `src/models/train.py`.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    CHECKPOINT_PATH,
    METRICS_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    ROC_CURVE_PATH,
    CLASS_NAMES,
)
from src.features.feature_extraction import extract_features_from_bytes, normalize
from src.models.model import CNNLSTMDetector

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VoiceShield: Audio Deepfake Detector",
    page_icon="\U0001F6E1️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_GAUGE_IDS = itertools.count()


# ---------------------------------------------------------------------------
# Aceternity / shadcn-inspired styling
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp {
            background-color: #09090b;
        }

        /* ---- Animated Aceternity-style grid background ---- */
        .aceternity-grid {
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background-image:
                linear-gradient(to right, rgba(148,163,184,0.08) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(148,163,184,0.08) 1px, transparent 1px);
            background-size: 42px 42px;
            -webkit-mask-image: radial-gradient(ellipse 75% 55% at 50% 0%, #000 60%, transparent 100%);
            mask-image: radial-gradient(ellipse 75% 55% at 50% 0%, #000 60%, transparent 100%);
            animation: grid-pan 40s linear infinite;
        }
        @keyframes grid-pan {
            0%   { background-position: 0px 0px, 0px 0px; }
            100% { background-position: 420px 420px, 420px 420px; }
        }
        .aceternity-glow {
            position: fixed;
            top: -22%;
            left: 50%;
            transform: translateX(-50%);
            width: 70vw;
            height: 70vw;
            max-width: 900px;
            max-height: 900px;
            background: radial-gradient(circle, rgba(124,58,237,0.30) 0%, rgba(124,58,237,0) 70%);
            z-index: 0;
            pointer-events: none;
            animation: glow-pulse 7s ease-in-out infinite;
        }
        @keyframes glow-pulse {
            0%, 100% { opacity: 0.55; transform: translateX(-50%) scale(1); }
            50%      { opacity: 1;    transform: translateX(-50%) scale(1.06); }
        }

        /* Make sure normal content paints above the background layers */
        .block-container, [data-testid="stMainBlockContainer"] { position: relative; z-index: 1; padding-top: 2.5rem; max-width: 800px !important; margin: 0 auto; }
        [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }

        /* ---- Header ---- */
        .gradient-title {
            font-size: 3.1rem;
            font-weight: 800;
            text-align: center;
            letter-spacing: -0.03em;
            background: linear-gradient(180deg, #ffffff 10%, #a1a1aa 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            text-align: center;
            color: #a1a1aa;
            font-size: 1.05rem;
            margin-bottom: 2.5rem;
        }
        .subtitle .accent { color: #a78bfa; font-weight: 600; }

        /* ---- shadcn-style glass card ---- */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 1.1rem;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
            backdrop-filter: blur(8px);
            padding: 1.75rem 1.85rem;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 12px 40px rgba(0,0,0,0.45);
            margin-bottom: 1.5rem;
        }
        .card-title {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #a1a1aa;
            margin-bottom: 0.9rem;
        }

        /* ---- Dropzone override ---- */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.02) !important;
            border: 1.5px dashed rgba(124,58,237,0.45) !important;
            border-radius: 1rem !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(167,139,250,0.8) !important;
        }

        /* ---- Verdict badges ---- */
        .verdict-row { display: flex; justify-content: center; margin: 0.5rem 0 1.6rem 0; }
        .df-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.65rem 1.6rem;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 1.15rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            animation: badge-in 0.5s cubic-bezier(0.22,1,0.36,1);
        }
        @keyframes badge-in {
            from { opacity: 0; transform: scale(0.85) translateY(6px); }
            to   { opacity: 1; transform: scale(1) translateY(0); }
        }
        .df-badge.genuine {
            background: rgba(34,197,94,0.12);
            color: #4ade80;
            border: 1px solid rgba(34,197,94,0.45);
            box-shadow: 0 0 30px rgba(34,197,94,0.25);
        }
        .df-badge.deepfake {
            background: rgba(244,63,94,0.12);
            color: #fb7185;
            border: 1px solid rgba(244,63,94,0.45);
            box-shadow: 0 0 30px rgba(244,63,94,0.25);
        }

        /* ---- Confidence gauges ---- */
        .gauge-wrap { margin-bottom: 1.1rem; }
        .gauge-label {
            display: flex; justify-content: space-between;
            font-size: 0.85rem; color: #d4d4d8; margin-bottom: 0.4rem; font-weight: 600;
        }
        .gauge-label .pct { color: #fafafa; font-weight: 800; }
        .gauge-track {
            width: 100%; height: 12px; border-radius: 9999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.06);
            overflow: hidden;
        }
        .gauge-fill {
            position: relative; height: 100%; border-radius: 9999px; width: 0%;
            overflow: hidden;
        }
        .gauge-fill::after {
            content: ""; position: absolute; inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.30), transparent);
            animation: shimmer 2.2s linear infinite;
        }
        @keyframes shimmer {
            0%   { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        /* ---- Metric pills (analytics overlay) ---- */
        .metric-pill {
            border-radius: 0.85rem;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            padding: 0.9rem 1rem;
            text-align: center;
        }
        .metric-pill .val { font-size: 1.6rem; font-weight: 800; color: #fafafa; }
        .metric-pill .lbl { font-size: 0.72rem; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }
        .metric-pill.ok .val { color: #4ade80; }
        .metric-pill.bad .val { color: #fb7185; }

        footer, #MainMenu { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_background() -> None:
    st.markdown(
        '<div class="aceternity-glow"></div><div class="aceternity-grid"></div>',
        unsafe_allow_html=True,
    )


def render_gauge(label: str, value: float, color_from: str, color_to: str) -> None:
    """Render a smoothly animated confidence gauge (0..1 -> 0..100%)."""
    pct = max(0.0, min(1.0, value)) * 100.0
    gid = next(_GAUGE_IDS)
    st.markdown(
        f"""
        <div class="gauge-wrap">
          <div class="gauge-label"><span>{label}</span><span class="pct">{pct:5.1f}%</span></div>
          <div class="gauge-track">
            <div class="gauge-fill gauge-fill-{gid}"
                 style="background: linear-gradient(90deg, {color_from}, {color_to});"></div>
          </div>
        </div>
        <style>
          @keyframes fill-{gid} {{ from {{ width: 0%; }} to {{ width: {pct:.2f}%; }} }}
          .gauge-fill-{gid} {{ animation: fill-{gid} 1.2s cubic-bezier(0.22, 1, 0.36, 1) forwards,
                                          shimmer 2.2s linear infinite; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_verdict_badge(label: int, prob_fake: float) -> None:
    if label == 1:
        st.markdown(
            f"""<div class="verdict-row">
                  <span class="df-badge deepfake">&#9888;&#65039; {CLASS_NAMES[1]}</span>
                </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="verdict-row">
                  <span class="df-badge genuine">&#9989; {CLASS_NAMES[0]}</span>
                </div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_checkpoint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    arch = checkpoint["architecture"]
    model = CNNLSTMDetector(
        input_dim=arch["input_dim"],
        cnn_channels=tuple(arch["cnn_channels"]),
        lstm_hidden=arch["lstm_hidden"],
        lstm_layers=arch["lstm_layers"],
        bidirectional=arch["bidirectional"],
        dropout=arch["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


@st.cache_data(show_spinner=False)
def load_metrics_report():
    if not METRICS_REPORT_PATH.exists():
        return None
    with open(METRICS_REPORT_PATH) as f:
        return json.load(f)


def run_inference(audio_bytes: bytes, suffix: str):
    model, checkpoint, device = load_model_and_checkpoint()
    features = extract_features_from_bytes(audio_bytes, suffix=suffix)
    norm_features = normalize(features, checkpoint["norm_mean"], checkpoint["norm_std"])
    x = torch.from_numpy(norm_features).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_fake = torch.sigmoid(model(x)).item()
    threshold = checkpoint.get("decision_threshold", 0.5)
    return features, prob_fake, threshold


def render_feature_heatmap(features: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.6))
    fig.patch.set_facecolor("#0c0c11")
    ax.set_facecolor("#0c0c11")
    im = ax.imshow(features, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Frame", color="#a1a1aa", fontsize=9)
    ax.set_ylabel("Coefficient", color="#a1a1aa", fontsize=9)
    ax.set_title("LFCC + Δ + ΔΔ Feature Map", color="#fafafa", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#71717a", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#27272a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(colors="#71717a", labelsize=7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
inject_css()
render_background()

st.markdown('<div class="gradient-title">VoiceShield: Audio Deepfake Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">CNN-LSTM acoustic analysis over '
    '<span class="accent">LFCC + Δ + ΔΔ</span> biomarkers '
    '&mdash; trained on the FoR (Fake-or-Real) corpus</p>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="card-title">Upload Audio Evidence</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop a .wav or .mp3 file here, or click to browse",
        type=["wav", "mp3"],
        label_visibility="visible",
    )

model_ready = CHECKPOINT_PATH.exists()
if not model_ready:
    st.warning(
        f"No trained checkpoint found at `{CHECKPOINT_PATH}`. "
        f"Run `python -m src.models.train` to train the detector first."
    )

if uploaded_file is not None and model_ready:
    audio_bytes = uploaded_file.read()
    suffix = Path(uploaded_file.name).suffix or ".wav"

    with st.container(border=True):
        st.markdown('<div class="card-title">Audio Playback</div>', unsafe_allow_html=True)
        st.audio(audio_bytes)

    with st.spinner("Running model inference…"):
        features, prob_fake, threshold = run_inference(audio_bytes, suffix)

    prob_genuine = 1.0 - prob_fake
    predicted_label = 1 if prob_fake >= threshold else 0

    with st.container(border=True):
        st.markdown('<div class="card-title">Inference Readout</div>', unsafe_allow_html=True)
        render_verdict_badge(predicted_label, prob_fake)

        gauge_col1, gauge_col2 = st.columns(2)
        with gauge_col1:
            render_gauge(CLASS_NAMES[0], prob_genuine, "#16a34a", "#4ade80")
        with gauge_col2:
            render_gauge(CLASS_NAMES[1], prob_fake, "#e11d48", "#fb7185")

        render_feature_heatmap(features)

# ---------------------------------------------------------------------------
# Model Analytics Overlay
# ---------------------------------------------------------------------------
with st.expander("Model Analytics & Performance Report", expanded=False):
    report = load_metrics_report()
    if report is None:
        st.info("No metrics report found yet. Run `python -m src.models.train` to generate one.")
    else:
        eval_split = report["test"]
        targets = report["targets"]

        st.markdown("#### Held-out Test Set Verification")
        m1, m2, m3, m4 = st.columns(4)
        for col, (label, value, target, fmt) in zip(
            (m1, m2, m3, m4),
            [
                ("Accuracy", eval_split["accuracy"], targets["accuracy"], "{:.2%}"),
                ("F1-Score", eval_split["f1"], targets["f1"], "{:.2%}"),
                ("EER", eval_split["eer"], targets["eer"], "{:.2%}"),
                ("ROC AUC", eval_split["auc"], None, "{:.3f}"),
            ],
        ):
            ok = True if target is None else (
                value <= target if label == "EER" else value >= target
            )
            css_class = "ok" if ok else "bad"
            with col:
                st.markdown(
                    f"""<div class="metric-pill {css_class}">
                          <div class="val">{fmt.format(value)}</div>
                          <div class="lbl">{label}</div>
                        </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("#### Per-Class Accuracy")
        pc1, pc2 = st.columns(2)
        for col, (cls_name, acc) in zip((pc1, pc2), eval_split["per_class_accuracy"].items()):
            ok = acc >= targets["per_class_accuracy"]
            css_class = "ok" if ok else "bad"
            with col:
                st.markdown(
                    f"""<div class="metric-pill {css_class}">
                          <div class="val">{acc:.2%}</div>
                          <div class="lbl">{cls_name}</div>
                        </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("#### ROC Curve & Confusion Matrix")
        img1, img2 = st.columns(2)
        with img1:
            if ROC_CURVE_PATH.exists():
                st.image(str(ROC_CURVE_PATH), use_container_width=True)
        with img2:
            if CONFUSION_MATRIX_PATH.exists():
                st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)



        with st.expander("Training Configuration"):
            st.json(report["config"])
