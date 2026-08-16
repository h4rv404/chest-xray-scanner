import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import gdown

st.set_page_config(
    page_title="RadScan AI — Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    * { font-family: 'DM Sans', sans-serif !important; }

    .stApp { background-color: #080c14; color: #e2e8f0; }
    footer, #MainMenu, header { display: none !important; visibility: hidden !important; }

    /* ── Header ── */
    .radscan-header {
        background: linear-gradient(180deg, #0d1220 0%, #080c14 100%);
        border-bottom: 1px solid #1a2235;
        padding: 2.5rem 3rem 2rem;
        margin: -1rem -1rem 2.5rem -1rem;
    }
    .radscan-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.2);
        color: #60a5fa;
        padding: 5px 14px; border-radius: 20px;
        font-size: 11px; font-weight: 600;
        letter-spacing: 1.2px; text-transform: uppercase;
        margin-bottom: 14px;
    }
    .radscan-title {
        font-size: 2.8rem; font-weight: 700;
        color: #f8faff; margin: 0; letter-spacing: -1px;
        line-height: 1.1;
    }
    .radscan-sub {
        color: #64748b; font-size: 1.05rem;
        margin-top: 8px; font-weight: 400;
    }

    /* ── Upload ── */
    div[data-testid="stFileUploader"] {
        background: #0d1220 !important;
        border: 2px dashed #1e3a5f !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 2rem;
        transition: border-color 0.3s;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #3b82f6 !important;
    }
    div[data-testid="stFileUploader"] label {
        font-size: 1rem !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* ── Primary Finding ── */
    .pf-card {
        background: #0d1220;
        border: 1px solid #1a2235;
        border-left: 4px solid #ef4444;
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 2rem;
        display: flex; align-items: center;
        justify-content: space-between;
    }
    .pf-label {
        font-size: 11px; color: #ef4444;
        text-transform: uppercase; letter-spacing: 1.5px;
        font-weight: 700; margin-bottom: 6px;
    }
    .pf-value {
        font-size: 1.8rem; font-weight: 700; color: #fff;
        letter-spacing: -0.5px;
    }
    .pf-badge {
        background: rgba(239,68,68,0.12);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.25);
        padding: 8px 20px; border-radius: 24px;
        font-size: 1rem; font-weight: 700;
        letter-spacing: -0.3px;
    }

    /* ── Image Cards ── */
    .img-card {
        background: #0d1220;
        border: 1px solid #1a2235;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 2rem;
    }
    .img-card-header {
        padding: 12px 18px;
        border-bottom: 1px solid #1a2235;
        font-size: 11px; font-weight: 700;
        color: #475569;
        text-transform: uppercase; letter-spacing: 1.5px;
        display: flex; align-items: center; gap: 8px;
    }
    .img-card-dot-white { width: 7px; height: 7px; background: #94a3b8; border-radius: 50%; }
    .img-card-dot-red { width: 7px; height: 7px; background: #ef4444; border-radius: 50%; }
    .img-card-body {
        padding: 20px;
        display: flex; justify-content: center;
        background: #060910;
    }
    .img-card-body img {
        border-radius: 10px;
        max-width: 100%;
    }

    /* ── Results Section ── */
    .results-card {
        background: #0d1220;
        border: 1px solid #1a2235;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
    }
    .results-title {
        font-size: 13px; font-weight: 700;
        color: #475569;
        text-transform: uppercase; letter-spacing: 1.5px;
        margin-bottom: 1.4rem;
        padding-bottom: 12px;
        border-bottom: 1px solid #1a2235;
        display: flex; align-items: center;
        justify-content: space-between;
    }
    .count-badge-red {
        background: rgba(239,68,68,0.1);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.2);
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 700;
    }
    .count-badge-green {
        background: rgba(34,197,94,0.08);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.15);
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 700;
    }

    /* ── Animated Bar Items ── */
    .finding-row {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid #0d1220;
        opacity: 0;
        animation: fadeSlideIn 0.5s ease forwards;
    }
    .finding-row:last-child { border-bottom: none; }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .f-dot-red { width: 8px; height: 8px; background: #ef4444; border-radius: 50%; flex-shrink: 0; }
    .f-dot-green { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; flex-shrink: 0; }

    .f-name {
        font-size: 0.95rem; font-weight: 500;
        color: #cbd5e1; min-width: 145px;
    }

    .f-bar-track {
        flex: 1; height: 7px;
        background: #1a2235;
        border-radius: 10px; overflow: hidden;
    }
    .f-bar-red {
        height: 100%; border-radius: 10px;
        background: linear-gradient(90deg, #b91c1c, #f87171);
        width: 0%;
        animation: growBar 1.2s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }
    .f-bar-green {
        height: 100%; border-radius: 10px;
        background: linear-gradient(90deg, #166534, #4ade80);
        width: 0%;
        animation: growBar 1.2s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }

    @keyframes growBar {
        from { width: 0%; }
        to { width: var(--target-width); }
    }

    .f-score-red {
        font-size: 0.9rem; font-weight: 700;
        color: #f87171; min-width: 42px; text-align: right;
        font-family: 'DM Mono', monospace !important;
    }
    .f-score-green {
        font-size: 0.9rem; font-weight: 700;
        color: #4ade80; min-width: 42px; text-align: right;
        font-family: 'DM Mono', monospace !important;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: rgba(234,179,8,0.05);
        border: 1px solid rgba(234,179,8,0.15);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-top: 2rem;
        display: flex; align-items: flex-start; gap: 10px;
    }
    .disclaimer-text {
        color: #ca8a04; font-size: 0.88rem;
        font-weight: 500; line-height: 1.5; margin: 0;
    }

    /* ── Empty State ── */
    .empty-state {
        text-align: center;
        padding: 6rem 2rem;
    }
    .empty-icon { font-size: 4rem; margin-bottom: 1.2rem; }
    .empty-title { font-size: 1.2rem; color: #64748b; font-weight: 600; }
    .empty-sub { font-size: 0.9rem; color: #334155; margin-top: 8px; }

    /* Animation delays for staggered effect */
    .finding-row:nth-child(1) { animation-delay: 0.05s; }
    .finding-row:nth-child(2) { animation-delay: 0.1s; }
    .finding-row:nth-child(3) { animation-delay: 0.15s; }
    .finding-row:nth-child(4) { animation-delay: 0.2s; }
    .finding-row:nth-child(5) { animation-delay: 0.25s; }
    .finding-row:nth-child(6) { animation-delay: 0.3s; }
    .finding-row:nth-child(7) { animation-delay: 0.35s; }
    .finding-row:nth-child(8) { animation-delay: 0.4s; }
</style>
""", unsafe_allow_html=True)

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural Thickening', 'Hernia'
]

MODEL_PATH = 'pretrained_model.h5'
GDRIVE_ID = '1qTEW8ftNXU7wIHaGIZVIrIYEWEaGwkGQ'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model..."):
            gdown.download(f'https://drive.google.com/uc?id={GDRIVE_ID}', MODEL_PATH, quiet=False)
    base_model = DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(14, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(MODEL_PATH)
    return model

def generate_heatmap(model, img_array, pred_index):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer = layer
            break
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(np.array(heatmap), 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return heatmap

def overlay_heatmap(img_path, heatmap):
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (400, 400))
    heatmap_resized = cv2.resize(np.array(heatmap), (400, 400))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_img, 0.55, heatmap_colored, 0.45, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

def make_bar_row(label, score, color):
    pct = int(score * 100)
    dot = f'<div class="f-dot-{color}"></div>'
    bar = f'<div class="f-bar-{color}" style="--target-width:{pct}%"></div>'
    score_html = f'<div class="f-score-{color}">{pct}%</div>'
    return f"""
    <div class="finding-row">
        {dot}
        <div class="f-name">{label}</div>
        <div class="f-bar-track">{bar}</div>
        {score_html}
    </div>
    """

# ── Header ──
st.markdown("""
<div class="radscan-header">
    <div class="radscan-badge">AI-Powered · DenseNet121 · 14 Conditions</div>
    <div class="radscan-title">🫁 RadScan AI</div>
    <div class="radscan-sub">Upload a chest X-ray for instant AI-assisted pulmonary analysis</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Chest X-Ray (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    temp_path = "temp_xray.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Running AI analysis..."):
        model = load_model()
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0
        preds = model.predict(img_array)[0]
        top_index = np.argmax(preds)
        heatmap = generate_heatmap(model, img_array, top_index)
        result_img = overlay_heatmap(temp_path, heatmap)

    # Primary Finding
    st.markdown(f"""
    <div class="pf-card">
        <div>
            <div class="pf-label">Primary Finding</div>
            <div class="pf-value">{LABELS[top_index]}</div>
        </div>
        <div class="pf-badge">{preds[top_index]*100:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # Images — centered and bigger
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="img-card">
            <div class="img-card-header">
                <div class="img-card-dot-white"></div>
                Original X-Ray
            </div>
            <div class="img-card-body">
        """, unsafe_allow_html=True)
        st.image(uploaded_file, use_column_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="img-card">
            <div class="img-card-header">
                <div class="img-card-dot-red"></div>
                AI Heatmap — {LABELS[top_index]}
            </div>
            <div class="img-card-body">
        """, unsafe_allow_html=True)
        st.image(result_img, use_column_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    # Results
    detected = sorted([(l, s) for l, s in zip(LABELS, preds) if s > 0.5], key=lambda x: x[1], reverse=True)
    not_detected = sorted([(l, s) for l, s in zip(LABELS, preds) if s <= 0.5], key=lambda x: x[1], reverse=True)

    col_a, col_b = st.columns(2)

    with col_a:
        rows_html = "".join([make_bar_row(l, s, "red") for l, s in detected])
        st.markdown(f"""
        <div class="results-card">
            <div class="results-title">
                Detected
                <span class="count-badge-red">{len(detected)}</span>
            </div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        rows_html = "".join([make_bar_row(l, s, "green") for l, s in not_detected])
        st.markdown(f"""
        <div class="results-card">
            <div class="results-title">
                Not Detected
                <span class="count-badge-green">{len(not_detected)}</span>
            </div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <div class="disclaimer-text">
            ⚠️ For clinical assistance only. This AI analysis must be reviewed and confirmed
            by a qualified radiologist before any medical decision is made.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🫁</div>
        <div class="empty-title">Upload a chest X-ray to begin analysis</div>
        <div class="empty-sub">Supports JPG, JPEG, PNG · Max 200MB</div>
    </div>
    """, unsafe_allow_html=True)
