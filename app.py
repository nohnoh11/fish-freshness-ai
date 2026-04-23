import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Fish Freshness AI", page_icon="🐟", layout="centered")

# =========================
# CUSTOM UI STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #22c55e;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
}
.card {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}
.badge {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("keras_model.h5")

with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# =========================
# HEADER
# =========================
st.markdown('<div class="title">🐟 Fish Freshness AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detection System</div>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙ Controls")

mode = st.sidebar.radio("Mode", ["Upload Image", "Camera Capture"])
show_probs = st.sidebar.checkbox("Show Probabilities", True)

# =========================
# PREDICT FUNCTION
# =========================
def predict_image(image):
    img = np.array(image)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (224,224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    index = np.argmax(preds)

    return preds, index

# =========================
# IMAGE INPUT
# =========================
image = None

if mode == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if file:
        image = Image.open(file)

elif mode == "Camera Capture":
    cam = st.camera_input("Take Photo")
    if cam:
        image = Image.open(cam)

# =========================
# RESULT DISPLAY
# =========================
if image is not None:
    st.image(image, use_container_width=True)

    with st.spinner("Analyzing..."):
        preds, index = predict_image(image)

    confidence = float(preds[index]) * 100
    label = class_names[index]

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # RESULT BADGE
    if "Non" in label:
        st.markdown('<div class="badge" style="background:#ef4444;">❌ NON FRESH</div>', unsafe_allow_html=True)
    elif "Semi" in label:
        st.markdown('<div class="badge" style="background:#f59e0b;">⚠ SEMI FRESH</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge" style="background:#22c55e;">✅ FRESH</div>', unsafe_allow_html=True)

    st.progress(int(confidence))
    st.write(f"### Confidence: {confidence:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

    # PROBABILITIES
    if show_probs:
        st.subheader("📊 Class Probabilities")
        for i, prob in enumerate(preds):
            st.write(f"{class_names[i]} — {prob*100:.2f}%")
            st.progress(int(prob*100))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Fish Freshness Detection System")
