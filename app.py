import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Fish Freshness AI", page_icon="🐟", layout="centered")

# =========================
# SESSION ANALYTICS
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

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
st.markdown('<div class="subtitle">Detection + Analytics System</div>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙ Controls")

mode = st.sidebar.radio("Mode", ["Upload Image", "Camera Capture", "Live Webcam"])
show_probs = st.sidebar.checkbox("Show Probabilities", True)

if st.sidebar.button("🔄 Reset Analytics"):
    st.session_state.history = []
    st.rerun()

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
# IMAGE / CAMERA INPUT
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
# IMAGE RESULT
# =========================
if image is not None:
    st.image(image, use_container_width=True)

    with st.spinner("Analyzing..."):
        preds, index = predict_image(image)

    confidence = float(preds[index]) * 100
    label = class_names[index]

    # SAVE TO ANALYTICS
    st.session_state.history.append({
        "label": label,
        "confidence": confidence
    })

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # RESULT DISPLAY
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
# LIVE WEBCAM
# =========================
if mode == "Live Webcam":
    st.warning("Click START to run webcam")

    start = st.button("▶ Start Webcam")
    FRAME_WINDOW = st.image([])

    if start:
        camera = cv2.VideoCapture(0)

        while True:
            ret, frame = camera.read()
            if not ret:
                st.error("Camera not working")
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            resized = cv2.resize(img, (224,224))
            normalized = resized.astype(np.float32) / 255.0
            input_img = np.expand_dims(normalized, axis=0)

            preds = model.predict(input_img, verbose=0)[0]
            index = np.argmax(preds)
            confidence = float(preds[index]) * 100
            label = class_names[index]

            # SAVE TO ANALYTICS
            st.session_state.history.append({
                "label": label,
                "confidence": confidence
            })

            # COLOR
            color = (0,255,0)
            if "Non" in label:
                color = (0,0,255)
            elif "Semi" in label:
                color = (0,165,255)

            text = f"{label} ({confidence:.1f}%)"

            cv2.putText(frame, text, (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            time.sleep(0.03)

# =========================
# ANALYTICS DASHBOARD
# =========================
st.markdown("## 📊 Analytics Dashboard")

history = st.session_state.history

if len(history) > 0:
    total = len(history)

    fresh = sum(1 for x in history if "Fresh" in x["label"] and "Non" not in x["label"])
    semi = sum(1 for x in history if "Semi" in x["label"])
    non = sum(1 for x in history if "Non" in x["label"])

    avg_conf = sum(x["confidence"] for x in history) / total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", total)
    col2.metric("Fresh", fresh)
    col3.metric("Semi", semi)
    col4.metric("Non", non)

    st.write(f"**Average Confidence:** {avg_conf:.2f}%")

    st.bar_chart({
        "Fresh": fresh,
        "Semi Fresh": semi,
        "Non Fresh": non
    })

else:
    st.info("No analytics data yet.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Fish Freshness Detection System with Analytics")
