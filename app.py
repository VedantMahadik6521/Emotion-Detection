import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN
import plotly.express as px

# ------------------ CONSTANTS ------------------
RAF_DB_EMOTIONS = [
    "😲 Surprise",
    "😨 Fear",
    "🤢 Disgust",
    "😊 Happiness",
    "😢 Sadness",
    "😠 Anger",
    "😐 Neutral",
]

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# ------------------ LOAD FACE DETECTOR ------------------
@st.cache_resource
def load_face_detector():
    return MTCNN()

# ------------------ PREPROCESS IMAGE ------------------
def preprocess_image(image, target_size=(100, 100)):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 2:  # grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# ------------------ FACE DETECTION ------------------
def detect_faces(image, detector):
    if isinstance(image, Image.Image):
        image = np.array(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
    results = detector.detect_faces(rgb_image)
    faces = []
    for res in results:
        x, y, w, h = res["box"]
        padding = int(max(w, h) * 0.2)
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(rgb_image.shape[1], x + w + padding), min(rgb_image.shape[0], y + h + padding)
        face_crop = rgb_image[y1:y2, x1:x2]
        faces.append((face_crop, (x1, y1, x2, y2)))
    return faces

# ------------------ PREDICT EMOTION ------------------
def predict_emotion(model, image):
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    return RAF_DB_EMOTIONS[predicted_class], confidence, predictions[0]

# ------------------ STREAMLIT APP ------------------
st.set_page_config(page_title="Emotion Detection 🎭", page_icon="😊", layout="wide")

st.title("🎭 Real-Time Emotion Detection")
st.write("Detect emotions from faces using CNN + MTCNN")

# Sidebar
st.sidebar.header("⚙️ Model Configuration")
model_file = st.sidebar.file_uploader("📂 Upload .keras model", type=["keras"])

if not model_file:
    st.sidebar.warning("⚠️ Please upload a `.keras` model to begin")
    st.info("📥 Upload your trained model in the sidebar")
    st.stop()

# Save uploaded model
with open("temp_model.keras", "wb") as f:
    f.write(model_file.getbuffer())

model = load_model("temp_model.keras")

# Set input target size
if len(model.input_shape) == 4:
    target_h, target_w = model.input_shape[1], model.input_shape[2]
    target_size = (target_w, target_h)
else:
    target_size = (100, 100)

detector = load_face_detector()

# ------------------ Tabs ------------------
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Image", "🎥 Webcam Mode", "📊 Analytics", "ℹ️ About"])

# ============= TAB 1: UPLOAD IMAGE =============
with tab1:
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        if st.button("🚀 Predict Emotions", type="primary", use_container_width=True):
            with st.spinner("⏳ Processing..."):
                faces = detect_faces(np.array(image), detector)
                if not faces:
                    st.warning("⚠️ No face detected, using full image")
                    faces = [(np.array(image), (0, 0, image.size[0], image.size[1]))]

                for idx, (face_crop, (x1, y1, x2, y2)) in enumerate(faces, 1):
                    processed = preprocess_image(face_crop, target_size)
                    emotion, confidence, all_preds = predict_emotion(model, processed)

                    st.markdown(f"### 👤 Face {idx}")
                    st.image(face_crop, caption=f"Detected Face {idx}", use_column_width=True)
                    st.write(f"🎯 **Predicted Emotion:** {emotion}")
                    st.metric("🔥 Confidence", f"{confidence:.1%}")

                    # Plotly bar chart
                    fig = px.bar(
                        x=[float(p) for p in all_preds],
                        y=RAF_DB_EMOTIONS,
                        orientation="h",
                        labels={"x": "Probability", "y": "Emotion"},
                        text=[f"{p:.1%}" for p in all_preds],
                        title="Emotion Probabilities"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 2: WEBCAM MODE =============
with tab2:
    st.write("Enable real-time webcam emotion detection 👇")
    run_camera = st.checkbox("🎥 Enable Webcam")

    if run_camera:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = detect_faces(frame, detector)
            for face_crop, (x1, y1, x2, y2) in faces:
                processed = preprocess_image(face_crop, target_size)
                emotion, confidence, _ = predict_emotion(model, processed)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion} {confidence:.1%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frame_placeholder.image(frame, channels="BGR")

# ============= TAB 3: ANALYTICS =============
with tab3:
    st.write("📊 In a real deployment, you could log predictions here")
    st.info("Future idea: store results in CSV/DB and plot emotion trends over time.")

# ============= TAB 4: ABOUT =============
with tab4:
    st.markdown("""
    ### ℹ️ About This App
    - Built with **TensorFlow + MTCNN + Streamlit**
    - Detects multiple faces and predicts emotions
    - Supports both **image upload** and **real-time webcam**
    - Shows explainable probability charts for each emotion

    💡 Created as a demo project to showcase **AI + CV + Web Apps**.
    """)
