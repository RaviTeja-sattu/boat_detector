import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where app.py lives
IMAGE_DIR = os.path.join(BASE_DIR, "images")           # absolute path to images folder

# === Setup ===
st.set_page_config(page_title="Boat Detector", layout="wide")

MODEL_PATHS = {
    "Boat (YOLO)": os.path.join(BASE_DIR, "models", "yolo_boat.pt"),
    "Horizon (U-Net)": os.path.join(BASE_DIR, "models", "unet_horizon_best.pth")  # placeholder
}

# Load image files from IMAGE_DIR
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# === Sidebar: Model Choice ===
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))

# === Load YOLO (cached) ===
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

# === Detection Logic ===
def detect_boats(image, model_path):
    model = load_yolo_model(model_path)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img_cv)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        label = f"boat {scores[i]:.2f}"
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_cv, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# === Image Selection Grid ===
st.title("🛥️ Boat Detector - Grid Image Selector")

selected_image_path = None
cols = st.columns(5)  # 5 columns grid

st.subheader("📸 Choose an Image")
for i, img_name in enumerate(image_files):
    with cols[i % 5]:  # cycle through columns
        img_path = os.path.join(IMAGE_DIR, img_name)
        image = Image.open(img_path)
        st.image(image, use_container_width=True)
        if st.button(f"Select {img_name}", key=img_name):
            selected_image_path = img_path
            st.session_state['selected_image'] = img_path  # store selection in session

# === Get Image from Session (if any) ===
if 'selected_image' in st.session_state:
    selected_image_path = st.session_state['selected_image']
    image = Image.open(selected_image_path).convert('RGB')
    st.subheader("🎯 Selected Image")
    st.image(image, caption=os.path.basename(selected_image_path), use_container_width=True)

    if st.button("🚀 Run Detection"):
        with st.spinner("Running model..."):
            if model_choice == "Boat (YOLO)":
                result = detect_boats(image, MODEL_PATHS[model_choice])
                st.image(result, caption="Detection Output", use_container_width=True)
            else:
                st.warning("Horizon detection (U-Net) not implemented yet.")
