
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained or custom)
model = YOLO("yolov8n.pt")  # Replace with 'best.pt' if you trained a custom model

st.title("🥕 DehydLTE Vegetable Detector")
st.write("Upload an image to detect vegetables like peppers with bounding boxes and accuracy.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Inference
    results = model(image_np)

    # Plot results
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Detected Vegetables", use_column_width=True)

    # Show predictions
    st.subheader("Detections:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0]) * 100
        st.write(f"🟩 {label} - {confidence:.2f}%")
