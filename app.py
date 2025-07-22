import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # Replace with 'best.pt' if using a custom-trained model

st.set_page_config(page_title="DehydALTE", layout="centered")
st.title("DehydALTE")
st.markdown("Detect vegetables.")

# Select input type
option = st.radio("Select input type:", ["📸 Capture from Webcam", "🖼 Upload an Image"])

image = None

if option == "📸 Capture from Webcam":
    cam_input = st.camera_input("Take a picture")
    if cam_input is not None:
        image = Image.open(cam_input).convert("RGB")

elif option == "🖼 Upload an Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# Run detection
if image is not None:
    st.subheader("Detected Vegetables:")
    image_np = np.array(image)
    results = model(image_np)
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Vegetable Detection", use_column_width=True)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0]) * 100
        st.write(f"🟩 {label} - {confidence:.2f}%")

