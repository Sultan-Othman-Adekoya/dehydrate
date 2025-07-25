
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO("/content/best (1).pt")

# WebRTC config (optional STUN server)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="DehydrALTE Vegetable Detector", layout="centered")
st.title("DehydrALTE Vegetable Detector")
st.markdown("Detect vegetables (like tomatoes, peppers, onions) with bounding boxes and accuracy in real time.")

# Choose input mode
option = st.radio("Choose Input Mode", ["🎥 Live Webcam Detection", "🖼 Upload Image"])

# Video processor class for WebRTC
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.1%}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

if option == "🎥 Live Webcam Detection":
    webrtc_streamer(
        key="vegetable-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

elif option == "🖼 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "jfif", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_np = np.array(image)
        results = model(image_np)[0]
        res_plotted = results.plot()

        st.image(res_plotted, caption="Detected Vegetables", use_container_width=True)

        st.subheader("Detections:")
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0]) * 100
            st.write(f"🟩 {label} - {confidence:.2f}%")
