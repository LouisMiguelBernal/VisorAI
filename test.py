import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import base64
import warnings
from ultralytics import YOLO
import time
# ---------------- SETUP ----------------
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set Streamlit page config
st.set_page_config(page_title="VisorAI", layout="wide", page_icon="assets/icon.png")

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Convert logo to base64
logo_base64 = get_base64_image('assets/icon.png')

# Streamlit UI: Logo & Title
st.markdown(
    f"""
    <div style="display: flex; align-items: center; padding-top: 50px;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 100px; height: auto; margin-right: 10px;">
        <h1 style="margin: 0;">Visor<span style="color:#4CAF50;">AI</span></h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Style Tabs
st.markdown("""
    <style>
    .stTabs [role="tablist"] button {
        font-size: 1.2rem;
        padding: 12px 24px;
        margin-right: 10px;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
    }
    .stTabs [role="tablist"] button[aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- SESSION STATE INIT ----------------
if "last_detected_classes" not in st.session_state:
    st.session_state.last_detected_classes = set()
if "last_detection_time" not in st.session_state:
    st.session_state.last_detection_time = 0
if "last_detected_label" not in st.session_state:
    st.session_state.last_detected_label = None
if "screenshots" not in st.session_state:
    st.session_state.screenshots = []

# ---------------- AUDIO FILES ----------------
SOUND_FILES = {
    "Bicycle Lane": "assets/Bicycle_Lane.mp3",
    "Broken and Solid Yellow Lines": "assets/Broken_and_Solid_Yellow_Lines.mp3",
    "Bus Lane": "assets/Bus_Lane.mp3",
    "Cats Eye": "assets/Cats_Eye.mp3",
    "Continuity Lane": "assets/Continuity_Lane.mp3",
    "Double Solid Yellow or White Line": "assets/Double_Solid_Yellow_or_White_Line.mp3",
    "Holding Lane": "assets/Holding_Lane.mp3",
    "Loading and Unloading Zone": "assets/Loading_and_Unloading_Zone.mp3",
    "Motorcycle Lane": "assets/Motorcycle_Lane.mp3",
    "No Loading and Unloading Curb": "assets/No_Loading_and_Unloading_Curb.mp3",
    "No Parking Curb": "assets/No_Parking_Curb.mp3",
    "Parking Bay": "assets/Parking_Bay.mp3",
    "Pavement Arrow": "assets/Pavement_Arrow.mp3",
    "Pedestrian Lane": "assets/Pedestrian_Lane.mp3",
    "Railroad Crossing": "assets/Railroad_Crossing.mp3",
    "Rumble Strips": "assets/Rumble_Strips.mp3",
    "Single Solid Line": "assets/Single_Solid_Line.mp3",
    "Speed Limit": "assets/Speed_Limit.mp3",
    "Transition Line": "assets/Transition_Line.mp3"
}

# ---------------- DEFINITION TEXTS ----------------
DEFINITIONS = {
    "Bicycle Lane": "A designated lane on the road for bicycle riders only.",
    "Broken and Solid Yellow Lines": "Indicates passing rules depending on your side.",
    "Bus Lane": "Lane reserved for buses.",
    # Add other definitions accordingly
}

# ---------------- AUDIO FUNCTION ----------------
def autoplay_audio(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            """
            st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Audio error: {e}")

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    model_path = "assets/visorai.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    return YOLO(model_path)

model = load_model()
if model is None:
    st.stop()

# ---------------- DETECTION FUNCTION ----------------
def detect_from_frame(frame):
    results = model(frame, verbose=False)
    img_with_boxes = results[0].plot(conf=True)

    detected_classes = set()
    for box in results[0].boxes:
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        if confidence > 0.3 and class_name in SOUND_FILES:
            detected_classes.add(class_name)

    return img_with_boxes, detected_classes


# ---------------- LIVE DETECTION UI ----------------
st.title("🚦 Live Road Feature Detection")

run_live = st.toggle("Enable Live Camera Detection")

FRAME_CAPTURE_INTERVAL = 2  # seconds
screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

if run_live:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run_live and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_img, detected = detect_from_frame(frame_rgb)

        # Display live stream
        stframe.image(result_img, channels="RGB", use_container_width=True)

        # Take screenshot on new detection
        new_detections = detected - st.session_state.last_detected_classes
        if new_detections:
            detected_label = list(new_detections)[0]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_path = f"{screenshot_dir}/{detected_label}_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, result_img)
            st.session_state.screenshots.append((screenshot_path, detected_label))
            st.session_state.last_detected_label = detected_label
            st.session_state.last_detection_time = time.time()

            if detected_label in SOUND_FILES:
                autoplay_audio(SOUND_FILES[detected_label])

        st.session_state.last_detected_classes = detected

        # Break loop after short interval
        time.sleep(0.1)

    cap.release()

# ---------------- DISPLAY DETECTED FEATURES ----------------
if st.session_state.screenshots:
    st.subheader("📸 Detected Features")

    for path, label in st.session_state.screenshots[::-1]:  # Show latest first
        with st.expander(f"🛑 {label}"):
            st.image(path, caption=label, use_container_width=True)
            st.markdown(f"**Definition:** {DEFINITIONS.get(label, 'Definition not available.')}")

# ---------------- CLEANUP RESET ----------------
if st.button("🔄 Clear Screenshots"):
    st.session_state.screenshots = []
    st.session_state.last_detected_classes = set()
    st.session_state.last_detected_label = None
    st.success("Cleared screenshots and detections.")
        

# ---------------- FOOTER ----------------
footer = f"""
<hr>
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; padding: 10px 0;">
  <div style="flex-grow: 1; text-align: left;">
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 100px; margin-right: 10px;">
        <h1 style="margin: 0;">Visor<span style="color:#4CAF50;">AI</span></h1>
    </div>
  </div>
  <div style="flex-grow: 1; text-align: center;">
    <span>Copyright 2024 | All Rights Reserved</span>
  </div>
  <div style="flex-grow: 1; text-align: right;">
</div>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
