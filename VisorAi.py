import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import base64
import warnings
from ultralytics import YOLO

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

# ------------------- STREAMLIT UI -------------------
detect, model_info = st.tabs(["Detection", "Model Information"])

# ---------------- SESSION STATE INIT ----------------
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "image_index" not in st.session_state:
    st.session_state.image_index = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "last_label" not in st.session_state:
    st.session_state.last_label = ""
if "last_detected_image" not in st.session_state:
    st.session_state.last_detected_image = None
if "last_detected_classes" not in st.session_state:
    st.session_state.last_detected_classes = set()

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



# ---------------- AUDIO FUNCTION ----------------
def autoplay_audio(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio controls autoplay="true">
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
def detect_and_visualize(image):
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = model(image_bgr, verbose=False)
    img_with_boxes = results[0].plot(conf=True)

    detected_classes = set()
    for box in results[0].boxes:
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        if confidence > 0.3 and class_name in SOUND_FILES:
            detected_classes.add(class_name)

    label = list(detected_classes)[0] if detected_classes else "Unknown"
    new_detections = detected_classes - st.session_state.last_detected_classes
    st.session_state.last_detected_classes = detected_classes

    return img_with_boxes, label, new_detections

# ---------------- IMAGE RESIZING ----------------
def resize_image(image, max_width=800, max_height=600):
    w, h = image.size
    if w > max_width or h > max_height:
        ratio_w = max_width / w
        ratio_h = max_height / h
        ratio = min(ratio_w, ratio_h)
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size)
    return image

def resize_cv2_image(image_bgr, max_width=800, max_height=600):
    h, w, _ = image_bgr.shape
    if w > max_width or h > max_height:
        ratio_w = max_width / w
        ratio_h = max_height / h
        ratio = min(ratio_w, ratio_h)
        new_dim = (int(w * ratio), int(h * ratio))
        return cv2.resize(image_bgr, new_dim, interpolation=cv2.INTER_AREA)
    return image_bgr

with detect:
    # ---------------- SIDEBAR: UPLOAD IMAGES ----------------
    with st.sidebar:
        st.header("📤 Upload Images")
        uploaded_files = st.file_uploader("Upload up to 25 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files and not st.session_state.uploaded_images:
            st.session_state.uploaded_images = uploaded_files[:25]
            st.session_state.image_index = 0
            st.session_state.score = 0
            st.session_state.submitted = False
            st.session_state.last_label = ""
            st.session_state.last_detected_image = None
            st.session_state.last_detected_classes.clear()

    # ---------------- IMAGE AND QUIZ LOGIC ----------------
    def display_image_and_quiz():
        index = st.session_state.image_index
        total_images = len(st.session_state.uploaded_images)

        if index < total_images:
            current_file = st.session_state.uploaded_images[index]
            current_image = Image.open(current_file).convert("RGB")

            st.subheader(f"🖼️ Image {index + 1} of {total_images}")

            # Create two columns for side-by-side display
            col1, col2 = st.columns([1, 1])

            with col1:
                resized_original = resize_image(current_image, max_width=800, max_height=400)
                st.image(resized_original, caption="Original Image", use_container_width=True)

                # Text input below original image
                user_input = st.text_input("📝 Type your answer below:")

                # Detect button under text input
                detect_clicked = st.button("🔍 Detect Road Sign", key=f"detect_{index}")
                
                # Show dictionary below detect button
                with st.expander("📖 View Road Sign Dictionary"):
                    st.markdown("""
                        *Valid Road Sign Names (used for answer checking):*
                        - Bicycle Lane
                        - Broken and Solid Yellow Lines
                        - Bus Lane
                        - Cats Eye
                        - Continuity Lane
                        - Double Solid Yellow or White Line
                        - Holding Lane
                        - Loading and Unloading Zone
                        - Motorcycle Lane
                        - No Loading and Unloading Curb
                        - No Parking Curb
                        - Parking Bay
                        - Pavement Arrow
                        - Pedestrian Lane
                        - Railroad Crossing
                        - Rumble Strips
                        - Single Solid Lane
                        - Speed Limit
                        - Transition Line
                        """)

            if detect_clicked and not st.session_state.submitted:
                result_img, label, new_detections = detect_and_visualize(current_image)
                st.session_state.last_label = label
                st.session_state.last_detected_image = result_img

                with col2:
                    resized_result = resize_cv2_image(result_img, max_width=800, max_height=400)
                    st.image(cv2.cvtColor(resized_result, cv2.COLOR_BGR2RGB), caption="Detected Image", use_container_width=True)

                # Play new sounds
                if new_detections:
                    for det in new_detections:
                        if det in SOUND_FILES:
                            autoplay_audio(SOUND_FILES[det])

                if user_input.strip().lower() == label.lower():
                    st.success("✅ Correct!")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Incorrect. Correct answer: {label}")


                st.session_state.submitted = True

            if st.session_state.submitted:
                if index < total_images - 1:
                    if st.button("➡️ Next Image", key=f"next_{index}"):
                        st.session_state.image_index += 1
                        st.session_state.submitted = False
                        st.session_state.last_label = ""
                        st.session_state.last_detected_image = None
                        st.session_state.last_detected_classes.clear()
                        st.rerun()
                else:
                    st.success("🎉 You've completed the quiz!")
                    st.markdown(f"### 🏁 Final Score: **{st.session_state.score} / {total_images}**")

                    if st.button("🔄 Restart Quiz"):
                        st.session_state.uploaded_images = []
                        st.session_state.image_index = 0
                        st.session_state.score = 0
                        st.session_state.submitted = False
                        st.session_state.last_label = ""
                        st.session_state.last_detected_image = None
                        st.session_state.last_detected_classes.clear()
                        st.rerun()

    # ---------------- MAIN ----------------
    if st.session_state.uploaded_images:
        display_image_and_quiz()

    else:
        # Reset session state when file is removed
        st.session_state.processed_image = None  # Reset detected image when file is removed
        st.session_state.last_detected_classes.clear()  # Clear detected classes
        st.image("assets/bg.jpg")
        
with model_info:
    st.title('Model Benchmark')

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
