import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import time
import torch

from ecochat import eco_chat_response  # <-- ChatGPT-powered EcoChat

st.set_page_config(page_title="SmartBin Dashboard", page_icon="♻", layout="wide")
st.title("♻ SmartBin — Real-Time Waste Detection + EcoChat (ChatGPT API)")

# ---------- YOLO model ----------
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")  # your trained model
        st.success("✅ YOLOv8 model loaded.")
        return model
    except Exception as e:
        st.warning(f"⚠ Could not load best.pt, using yolov8n.pt. Error: {e}")
        return YOLO("yolov8n.pt")

model = load_model()

# ---------- Session state ----------
if "detections" not in st.session_state:
    st.session_state.detections = []
if "unique_classes" not in st.session_state:
    st.session_state.unique_classes = set()

# ---------- Sidebar: camera config ----------
st.sidebar.header("📡 Camera")
default_url = "http://10.134.205.107:4747/video"  # replace with your DroidCam IP
url = st.sidebar.text_input("DroidCam URL", default_url)
start_btn = st.sidebar.button("▶ Start Detection", key="start_btn")
stop_btn = st.sidebar.button("⏹ Stop Stream", key="stop_btn")
status_box = st.sidebar.empty()

frame_box = st.empty()

# ---------- Detection util ----------
def log_detections(results):
    names = results[0].names
    for box in results[0].boxes:
        label = names[int(box.cls[0])]
        conf = f"{float(box.conf[0])*100:.1f}%"
        # Simple category mapping
        if any(x in label.lower() for x in ["metal", "can", "foil"]):
            category = "Metal"
        elif any(x in label.lower() for x in ["food", "wet", "banana", "bio"]):
            category = "Wet"
        else:
            category = "Dry"

        st.session_state.detections.append({
            "Item": label,
            "Category": category,
            "Confidence": conf,
            "Time": datetime.now().strftime("%H:%M:%S")
        })
        st.session_state.unique_classes.add(label)

def process_frame(frame):
    # Speed tips: smaller frame, fewer pixels to process
    frame_resized = cv2.resize(frame, (640, 480))
    results = model(
        frame_resized,
        conf=0.6,
        imgsz=480,
        stream=False,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
    )
    log_detections(results)
    annotated = results[0].plot()  # draw boxes
    return annotated

# ---------- Main stream loop ----------
if start_btn:
    status_box.info("Connecting to DroidCam...")
    cap = cv2.VideoCapture(url)
    time.sleep(1.2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        time.sleep(1)
    if not cap.isOpened():
        status_box.warning("DroidCam not reachable, switching to webcam.")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ No camera found. Check DroidCam or your webcam.")
    else:
        status_box.success("✅ Camera connected. Running YOLO...")
        last_push = 0.0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                status_box.error("⚠ No frame. Check connection.")
                break

            annotated = process_frame(frame)

            # Throttle UI redraw (~4–6 FPS keeps Streamlit smooth)
            now = time.time()
            if now - last_push >= 0.18:
                frame_box.image(annotated, channels="BGR", use_container_width=True)
                last_push = now

            if stop_btn:
                status_box.warning("🛑 Stopped.")
                break

        cap.release()
        cv2.destroyAllWindows()

# ---------- EcoChat (sidebar) ----------
st.sidebar.markdown("---")
st.sidebar.header("💬 EcoChat (ChatGPT)")

select_label = (
    st.sidebar.selectbox("Detected items", sorted(st.session_state.unique_classes))
    if st.session_state.unique_classes else None
)
custom_item = st.sidebar.text_input("Or type an object:")

if st.sidebar.button("Ask EcoChat", key="chat_btn"):
    query = custom_item.strip() if custom_item else (select_label or "").strip()
    if query:
        with st.sidebar.expander(f"EcoChat advice for '{query}'", expanded=True):
            st.write(eco_chat_response(query))
    else:
        st.sidebar.warning("Select or type an item first.")

# ---------- Detection History ----------
st.markdown("---")
st.subheader("📋 Detection History")
if st.session_state.detections:
    df = pd.DataFrame(st.session_state.detections).tail(15)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No detections yet — start the live stream.")

# ---------- Summary ----------
st.markdown("---")
st.subheader("📊 Waste Category Summary")
if st.session_state.detections:
    df_all = pd.DataFrame(st.session_state.detections)
    st.metric("Total Detections", len(df_all))
    st.metric("Last Detection Time", df_all["Time"].iloc[-1])
    counts = df_all["Category"].value_counts()
    st.bar_chart(counts)
else:
    st.info("Start detection to view summary.")