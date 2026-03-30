# ==============================================================
# ♻️ SmartBin YOLOv8 — Multi-Object Stable Detection (macOS + DroidCam)
# ==============================================================

from ultralytics import YOLO
import cv2
import numpy as np
import random
import time
from collections import deque

# --- 1️⃣ Load your trained YOLOv8 model ---
model = YOLO("best.pt")

# --- 2️⃣ DroidCam video stream ---
# Example: http://192.168.x.x:4747/video
SOURCE = "http://10.134.205.107:4747/video"
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise SystemExit("❌ Could not connect to DroidCam. Check IP or app.")

print("✅ SmartBin YOLOv8 Multi-Object Detection Started (Press 'Q' to quit)")

# --- 3️⃣ Detection Settings ---
CONF_THRES = 0.6          # confidence threshold
IOU_THRES = 0.45          # intersection threshold for NMS
IMGSZ = 640
SMOOTH_FRAMES = 8         # number of frames to average per class for smoothing

# --- 4️⃣ Assign colors per class ---
colors = {name: [random.randint(0, 255) for _ in range(3)] for name in model.names.values()}

# --- 5️⃣ Initialize smoothing history for each class ---
history = {name: deque(maxlen=SMOOTH_FRAMES) for name in model.names.values()}

def smooth_box(cls_name, new_box):
    """Averages recent boxes for smoother detection."""
    hist = history[cls_name]
    hist.append(new_box)
    return tuple(map(int, np.mean(hist, axis=0))) if len(hist) > 0 else new_box

# --- 6️⃣ Prepare window ---
cv2.namedWindow("♻️ SmartBin YOLOv8 Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("♻️ SmartBin YOLOv8 Live", 960, 720)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received. Retrying...")
        time.sleep(0.1)
        continue

    # --- Maintain aspect ratio (letterbox padding) ---
    h, w = frame.shape[:2]
    scale = IMGSZ / max(h, w)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    pad_h = IMGSZ - resized.shape[0]
    pad_w = IMGSZ - resized.shape[1]
    padded = cv2.copyMakeBorder(resized,
                                pad_h // 2, pad_h - pad_h // 2,
                                pad_w // 2, pad_w - pad_w // 2,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # --- YOLO inference ---
    results = model.predict(padded, conf=CONF_THRES, iou=IOU_THRES,
                            imgsz=IMGSZ, verbose=False)

    # --- FPS calculation ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # --- Draw all detections ---
    for box in results[0].boxes:
        # Get bounding box info
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        cls_name = model.names[cls]
        conf = float(box.conf[0])
        color = colors[cls_name]

        # Smooth this class’s box coordinates
        smoothed_box = smooth_box(cls_name, (x1, y1, x2, y2))
        label = f"{cls_name} {conf:.2f}"

        # Draw precise bounding box and label
        cv2.rectangle(padded,
                      (smoothed_box[0], smoothed_box[1]),
                      (smoothed_box[2], smoothed_box[3]),
                      color, 2)
        cv2.putText(padded, label,
                    (smoothed_box[0], smoothed_box[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # --- Overlay FPS counter ---
    cv2.putText(padded, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # --- Show feed ---
    cv2.imshow("♻️ SmartBin YOLOv8 Live", padded)

    # --- Quit on 'Q' ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting stream...")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("✅ Stream closed cleanly.")