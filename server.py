"""
SmartBin Dashboard — Flask Backend
Real-time YOLO waste detection with DroidCam integration
"""

import cv2
import time
import json
import torch
import threading
from datetime import datetime
from collections import deque
from flask import Flask, render_template, Response, request, jsonify

import classifier  # Refined classification metadata

# --------------- Configuration ---------------
DEFAULT_DROIDCAM_URL = "http://10.134.205.107:4747/video"
YOLO_CONF_THRESHOLD = 0.55
YOLO_IMG_SIZE = 320           # Optimized for speed
MAX_DETECTION_HISTORY = 200
STREAM_FPS = 20               # Cap for network efficiency
JPEG_QUALITY = 45             # Balanced for detail and bandwidth

# --------------- Flask App ---------------
app = Flask(__name__)

# --------------- Shared State ---------------
detections = deque(maxlen=MAX_DETECTION_HISTORY)
stream_active = False
stream_lock = threading.Lock()
settings = {
    "droidcam_url": DEFAULT_DROIDCAM_URL,
    "conf_threshold": YOLO_CONF_THRESHOLD,
    "use_webcam": False,
}

# --------------- YOLO Model ---------------
model = None

def get_model():
    global model
    if model is None:
        try:
            # Now load the model dictionary from classifier
            model = classifier.load_model()
            print("🚀 SmartBin AI Models Loaded Successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            # Fallback to single tiny model if everything fails
            from ultralytics import YOLO
            model = {"coco": YOLO("yolov8n.pt"), "custom": None}
    return model

# --------------- Waste Category Mapper ---------------
# (Delegated to classifier.py)

# --------------- Threaded Camera Reader ---------------
# Grabs frames in a background thread so we always get the
# LATEST frame and never stall on the OpenCV internal buffer.
class CameraStream:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        # Minimize internal buffer (OpenCV queues frames otherwise)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

        if self.cap.isOpened():
            self.grabbed, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _reader(self):
        """Continuously grab frames; only the latest one is kept."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()


# --------------- Threaded Inference Engine ---------------
class InferenceEngine:
    def __init__(self, model_getter):
        self.model_getter = model_getter
        self.frame = None
        self.annotated = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = None

    def start(self):
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        models_dict = self.model_getter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        half = device == "cuda"  # FP16 speedup for NVIDIA GPUs

        while not self.stopped:
            with self.lock:
                target_frame = self.frame
                self.frame = None # Consume the frame
            
            if target_frame is None:
                time.sleep(0.01)
                continue

            try:
                annotated_tmp = target_frame.copy()
                
                # Iterate through all active models (Standard COCO and Custom)
                for model_key, mdl in models_dict.items():
                    if mdl is None: continue
                    
                    results = mdl(
                        target_frame,
                        conf=settings.get("conf_threshold", YOLO_CONF_THRESHOLD),
                        imgsz=YOLO_IMG_SIZE,
                        stream=False,
                        device=device,
                        verbose=False,
                        half=half
                    )
                    
                    names = results[0].names
                    
                    # Draw boxes and log detections
                    for box in results[0].boxes:
                        label_small = names[int(box.cls[0])]
                        conf_val = float(box.conf[0])
                        
                        # Enhanced classification from classifier.py
                        waste_info = classifier.get_waste_info(label_small)
                        if waste_info is None:
                            continue # Skip non-waste items

                        bin_type, degradability = waste_info
                        display_name = classifier.get_display_name(label_small)
                        bgr_color = classifier.BIN_COLORS.get(bin_type, (128, 128, 128))
                        
                        # Convert BGR to Hex (for history)
                        color_hex = '#{:02x}{:02x}{:02x}'.format(bgr_color[2], bgr_color[1], bgr_color[0])
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        w, h = (x2 - x1), (y2 - y1)
                        scale = 0.85
                        nw, nh = w * scale, h * scale
                        nx1, ny1 = int(cx - nw / 2.0), int(cy - nh / 2.0)
                        nx2, ny2 = int(cx + nw / 2.0), int(cy + nh / 2.0)
                        nx1, ny1 = max(0, nx1), max(0, ny1)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_tmp, (nx1, ny1), (nx2, ny2), bgr_color, 2)
                        
                        # Label: two lines of info
                        lbl_line1 = f"{display_name} ({conf_val:.0%})"
                        lbl_line2 = f"{classifier.BIN_LABELS[bin_type]} | {degradability}"
                        
                        # Background for text
                        (tw1, th1), _ = cv2.getTextSize(lbl_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        (tw2, th2), _ = cv2.getTextSize(lbl_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        max_w = max(tw1, tw2) + 10
                        bg_y1 = ny1 - (th1 + th2 + 15) if ny1 - 30 > 0 else 0
                        bg_y2 = ny1 if ny1 - 30 > 0 else (th1 + th2 + 15)
                        
                        cv2.rectangle(annotated_tmp, (nx1, bg_y1), (nx1 + max_w, bg_y2), bgr_color, -1)
                        
                        # Draw lines
                        ty1 = bg_y1 + th1 + 5
                        ty2 = bg_y1 + th1 + th2 + 12
                        cv2.putText(annotated_tmp, lbl_line1, (nx1+4, ty1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                        cv2.putText(annotated_tmp, lbl_line2, (nx1+4, ty2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                        detections.appendleft({
                            "item": display_name,
                            "category": bin_type,
                            "degradability": degradability,
                            "confidence": round(conf_val * 100, 1),
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "color": color_hex,
                        })

                with self.lock:
                    self.annotated = annotated_tmp

            except Exception as e:
                print(f"Inference error: {e}")
                time.sleep(0.1)

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame

    def get_annotated(self):
        with self.lock:
            return self.annotated

    def stop(self):
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1.0)


# --------------- Frame Generator ---------------
def generate_frames():
    global stream_active
    
    source = settings["droidcam_url"]
    if settings.get("use_webcam"):
        source = 0

    cam = CameraStream(source)
    if not cam.is_opened() and source != 0:
        print("⚠ DroidCam not reachable, trying webcam...")
        cam = CameraStream(0)

    if not cam.is_opened():
        print("❌ No camera source available.")
        stream_active = False
        return

    # Start the async inference engine
    engine = InferenceEngine(get_model)
    engine.start()

    stream_active = True
    print(f"✅ Low-latency stream started: {source}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    target_sleep = 1.0 / STREAM_FPS

    try:
        while stream_active:
            loop_start = time.time()
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Update inference engine with latest frame
            # (It processes them in the background)
            frame_small = cv2.resize(frame, (480, 360))
            engine.update_frame(frame_small)

            # Get latest annotated frame from engine
            out = engine.get_annotated()
            if out is None:
                out = frame_small

            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', out, encode_params)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Dynamic sleep to maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, target_sleep - elapsed)
            time.sleep(sleep_time)

    finally:
        engine.stop()
        cam.release()
        stream_active = False
        print("📷 Camera and Inference engine stopped.")

# --------------- Routes ---------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/detections")
def api_detections():
    limit = request.args.get("limit", 20, type=int)
    return jsonify(list(detections)[:limit])

@app.route("/api/stats")
def api_stats():
    all_dets = list(detections)
    total = len(all_dets)
    cats = {}
    for d in all_dets:
        c = d["category"]
        cats[c] = cats.get(c, 0) + 1

    return jsonify({
        "total": total,
        "categories": cats,
        "last_time": all_dets[0]["time"] if all_dets else "--:--:--",
    })

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        data = request.get_json(force=True)
        if "droidcam_url" in data:
            settings["droidcam_url"] = data["droidcam_url"]
        if "conf_threshold" in data:
            settings["conf_threshold"] = float(data["conf_threshold"])
        if "use_webcam" in data:
            settings["use_webcam"] = bool(data["use_webcam"])
        return jsonify({"status": "ok", "settings": settings})
    return jsonify(settings)

@app.route("/api/ecochat", methods=["POST"])
def api_ecochat():
    data = request.get_json(force=True)
    query = data.get("message", "").strip()
    if not query:
        return jsonify({"error": "Empty message"}), 400
    try:
        from ecochat import eco_chat_response
        reply = eco_chat_response(query)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"⚠️ EcoChat unavailable: {e}"})

@app.route("/api/ecochat/recycle", methods=["POST"])
def api_ecochat_recycle():
    """Quick recycling tip for a specific detected item."""
    data = request.get_json(force=True)
    item = data.get("item", "").strip()
    bin_type = data.get("bin_type", "").strip()
    degradability = data.get("degradability", "").strip()
    if not item:
        return jsonify({"error": "No item specified"}), 400
    try:
        from ecochat import eco_chat_recycle_tip
        reply = eco_chat_recycle_tip(item, bin_type, degradability)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"⚠️ EcoChat unavailable: {e}"})

@app.route("/api/stream/status")
def stream_status():
    return jsonify({"active": stream_active})

@app.route("/api/detections/clear", methods=["POST"])
def clear_detections():
    detections.clear()
    return jsonify({"status": "cleared"})

# --------------- Main ---------------
if __name__ == "__main__":
    # Pre-load model
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
