"""
Waste Classification Module using YOLOv8
Detects everyday objects and classifies them with:
  1. Bin Type: Wet Bin / Dry Bin / Metal Bin
  2. Degradability: Biodegradable / Non-Biodegradable
"""

from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).parent

# Each item maps to (bin_type, degradability)
#   bin_type: "Wet" (organic/food), "Dry" (paper, plastic, glass, cloth, electronics), "Metal" (metallic items)
#   degradability: "Biodegradable" or "Non-Biodegradable"
WASTE_INFO = {
    # --- WET BIN (organic / food waste) --- all biodegradable
    "banana":       ("Wet", "Biodegradable"),
    "apple":        ("Wet", "Biodegradable"),   # treated as tomato
    "orange":       ("Wet", "Biodegradable"),
    "broccoli":     ("Wet", "Biodegradable"),
    "carrot":       ("Wet", "Biodegradable"),
    "hot dog":      ("Wet", "Biodegradable"),
    "pizza":        ("Wet", "Biodegradable"),
    "donut":        ("Wet", "Biodegradable"),   # treated as tomato
    "cake":         ("Wet", "Biodegradable"),
    "sandwich":     ("Wet", "Biodegradable"),
    "potted plant": ("Wet", "Biodegradable"),
    "tomato":       ("Wet", "Biodegradable"),        # custom-trained

    # --- METAL BIN --- all non-biodegradable
    "fork":          ("Metal", "Non-Biodegradable"),
    "knife":         ("Metal", "Non-Biodegradable"),
    "spoon":         ("Metal", "Non-Biodegradable"),
    "scissors":      ("Metal", "Non-Biodegradable"),
    "clock":         ("Metal", "Non-Biodegradable"),
    "fire hydrant":  ("Metal", "Non-Biodegradable"),
    "stop sign":     ("Metal", "Non-Biodegradable"),
    "parking meter": ("Metal", "Non-Biodegradable"),
    "bicycle":       ("Metal", "Non-Biodegradable"),
    "car":           ("Metal", "Non-Biodegradable"),
    "motorcycle":    ("Metal", "Non-Biodegradable"),
    "bus":           ("Metal", "Non-Biodegradable"),
    "train":         ("Metal", "Non-Biodegradable"),
    "truck":         ("Metal", "Non-Biodegradable"),
    "airplane":      ("Metal", "Non-Biodegradable"),
    "boat":          ("Metal", "Non-Biodegradable"),
    "sink":          ("Metal", "Non-Biodegradable"),
    "keys":          ("Metal", "Non-Biodegradable"),  # custom-trained

    # --- DRY BIN --- plastic, glass, electronics, paper, cloth, synthetic
    # Plastic
    "bottle":      ("Dry", "Non-Biodegradable"),
    "cup":         ("Dry", "Non-Biodegradable"),
    "frisbee":     ("Dry", "Non-Biodegradable"),
    "sports ball": ("Dry", "Non-Biodegradable"),
    "kite":        ("Dry", "Non-Biodegradable"),
    "umbrella":    ("Dry", "Non-Biodegradable"),
    "handbag":     ("Dry", "Non-Biodegradable"),
    "suitcase":    ("Dry", "Non-Biodegradable"),
    "backpack":    ("Dry", "Non-Biodegradable"),
    "toothbrush":  ("Dry", "Non-Biodegradable"),

    # Glass / Ceramic
    "wine glass":  ("Dry", "Non-Biodegradable"),
    "bowl":        ("Dry", "Non-Biodegradable"),
    "vase":        ("Dry", "Non-Biodegradable"),

    # Electronics
    "tv":           ("Dry", "Non-Biodegradable"),
    "laptop":       ("Metal", "Non-Biodegradable"),
    "mouse":        ("Metal", "Non-Biodegradable"),
    "remote":       ("Dry", "Non-Biodegradable"),
    "keyboard":     ("Metal", "Non-Biodegradable"),
    "cell phone":   ("Dry", "Non-Biodegradable"),
    "microwave":    ("Dry", "Non-Biodegradable"),
    "oven":         ("Dry", "Non-Biodegradable"),
    "toaster":      ("Dry", "Non-Biodegradable"),
    "refrigerator": ("Dry", "Non-Biodegradable"),
    "hair drier":   ("Dry", "Non-Biodegradable"),

    # Paper / Wood (dry bin, biodegradable)
    "paper":        ("Dry", "Biodegradable"),         # custom-trained
    "book":         ("Dry", "Biodegradable"),
    "bench":        ("Dry", "Biodegradable"),
    "dining table": ("Dry", "Biodegradable"),
    "baseball bat": ("Dry", "Biodegradable"),

    # Cloth / Synthetic (dry bin, non-biodegradable)
    "tie":            ("Dry", "Non-Biodegradable"),
    "teddy bear":     ("Dry", "Non-Biodegradable"),
    "chair":          ("Dry", "Non-Biodegradable"),
    "couch":          ("Dry", "Non-Biodegradable"),
    "bed":            ("Dry", "Non-Biodegradable"),
    "skateboard":     ("Dry", "Non-Biodegradable"),
    "surfboard":      ("Dry", "Non-Biodegradable"),
    "snowboard":      ("Dry", "Non-Biodegradable"),
    "tennis racket":  ("Dry", "Non-Biodegradable"),
    "baseball glove": ("Dry", "Non-Biodegradable"),
    "skis":           ("Dry", "Non-Biodegradable"),
    "traffic light":  ("Dry", "Non-Biodegradable"),

    # Animals (not typical waste, skip)
    "bird": None,
    "cat": None,
    "dog": None,
    "horse": None,
    "sheep": None,
    "cow": None,
    "elephant": None,
    "bear": None,
    "zebra": None,
    "giraffe": None,
    "person": None,
}

# Friendly display names
ITEM_DISPLAY_NAMES = {
    "bottle": "Plastic Bottle",
    "cup": "Cup / Glass",
    "fork": "Fork (Metal)",
    "knife": "Knife (Metal)",
    "spoon": "Spoon (Metal)",
    "scissors": "Scissors (Metal)",
    "cell phone": "Mobile Phone",
    "laptop": "Laptop (Metal)",
    "keyboard": "Keyboard (Metal)",
    "mouse": "Mouse (Metal)",
    "remote": "Remote Control",
    "tv": "Television",
    "book": "Book / Paper",
    "banana": "Banana",
    "apple": "Tomato",   # relabelled
    "donut": "Tomato",   # relabelled
    "orange": "Orange",
    "broccoli": "Broccoli",
    "carrot": "Carrot",
    "pizza": "Pizza",
    "cake": "Cake",
    "sandwich": "Sandwich",
    "hot dog": "Hot Dog",
    "donut": "Donut",
    "potted plant": "Plant / Wood",
    "bench": "Wooden Bench",
    "dining table": "Wooden Table",
    "clock": "Clock (Metal)",
    "vase": "Vase (Glass)",
    "toothbrush": "Toothbrush",
    "hair drier": "Hair Dryer",
    "microwave": "Microwave",
    "toaster": "Toaster",
    "refrigerator": "Refrigerator",
    "backpack": "Backpack",
    "handbag": "Handbag",
    "suitcase": "Suitcase",
    "umbrella": "Umbrella",
    "tie": "Tie (Cloth)",
    "sports ball": "Ball (Rubber)",
    "wine glass": "Wine Glass",
    "bowl": "Bowl",
    "teddy bear": "Teddy Bear",
    "baseball bat": "Baseball Bat (Wood)",
    "fire hydrant": "Fire Hydrant (Metal)",
    "stop sign": "Stop Sign (Metal)",
    "parking meter": "Parking Meter (Metal)",
    "sink": "Sink (Metal)",
    "bicycle": "Bicycle (Metal)",
    "car": "Car (Metal)",
    "motorcycle": "Motorcycle (Metal)",
    # Custom-trained classes
    "tomato": "Tomato",
    "keys": "Keys (Metal)",
    "paper": "Paper",
}

# Colors for bin types
COLOR_WET = (34, 139, 34)       # Green
COLOR_DRY = (220, 160, 0)       # Blue/Teal
COLOR_METAL = (80, 80, 220)     # Red-ish
COLOR_NA = (128, 128, 128)      # Gray

BIN_COLORS = {
    "Wet": COLOR_WET,
    "Dry": COLOR_DRY,
    "Metal": COLOR_METAL,
}

BIN_LABELS = {
    "Wet": "Wet Bin (Green)",
    "Dry": "Dry Bin (Blue)",
    "Metal": "Metal Bin (Grey)",
}


def load_model(model_size="n"):
    """Load YOLOv8 model(s). Returns a dict with 'coco' and optionally 'custom' models."""
    # Using best.pt as requested/assumed updated primary model
    try:
        coco_model = YOLO("best.pt")
    except:
        coco_model = YOLO(f"yolov8{model_size}.pt")
        
    models = {"coco": coco_model, "custom": None}

    # Optional second model if it exists
    custom_path = PROJECT_DIR / "yolov8n_custom.pt"
    if custom_path.exists():
        try:
            models["custom"] = YOLO(str(custom_path))
            print(f"Custom model loaded: {custom_path}")
        except Exception as e:
            print(f"Warning: Could not load custom model: {e}")

    return models


def get_waste_info(class_name):
    """Get (bin_type, degradability) for a detected object. Returns None if not waste."""
    return WASTE_INFO.get(class_name)


def get_display_name(class_name):
    """Get user-friendly display name for a detected object."""
    return ITEM_DISPLAY_NAMES.get(class_name, class_name.title())


def detect_and_classify(models, image, confidence_threshold=0.35):
    """
    Run YOLOv8 detection on an image and classify each object.
    Accepts either a models dict (from load_model) or a single YOLO model for backwards compat.

    Returns:
        annotated_image: Image with bounding boxes drawn
        detections: List of dicts with detection info
    """
    # Support both old (single model) and new (dict) calling convention
    if isinstance(models, dict):
        model_list = [m for m in [models.get("coco"), models.get("custom")] if m is not None]
    else:
        model_list = [models]

    detections = []
    annotated_image = image.copy()

    for model in model_list:
        results = model(image, conf=confidence_threshold, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                waste_info = get_waste_info(class_name)
                if waste_info is None:
                    continue

                bin_type, degradability = waste_info
                display_name = get_display_name(class_name)
                color = BIN_COLORS.get(bin_type, COLOR_NA)

                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)

                # Label: two lines of info
                line1 = f"{display_name} ({confidence:.0%})"
                line2 = f"{BIN_LABELS[bin_type]} | {degradability}"

                # Draw label background for line 1
                (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                max_w = max(tw1, tw2) + 10
                total_h = th1 + th2 + 20

                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - total_h),
                    (x1 + max_w, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated_image, line1,
                    (x1 + 4, y1 - th2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                )
                cv2.putText(
                    annotated_image, line2,
                    (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                )

                detections.append({
                    "item": display_name,
                    "class_name": class_name,
                    "bin_type": bin_type,
                    "degradability": degradability,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })

    return annotated_image, detections
