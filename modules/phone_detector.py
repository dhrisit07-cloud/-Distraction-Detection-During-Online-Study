import cv2
from ultralytics import YOLO
import numpy as np

# COCO class IDs for distractors
# 67 = cell phone, 76 = scissors, 73 = book, 46 = banana (food example)
DISTRACTOR_CLASSES = {
    65: "phone",  # remote (often confused with phone)
    67: "phone",  # cell phone
    73: "book",
    76: "scissors",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
}

# Only run YOLO every N frames to save CPU
YOLO_FRAME_INTERVAL = 2

_model = None

def get_model(model_path="models/yolov8n.pt"):
    """Lazy-load YOLO model (downloads automatically on first run)."""
    global _model
    if _model is None:
        _model = YOLO(model_path)
    return _model


def detect_distractors(frame, frame_count, model_path="models/yolov8n.pt"):
    """
    Run YOLO detection every YOLO_FRAME_INTERVAL frames.
    Returns list of detected distractor dicts: {label, confidence, bbox}
    Returns None on skipped frames (use previous result).
    """
    if frame_count % YOLO_FRAME_INTERVAL != 0:
        return None

    model   = get_model(model_path)
    results = model(frame, verbose=False)[0]

    detected = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in DISTRACTOR_CLASSES:
            conf  = float(box.conf[0])
            thresh = 0.15 if cls_id in [65, 67] else 0.30
            if conf > thresh:  # confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected.append({
                    "label":      DISTRACTOR_CLASSES[cls_id],
                    "confidence": conf,
                    "bbox":       (x1, y1, x2, y2),
                })

    return detected


def draw_detections(frame, detections):
    """Draw bounding boxes for detected distractors."""
    if not detections:
        return frame

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f"{d['label']} {d['confidence']:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    return frame


# Standalone test
if __name__ == "__main__":
    cap         = cv2.VideoCapture(0, cv2.CAP_MSMF)
    frame_count = 0
    last_detections = []

    print("Loading YOLO model (downloads ~6MB on first run)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_distractors(frame, frame_count)
        if result is not None:
            last_detections = result

        frame = draw_detections(frame, last_detections)

        h, w = frame.shape[:2]
        status = f"Distractors: {len(last_detections)}" if last_detections else "No distractors"
        color  = (0, 0, 255) if last_detections else (0, 200, 0)

        cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
        cv2.putText(frame, status, (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Phone Detector Module", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
