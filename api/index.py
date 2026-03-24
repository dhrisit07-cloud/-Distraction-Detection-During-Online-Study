from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import mediapipe as mp
import os
import sys

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.head_pose import get_head_pose
from modules.gaze_tracker import analyse_gaze
from modules.phone_detector import detect_distractors

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-app.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# New MediaPipe Tasks API Config
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_landmarker.task")
YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "yolov8n.pt")

# Initialize Landmarker (Serverless mode - IMAGE)
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)
landmarker = FaceLandmarker.create_from_options(options)

global_frame_count = 0
last_known_distractors = []

@app.get("/api/health")
def health():
    return {"status": "ok", "model_found": os.path.exists(MODEL_PATH)}

@app.post("/api/process")
async def process_frame(file: UploadFile = File(...)):
    global global_frame_count, last_known_distractors
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image data"}
    h, w = frame.shape[:2]

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Detect
    result = landmarker.detect(mp_image)

    head_dist = False
    gaze_dist = False
    drowsy    = False
    yaw = pitch = 0

    if result.face_landmarks:
        lm = result.face_landmarks[0]
        
        # Head pose
        y, p, r = get_head_pose(lm, w, h)
        yaw, pitch = y or 0, p or 0
        head_dist = abs(yaw) > 25 or abs(pitch) > 20

        # Gaze
        gaze_data = analyse_gaze(lm, w, h)
        gaze_dist = gaze_data["looking_away"]
        drowsy    = gaze_data["avg_ear"] < 0.20

    # Phone detection
    detections = detect_distractors(frame, global_frame_count, model_path=YOLO_MODEL_PATH)
    if detections is not None:
        last_known_distractors = detections
        
    distractor_labels = [d["label"] for d in last_known_distractors]
    phone_detected = "phone" in distractor_labels

    global_frame_count += 1

    return {
        "head_distracted": bool(head_dist),
        "gaze_distracted": bool(gaze_dist),
        "drowsy":          bool(drowsy),
        "phone_detected":  bool(phone_detected),
        "distractors":     distractor_labels,
        "yaw":             float(round(yaw, 1)) if yaw is not None else 0.0,
        "pitch":           float(round(pitch, 1)) if pitch is not None else 0.0
    }
