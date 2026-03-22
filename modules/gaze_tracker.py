import cv2
import mediapipe as mp
import numpy as np
import os

# New MediaPipe Tasks API imports
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to the model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_landmarker.task")

# Eye landmark indices (MediaPipe 468-point mesh + 10 iris points)
# Left eye
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133
# Right eye
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263
# Iris centres (468/473 are the center points in refined mesh)
LEFT_IRIS  = 468
RIGHT_IRIS = 473

# Thresholds
EAR_THRESHOLD        = 0.20   # below this = eyes closed (drowsy)
DROWSY_FRAME_LIMIT   = 30 * 2 # 2 seconds at ~30fps
GAZE_RATIO_THRESHOLD = 0.35   # iris too far left/right = looking away


def eye_aspect_ratio(landmarks, top, bottom, left, right, w, h):
    """Compute Eye Aspect Ratio (EAR). Low value = closed eye."""
    if not landmarks or len(landmarks) <= max(top, bottom, left, right):
        return 0.5
    
    p_top    = np.array([landmarks[top].x    * w, landmarks[top].y    * h])
    p_bottom = np.array([landmarks[bottom].x * w, landmarks[bottom].y * h])
    p_left   = np.array([landmarks[left].x   * w, landmarks[left].y   * h])
    p_right  = np.array([landmarks[right].x  * w, landmarks[right].y  * h])

    vertical   = np.linalg.norm(p_top - p_bottom)
    horizontal = np.linalg.norm(p_left - p_right)

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def gaze_ratio(landmarks, iris_idx, eye_left, eye_right, w, h):
    """
    Returns 0.0–1.0 where 0.5 = looking straight.
    Values near 0 or 1 = looking far left/right.
    """
    if not landmarks or len(landmarks) <= max(iris_idx, eye_left, eye_right):
        return 0.5
        
    iris_x  = landmarks[iris_idx].x * w
    left_x  = landmarks[eye_left].x  * w
    right_x = landmarks[eye_right].x * w

    eye_width = right_x - left_x
    if eye_width == 0:
        return 0.5

    ratio = (iris_x - left_x) / eye_width
    return ratio


def analyse_gaze(landmarks, w, h):
    """
    Returns a dict with:
      - left_ear, right_ear: eye aspect ratios
      - avg_ear: average EAR
      - drowsy: bool
      - gaze_left, gaze_right: gaze ratios per eye
      - looking_away: bool
    """
    left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,
                                  LEFT_EYE_LEFT,  LEFT_EYE_RIGHT,  w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                  RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, w, h)
    avg_ear   = (left_ear + right_ear) / 2.0
    drowsy    = avg_ear < EAR_THRESHOLD

    # Gaze ratios (only reliable when eyes are open)
    try:
        gl = gaze_ratio(landmarks, LEFT_IRIS,  LEFT_EYE_LEFT,  LEFT_EYE_RIGHT,  w, h)
        gr = gaze_ratio(landmarks, RIGHT_IRIS, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, w, h)
        # avg_gaze = (gl + gr) / 2.0  # Optional: use average for both eyes
        
        # Consider looking away if either eye is looking too far
        looking_away = (gl < GAZE_RATIO_THRESHOLD or gl > (1 - GAZE_RATIO_THRESHOLD) or
                        gr < GAZE_RATIO_THRESHOLD or gr > (1 - GAZE_RATIO_THRESHOLD))
    except Exception:
        gl = gr = 0.5
        looking_away = False

    return {
        "left_ear":    left_ear,
        "right_ear":   right_ear,
        "avg_ear":     avg_ear,
        "drowsy":      drowsy,
        "gaze_left":   gl,
        "gaze_right":  gr,
        "looking_away": looking_away,
    }


# Standalone test
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        exit()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    landmarker = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    drowsy_frames = 0

    print("Gaze Tracker Module (Tasks API) ready. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        
        # Convert OpenCV frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect landmarks
        result = landmarker.detect(mp_image)

        status = "EYES OK"
        color  = (0, 200, 0)

        if result.face_landmarks:
            lm   = result.face_landmarks[0]
            data = analyse_gaze(lm, w, h)

            cv2.putText(frame, f"EAR: {data['avg_ear']:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(frame, f"Gaze L: {data['gaze_left']:.2f}  R: {data['gaze_right']:.2f}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            if data["drowsy"]:
                drowsy_frames += 1
                if drowsy_frames >= DROWSY_FRAME_LIMIT:
                    status = "DROWSY"
                    color  = (0, 100, 255)
            else:
                drowsy_frames = max(0, drowsy_frames - 1)

            if data["looking_away"] and not data["drowsy"]:
                status = "LOOKING AWAY"
                color  = (0, 0, 255)
        else:
            status = "NO FACE"
            color = (0, 165, 255)

        cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
        cv2.putText(frame, f"Gaze: {status}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Gaze Tracker Module (Tasks API)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
