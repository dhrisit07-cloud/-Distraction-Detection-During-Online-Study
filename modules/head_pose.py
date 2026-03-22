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
# Note: Use absolute path or relative to project root
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_landmarker.task")

# Key landmark indices (canonical MediaPipe Face Mesh indices)
NOSE_TIP        = 1
CHIN            = 152
LEFT_EYE_CORNER = 263
RIGHT_EYE_CORNER= 33
LEFT_MOUTH      = 287
RIGHT_MOUTH     = 57

# Generic 3D face model points (in mm)
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),    # Nose tip
    (0.0,  -63.6,  -12.5),    # Chin
    (-43.3,  32.7, -26.0),    # Left eye corner
    ( 43.3,  32.7, -26.0),    # Right eye corner
    (-28.9, -28.9, -24.1),    # Left mouth corner
    ( 28.9, -28.9, -24.1),    # Right mouth corner
], dtype=np.float64)

# Thresholds
YAW_THRESHOLD   = 20   # degrees left/right
PITCH_THRESHOLD = 15   # degrees up/down


def get_head_pose(landmarks, frame_w, frame_h):
    """
    Compute yaw, pitch, roll from face landmarks. 
    'landmarks' is a list of NormalizedLandmarks from the Tasks API.
    Returns (yaw, pitch, roll) in degrees.
    """
    if not landmarks or len(landmarks) < 468:
        return None, None, None

    image_points = np.array([
        (landmarks[NOSE_TIP].x        * frame_w, landmarks[NOSE_TIP].y        * frame_h),
        (landmarks[CHIN].x            * frame_w, landmarks[CHIN].y            * frame_h), # Fixed index access in original chin point (was y twice)
        (landmarks[LEFT_EYE_CORNER].x * frame_w, landmarks[LEFT_EYE_CORNER].y * frame_h),
        (landmarks[RIGHT_EYE_CORNER].x* frame_w, landmarks[RIGHT_EYE_CORNER].y* frame_h),
        (landmarks[LEFT_MOUTH].x      * frame_w, landmarks[LEFT_MOUTH].y      * frame_h),
        (landmarks[RIGHT_MOUTH].x     * frame_w, landmarks[RIGHT_MOUTH].y     * frame_h),
    ], dtype=np.float64)

    focal_length  = frame_w
    center        = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)

    if sy >= 1e-6:
        pitch = np.degrees(np.arctan2( rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw   = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        roll  = np.degrees(np.arctan2( rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        yaw   = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        roll  = 0.0

    return yaw, pitch, roll


def is_distracted(yaw, pitch):
    """Return True if head pose indicates looking away from screen."""
    if yaw is None:
        return True
    return abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD


# Standalone test
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please download it first using the provided plan.")
        exit()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    landmarker = FaceLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    distracted_frames = 0
    FPS = 30
    LIMIT = FPS * 3  # 3 seconds

    print("Head Pose Module (Tasks API) ready. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        
        # Convert OpenCV frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect landmarks
        result = landmarker.detect(mp_image)

        status = "FOCUSED"
        color  = (0, 200, 0)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            yaw, pitch, roll = get_head_pose(lm, w, h)

            if yaw is not None:
                cv2.putText(frame, f"Yaw:   {yaw:+.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                cv2.putText(frame, f"Pitch: {pitch:+.1f}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                cv2.putText(frame, f"Roll:  {roll:+.1f}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                if is_distracted(yaw, pitch):
                    distracted_frames += 1
                else:
                    distracted_frames = max(0, distracted_frames - 2)

                if distracted_frames >= LIMIT:
                    status = "DISTRACTED"
                    color  = (0, 0, 255)
        else:
            distracted_frames += 1
            status = "NO FACE"
            color  = (0, 165, 255)

        cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
        cv2.putText(frame, f"Head Pose: {status}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Head Pose Module (Tasks API)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
