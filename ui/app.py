import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import json
from datetime import datetime

from modules.head_pose      import get_head_pose, is_distracted as head_is_distracted
from modules.gaze_tracker   import analyse_gaze
from modules.phone_detector import detect_distractors, draw_detections
from modules.attention_score import AttentionScoreEngine

# ── MediaPipe Tasks API Config ──────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_landmarker.task")

@st.cache_resource
def get_landmarker():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )
    return FaceLandmarker.create_from_options(options)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Focus Guardian",
    page_icon="🎯",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .focus-score-big {
        font-size: 72px;
        font-weight: 700;
        text-align: center;
        line-height: 1;
    }
    .nudge-box {
        padding: 12px 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin: 8px 0;
    }
    .nudge-0 { background: #1a3a1a; color: #4ade80; }
    .nudge-1 { background: #3a2a00; color: #fbbf24; }
    .nudge-2 { background: #3a1a00; color: #fb923c; }
    .nudge-3 { background: #3a0000; color: #f87171; }
    .nudge-4 { background: #2a0030; color: #e879f9; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        margin: 4px 0;
    }
    .signal-ok  { color: #4ade80; font-weight: 600; }
    .signal-bad { color: #f87171; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "engine"       not in st.session_state: st.session_state.engine       = None
if "running"      not in st.session_state: st.session_state.running      = False
if "frame_count"  not in st.session_state: st.session_state.frame_count  = 0
if "last_detections" not in st.session_state: st.session_state.last_detections = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Focus Guardian")
    st.caption("Privacy-first distraction detection. Everything runs locally.")
    st.divider()

    cam_index = st.selectbox("Camera index", [0, 1, 2], index=0)
    enable_yolo = st.toggle("Enable phone/object detection", value=True)

    st.divider()
    st.subheader("Thresholds")
    yaw_thresh   = st.slider("Head yaw threshold (°)",   10, 45, 20)
    pitch_thresh = st.slider("Head pitch threshold (°)", 10, 30, 15)
    ear_thresh   = st.slider("Drowsiness EAR", 0.10, 0.30, 0.20, step=0.01)

    st.divider()
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Start", use_container_width=True, type="primary")
    stop_btn  = col2.button("⏹ Stop",  use_container_width=True)

    if start_btn:
        st.session_state.running     = True
        st.session_state.engine      = AttentionScoreEngine()
        st.session_state.frame_count = 0
        st.session_state.last_detections = []

    if stop_btn:
        st.session_state.running = False
        if st.session_state.engine:
            path = st.session_state.engine.save_session()
            st.success(f"Session saved!")

    st.divider()
    st.subheader("Past sessions")
    session_dir = "data/sessions"
    if os.path.exists(session_dir):
        files = sorted(os.listdir(session_dir), reverse=True)[:5]
        if files:
            for f in files:
                try:
                    with open(os.path.join(session_dir, f)) as fh:
                        d = json.load(fh)
                    st.caption(f"📄 {f[:19]}  Score: {d.get('final_score','?')}")
                except Exception: pass
        else:
            st.caption("No sessions yet.")


# ── Main layout ───────────────────────────────────────────────────────────────
st.title("Focus Guardian 🎯")

if not st.session_state.running:
    st.info("Press **▶ Start** in the sidebar to begin a focus session.")
    st.markdown("""
    ### How it works
    - **Head pose** — detects if you look away from the screen
    - **Gaze tracker** — detects drowsiness and off-screen eye movement
    - **Object detector** — detects phones and other distractors on your desk
    - **Focus score** — combines all signals into a live 0–100 score
    - **Nudge system** — gently reminds you to refocus, not punish you

    > Everything runs **locally on your device**. No data ever leaves your computer.
    """)
    st.stop()

# ── Live session layout ───────────────────────────────────────────────────────
col_cam, col_stats = st.columns([3, 2])

with col_cam:
    st.subheader("Live feed")
    frame_placeholder = st.empty()
    nudge_placeholder  = st.empty()

with col_stats:
    st.subheader("Focus score")
    score_placeholder   = st.empty()
    signals_placeholder = st.empty()
    timer_placeholder   = st.empty()

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
landmarker = get_landmarker()

if not cap.isOpened():
    st.error(f"Could not open camera at index {cam_index}. Try a different index in the sidebar.")
    st.stop()

if not landmarker:
    st.error("Face Landmarker could not be initialized. Check model path.")
    st.stop()

while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        st.warning("Lost camera feed.")
        break

    h, w = frame.shape[:2]
    fc   = st.session_state.frame_count

    # ── MediaPipe inference (Tasks API) ───────────────────────────────────
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)

    head_dist  = False
    gaze_dist  = False
    drowsy     = False
    yaw = pitch = roll = None

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # Head pose
        yaw, pitch, roll = get_head_pose(lm, w, h)
        head_dist = (
            yaw is not None and
            (abs(yaw) > yaw_thresh or abs(pitch) > pitch_thresh)
        )

        # Gaze
        gaze_data = analyse_gaze(lm, w, h)
        gaze_dist = gaze_data["looking_away"]
        drowsy    = gaze_data["avg_ear"] < ear_thresh

        # Draw eye indicators on frame
        ear_text = f"EAR: {gaze_data['avg_ear']:.2f}"
        cv2.putText(frame, ear_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

    # ── YOLO inference ───────────────────────────────────────────────────
    if enable_yolo:
        det = detect_distractors(frame, fc)
        if det is not None:
            st.session_state.last_detections = det
        frame = draw_detections(frame, st.session_state.last_detections)

    # ── Attention score update ───────────────────────────────────────────
    engine = st.session_state.engine
    state  = engine.update(head_dist, gaze_dist, drowsy,
                           st.session_state.last_detections)

    # ── Overlay on frame ─────────────────────────────────────────────────
    score     = state["score"]
    nudge_lvl = state["nudge_level"]

    # Score bar at top
    bar_color = (0,200,0) if score >= 70 else (0,165,255) if score >= 40 else (0,0,255)
    bar_w     = int(w * score / 100)
    cv2.rectangle(frame, (0, 0),    (w, 6),     (50,50,50), -1)
    cv2.rectangle(frame, (0, 0),    (bar_w, 6), bar_color,  -1)

    # Status text
    cv2.rectangle(frame, (0, 6), (w, 40), (20,20,20), -1)
    status_text = "FOCUSED"
    if drowsy:      status_text = "DROWSY"
    elif head_dist: status_text = "HEAD TURNED"
    elif gaze_dist: status_text = "LOOKING AWAY"
    elif st.session_state.last_detections:
        status_text = f"DISTRACTOR: {st.session_state.last_detections[0]['label'].upper()}"

    status_color = bar_color
    cv2.putText(frame, f"Score: {score:.0f}  |  {status_text}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Angles overlay
    if yaw is not None:
        cv2.putText(frame, f"Y:{yaw:+.0f} P:{pitch:+.0f}", (w-130, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)

    # ── Display frame ────────────────────────────────────────────────────
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # ── Nudge box ────────────────────────────────────────────────────────
    msg = engine.get_nudge_message()
    if msg:
        nudge_placeholder.markdown(
            f'<div class="nudge-box nudge-{nudge_lvl}">{msg}</div>',
            unsafe_allow_html=True
        )
    else:
        nudge_placeholder.markdown(
            '<div class="nudge-box nudge-0">✓ On track</div>',
            unsafe_allow_html=True
        )

    # ── Score panel ──────────────────────────────────────────────────────
    sc = engine.get_score_color()
    score_placeholder.markdown(
        f'<div class="focus-score-big" style="color:{"#4ade80" if sc=="green" else "#fb923c" if sc=="orange" else "#f87171"}">'
        f'{score:.0f}</div><p style="text-align:center;color:#888">/ 100</p>',
        unsafe_allow_html=True
    )

    # ── Signal indicators ────────────────────────────────────────────────
    def sig(ok, label):
        cls = "signal-ok" if ok else "signal-bad"
        icon = "✓" if ok else "✗"
        return f'<span class="{cls}">{icon} {label}</span>'

    signals_placeholder.markdown(
        f"""
        {sig(not head_dist,  "Head straight")} &nbsp;
        {sig(not gaze_dist,  "Gaze on screen")} &nbsp;
        {sig(not drowsy,     "Awake")} &nbsp;
        {sig(not bool(st.session_state.last_detections), "No distractors")}
        <br><br>
        <b>Session:</b> {int(state['session_elapsed']//60):02d}:{int(state['session_elapsed']%60):02d} &nbsp;
        <b>Streak:</b> {int(state['focused_streak'])}s
        """,
        unsafe_allow_html=True
    )

    st.session_state.frame_count += 1
    time.sleep(0.01)   # Slight delay for Streamlit UI stability

cap.release()
