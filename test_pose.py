import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

img = cv2.imread('face.jpg')

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    output_facial_transformation_matrixes=True
)

landmarker = FaceLandmarker.create_from_options(options)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
result = landmarker.detect(mp_image)

if result.facial_transformation_matrixes:
    m = result.facial_transformation_matrixes[0]
    print("Matrix type:", type(m))
    print(m)
else:
    print("No faces or matrices detected.")
