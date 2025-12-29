import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="😴",
    layout="centered"
)

st.title("😴 Live Drowsiness Detection")
st.write("Webcam based eye-closure detection using MediaPipe")

# ---------------- EAR FUNCTION ----------------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ---------------- MEDIAPIPE ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
DROWSY_TIME = 2  # seconds

# ---------------- STREAMLIT UI ----------------
run = st.checkbox("Start Camera")
frame_window = st.image([])

cap = None
start_time = None

while run:
    if cap is None:
        cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status = "AWAKE"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = np.array([
                [int(face_landmarks.landmark[i].x * w),
                 int(face_landmarks.landmark[i].y * h)]
                for i in LEFT_EYE
            ])

            right_eye = np.array([
                [int(face_landmarks.landmark[i].x * w),
                 int(face_landmarks.landmark[i].y * h)]
                for i in RIGHT_EYE
            ])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time > DROWSY_TIME:
                    status = "DROWSY 😴"
                    cv2.putText(frame, "DROWSINESS ALERT!",
                                (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 255), 3)
            else:
                start_time = None

            cv2.putText(frame, f"Status: {status}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    frame_window.image(frame, channels="BGR")

if cap:
    cap.release()
