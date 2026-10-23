import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import winsound
import threading

# --- 1. Built-in Windows Audio Alert System (No Pygame Required) ---
alarm_playing = False

def sound_alarm():
    """Plays a loud 2500Hz wake-up beep continuously while drowsy."""
    global alarm_playing
    while alarm_playing:
        # 2500 Hz frequency for 400 milliseconds
        winsound.Beep(2500, 400)

# --- 2. Drowsiness Thresholds ---
EAR_THRESHOLD = 0.22         # Eye closure threshold
MAR_THRESHOLD = 0.65         # Yawn opening threshold
SLEEP_FRAMES_LIMIT = 25      # Consecutive frames before microsleep alarm (~1 sec)
YAWN_FRAMES_LIMIT = 20       # Consecutive frames to register 1 yawn

# MediaPipe Facial Landmark Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
INNER_LIPS = [78, 81, 13, 311, 308, 402, 14, 178]

def calculate_ear(eye_pts):
    v1 = dist.euclidean(eye_pts[1], eye_pts[5])
    v2 = dist.euclidean(eye_pts[2], eye_pts[4])
    horiz = dist.euclidean(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * horiz)

def calculate_mar(lip_pts):
    v1 = dist.euclidean(lip_pts[1], lip_pts[7])
    v2 = dist.euclidean(lip_pts[2], lip_pts[6])
    v3 = dist.euclidean(lip_pts[3], lip_pts[5])
    horiz = dist.euclidean(lip_pts[0], lip_pts[4])
    return (v1 + v2 + v3) / (3.0 * horiz)

# --- 3. Initialize MediaPipe & Webcam ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

sleep_frame_counter = 0
yawn_frame_counter = 0
total_yawns = 0
yawn_registered = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status_text = "STATUS: ACTIVE & ALERT"
    status_color = (0, 255, 0)  # Green

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

            left_eye = [landmarks[i] for i in LEFT_EYE]
            right_eye = [landmarks[i] for i in RIGHT_EYE]
            mouth = [landmarks[i] for i in INNER_LIPS]

            # Draw contours around eyes and mouth
            cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 255), 1)
            cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 255), 1)
            cv2.polylines(frame, [np.array(mouth)], True, (255, 255, 0), 1)

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)

            # --- A. Check for Prolonged Eye Closure (Microsleep) ---
            if ear < EAR_THRESHOLD:
                sleep_frame_counter += 1
                if sleep_frame_counter >= SLEEP_FRAMES_LIMIT:
                    status_text = "DANGER! WAKE UP! DROWSINESS DETECTED!"
                    status_color = (0, 0, 255)  # Red
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                    if not alarm_playing:
                        alarm_playing = True
                        threading.Thread(target=sound_alarm, daemon=True).start()
            else:
                sleep_frame_counter = 0
                alarm_playing = False

            # --- B. Check for Yawning (Early Fatigue Warning) ---
            if mar > MAR_THRESHOLD:
                yawn_frame_counter += 1
                if yawn_frame_counter >= YAWN_FRAMES_LIMIT:
                    if not yawn_registered:
                        total_yawns += 1
                        yawn_registered = True
                    if sleep_frame_counter < SLEEP_FRAMES_LIMIT:
                        status_text = "WARNING: YAWNING DETECTED - TAKE A BREAK"
                        status_color = (0, 165, 255)  # Orange
            else:
                yawn_frame_counter = 0
                yawn_registered = False

            # Display Live Telemetry Dashboard
            cv2.rectangle(frame, (10, 10), (330, 115), (0, 0, 0), -1)
            cv2.putText(frame, f"Eye Ratio (EAR): {ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Mouth Ratio (MAR): {mar:.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Total Yawns: {total_yawns}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    else:
        status_text = "WARNING: DRIVER FACE NOT VISIBLE!"
        status_color = (0, 165, 255)
        alarm_playing = False

    # Bottom Status Banner
    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow("Anti-Sleep Driver Safety System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

alarm_playing = False
cap.release()
cv2.destroyAllWindows()