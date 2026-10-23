import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- Page Configuration ---
st.set_page_config(page_title="Drowsy Driver Detection Demo", page_icon="🚗", layout="centered")

st.title("🚗 Real-Time Drowsy Driver Detection Demo")
st.write("This computer vision model tracks Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to detect microsleep and fatigue in real time.")

# --- Thresholds ---
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.65
SLEEP_FRAMES_LIMIT = 25
YAWN_FRAMES_LIMIT = 20

# MediaPipe Indices
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

# --- Video Processor Class ---
class DrowsyDetectorProcessor(VideoTransformerBase):
    def __init__(self):
        # Safe fallback check for MediaPipe initialization across versions
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            import mediapipe.python.solutions.face_mesh as fm
            self.face_mesh = fm.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
        self.sleep_frame_counter = 0
        self.yawn_frame_counter = 0
        self.total_yawns = 0
        self.yawn_registered = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        status_text = "STATUS: ACTIVE & ALERT"
        status_color = (0, 255, 0) # Green

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

                left_eye = [landmarks[i] for i in LEFT_EYE]
                right_eye = [landmarks[i] for i in RIGHT_EYE]
                mouth = [landmarks[i] for i in INNER_LIPS]

                cv2.polylines(img, [np.array(left_eye)], True, (0, 255, 255), 1)
                cv2.polylines(img, [np.array(right_eye)], True, (0, 255, 255), 1)
                cv2.polylines(img, [np.array(mouth)], True, (255, 255, 0), 1)

                ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
                mar = calculate_mar(mouth)

                # Eye Closure Check
                if ear < EAR_THRESHOLD:
                    self.sleep_frame_counter += 1
                    if self.sleep_frame_counter >= SLEEP_FRAMES_LIMIT:
                        status_text = "DANGER! WAKE UP! DROWSY!"
                        status_color = (0, 0, 255) # Red
                        cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 10)
                else:
                    self.sleep_frame_counter = 0

                # Yawn Check
                if mar > MAR_THRESHOLD:
                    self.yawn_frame_counter += 1
                    if self.yawn_frame_counter >= YAWN_FRAMES_LIMIT:
                        if not self.yawn_registered:
                            self.total_yawns += 1
                            self.yawn_registered = True
                        if self.sleep_frame_counter < SLEEP_FRAMES_LIMIT:
                            status_text = "WARNING: YAWNING DETECTED"
                            status_color = (0, 165, 255) # Orange
                else:
                    self.yawn_frame_counter = 0
                    self.yawn_registered = False

                # Overlay Text
                cv2.putText(img, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, f"Yawns: {self.total_yawns}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Status Banner
        cv2.rectangle(img, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(img, status_text, (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return img

# --- Streamlit WebRTC Component ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="drowsy-detector",
    video_processor_factory=DrowsyDetectorProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

st.markdown("---")
st.markdown("### How to use:")
st.write("1. Click **START** to give browser permission to your webcam.")
st.write("2. Close your eyes for ~1 second or yawn to see the alert trigger in real time.")