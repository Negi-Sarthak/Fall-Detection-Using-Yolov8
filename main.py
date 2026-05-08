import cv2
import time
import csv
import os
import sys
import threading
from collections import deque
from ultralytics import YOLO
from utils.fall_logic import is_fall
from utils.video_utils import save_clip, BUFFER_SECONDS, POST_FALL_SECS
from utils.overlay import draw_alert, draw_info

# ── Config ──────────────────────────────────────────
MODEL_PATH     = "models/yolov8n-pose.pt"
DEVICE         = "mps"    # Change to "0" on Jetson Nano
ALERT_DURATION = 3        # seconds to show alert banner
# ────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)

os.makedirs("logs", exist_ok=True)
log_file   = open("logs/fall_log.csv", "a", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["fall_number", "timestamp", "clip_path"])

frame_buffer     = deque(maxlen=BUFFER_SECONDS * 30)
fall_count       = 0
fall_time        = None
recording_post   = False
post_fall_frames = []
post_fall_start  = None
fps_history      = deque(maxlen=30)

# ── Source selection ─────────────────────────────────
if len(sys.argv) > 1 and sys.argv[1] == "video":
    cap = cv2.VideoCapture("data/test_video.mp4")
    print("Using pre-recorded video")
else:
    cap = cv2.VideoCapture(0)
    print("Using webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Fall detection running — press Q to quit")
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps_val   = 1 / (curr_time - prev_time + 1e-5)
    prev_time = curr_time
    fps_history.append(fps_val)

    # Run YOLOv8 pose inference
    results = model(frame, device=DEVICE, verbose=False)
    fall_this_frame = False
    annotated = frame.copy()

    for result in results:
        if result.keypoints is None:
            continue
        for person_kps in result.keypoints.data:
            if is_fall(person_kps):
                fall_this_frame = True
        annotated = result.plot()

    # Add frame to rolling pre-fall buffer
    frame_buffer.append(annotated.copy())

    # Trigger fall event
    if fall_this_frame and not recording_post:
        fall_count      += 1
        fall_time        = curr_time
        recording_post   = True
        post_fall_frames = []
        post_fall_start  = curr_time
        print(f"[FALL] #{fall_count} detected at {time.strftime('%H:%M:%S')}")

    # Record post-fall frames
    if recording_post:
        post_fall_frames.append(annotated.copy())
        if curr_time - post_fall_start >= POST_FALL_SECS:
            recording_post = False
            ts       = time.strftime("%Y-%m-%d %H:%M:%S")
            pre      = deque(frame_buffer)
            post     = post_fall_frames[:]
            save_fps = sum(fps_history) / len(fps_history) if fps_history else 15

            def save(pre=pre, post=post, ts=ts, fps=save_fps):
                path = save_clip(pre, post, ts, fps)
                log_writer.writerow([fall_count, ts, path])
                log_file.flush()

            threading.Thread(target=save, daemon=True).start()

    # Draw overlays
    annotated = draw_info(annotated, fps_val, fall_count)
    if fall_time and (curr_time - fall_time) < ALERT_DURATION:
        annotated = draw_alert(annotated, fall_count)

    cv2.imshow("Fall Detection — Edge AI", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
print(f"Session ended. Total falls logged: {fall_count}")