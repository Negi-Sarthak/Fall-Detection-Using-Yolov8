import cv2
import os

FPS            = 20
BUFFER_SECONDS = 5
POST_FALL_SECS = 5

os.makedirs("clips", exist_ok=True)

def save_clip(pre_frames, post_frames, timestamp, fps=FPS):
    """
    Saves a video clip combining pre-fall buffer and post-fall frames.
    Input: pre_frames (deque), post_frames (list), timestamp (str)
    Output: saved file path (str)
    """
    filename = f"clips/fall_{timestamp.replace(':', '-').replace(' ', '_')}.mp4"
    h, w = list(pre_frames)[0].shape[:2]
    out = cv2.VideoWriter(filename,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))
    for f in list(pre_frames) + post_frames:
        out.write(f)
    out.release()
    print(f"[CLIP] Saved: {filename}")
    return filename