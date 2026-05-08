import cv2

def draw_alert(frame, count):
    """Draws red fall alert banner on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"FALL DETECTED  (Total: {count})", (20, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 3)
    return frame

def draw_info(frame, fps_val, fall_count):
    """Draws FPS and fall count on the frame."""
    cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Falls: {fall_count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return frame