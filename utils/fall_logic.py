from collections import deque

VELOCITY_WINDOW = 8
hip_y_history = deque(maxlen=VELOCITY_WINDOW)

LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14

def get_keypoint(keypoints, idx):
    kp = keypoints[idx]
    return float(kp[0]), float(kp[1]), float(kp[2])

def is_fall(keypoints):
    try:
        _, ls_y, ls_conf = get_keypoint(keypoints, LEFT_SHOULDER)
        _, rs_y, rs_conf = get_keypoint(keypoints, RIGHT_SHOULDER)
        _, lh_y, lh_conf = get_keypoint(keypoints, LEFT_HIP)
        _, rh_y, rh_conf = get_keypoint(keypoints, RIGHT_HIP)
        _, lk_y, lk_conf = get_keypoint(keypoints, LEFT_KNEE)
        _, rk_y, rk_conf = get_keypoint(keypoints, RIGHT_KNEE)

        # Need confident detections
        if min(ls_conf, rs_conf, lh_conf, rh_conf) < 0.3:
            return False

        avg_shoulder_y = (ls_y + rs_y) / 2
        avg_hip_y      = (lh_y + rh_y) / 2
        avg_knee_y     = (lk_y + rk_y) / 2

        hip_y_history.append(avg_hip_y)

        # Velocity of hip movement
        velocity = 0
        if len(hip_y_history) == VELOCITY_WINDOW:
            velocity = hip_y_history[-1] - hip_y_history[0]  # positive = moving down

        # Body height (shoulder to knee distance)
        body_height = abs(avg_knee_y - avg_shoulder_y) + 1e-5

        # Hip-shoulder gap normalized by body height
        # When standing: hips well below shoulders (large positive value)
        # When fallen: hips close to shoulder level (small value)
        normalized_gap = (avg_hip_y - avg_shoulder_y) / body_height

        # Fall condition:
        # 1. Hips dropped unusually fast (velocity > 8% of body height)
        # 2. Hip-shoulder gap collapsed (person is more horizontal)
        if velocity > (body_height * 0.08) and normalized_gap < 0.3:
            return True

    except Exception:
        pass

    return False