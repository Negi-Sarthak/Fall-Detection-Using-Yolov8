# 🛡️ Fall Detection — Edge AI on Jetson Nano

Real-time human fall detection system powered by **YOLOv8 Pose Estimation**, designed and optimized to run on the **NVIDIA Jetson Nano**. The system processes live webcam or pre-recorded video feeds, detects falls using skeletal keypoint analysis, and automatically saves annotated video clips with a timestamped log.

---

## 📋 Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Fall Detection Logic](#fall-detection-logic)

---

## ✨ Features

- 🎯 **Real-time pose estimation** using YOLOv8n-pose (nano variant for edge efficiency)
- 🔍 **Keypoint-based fall detection** — no dataset training required
- 📹 **Automatic clip saving** — pre-fall buffer + post-fall footage stitched together
- 📝 **CSV logging** of every fall event with timestamp and clip path
- 🖥️ **On-screen overlay** — live FPS counter, fall count, and alert banner
- ⚡ **Threaded clip saving** — keeps inference uninterrupted during disk I/O
- 📷 **Dual input support** — webcam or pre-recorded `.mp4` video

---

## ⚙️ How It Works

```
Camera / Video
     │
     ▼
YOLOv8n-Pose ──► Keypoints (17 joints per person)
     │
     ▼
Fall Logic
  ├─ Hip downward velocity > 8% of body height?
  └─ Hip-shoulder gap collapsed (normalized gap < 0.3)?
          │ YES
          ▼
   Fall Triggered
  ├─ Alert banner drawn on frame
  ├─ Pre-fall buffer + post-fall frames stitched
  ├─ Clip saved to /clips/
  └─ Entry written to /logs/fall_log.csv
```

---

## 📁 Project Structure

```
fall_detection_project/
├── main.py                  # Entry point — capture loop, orchestration
├── requirements.txt         # Python dependencies
├── models/
│   └── yolov8n-pose.pt      # YOLOv8 nano pose model (download separately)
├── utils/
│   ├── fall_logic.py        # Keypoint-based fall detection algorithm
│   ├── video_utils.py       # Clip saving logic and buffer constants
│   └── overlay.py           # On-screen drawing (alert banner, FPS, counters)
├── data/
│   └── test_video.mp4       # Optional: pre-recorded test video
├── clips/                   # Auto-created: saved fall video clips
└── logs/
    └── fall_log.csv         # Auto-created: fall event log
```

---

## 🔧 Hardware Requirements

| Component | Minimum Spec |
|---|---|
| **Board** | NVIDIA Jetson Nano (4 GB recommended) |
| **Camera** | USB webcam or CSI camera module |
| **Storage** | 32 GB microSD (Class 10 / U3) |
| **Power** | 5V 4A barrel jack (barrel jack mode recommended) |
| **RAM** | 4 GB LPDDR4 (shared with GPU) |

> ⚠️ The Jetson Nano 2 GB variant may struggle with YOLOv8n-pose. Use the 4 GB model for best results.

---

## 💻 Software Requirements

| Software | Version |
|---|---|
| JetPack SDK | 4.6.x |
| Python | 3.8+ |
| CUDA | 10.2 (bundled with JetPack 4.6) |
| OpenCV | 4.8+ |
| Ultralytics (YOLOv8) | 8.0+ |

---


## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fall-detection-jetson.git
cd fall-detection-jetson/fall_detection_project
```

### 2. Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

> 💡 On Jetson Nano, `opencv-python` from pip may conflict with the system OpenCV. If you face issues, skip it and use the pre-installed system OpenCV:
> ```bash
> pip3 install ultralytics
> ```

### 3. Download the YOLOv8 Pose Model

```bash
mkdir -p models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
cd ..
```

Or let it auto-download on first run (requires internet connection on the Nano).

### 4. Set Device to GPU

In `main.py`, change the `DEVICE` constant from `"mps"` (Apple Silicon) to `"0"` (NVIDIA GPU):

```python
# main.py line 15
DEVICE = "0"    # Use NVIDIA GPU on Jetson Nano
```

---

## ▶️ Usage

### Run with Webcam (Live)

```bash
python3 main.py
```

### Run with Pre-recorded Video

Place your video at `data/test_video.mp4`, then:

```bash
python3 main.py video
```

### Quit

Press **`Q`** in the display window to stop the session.

---

## ⚙️ Configuration

All tunable parameters are near the top of their respective files:

| Parameter | File | Default | Description |
|---|---|---|---|
| `MODEL_PATH` | `main.py` | `models/yolov8n-pose.pt` | Path to YOLOv8 pose model |
| `DEVICE` | `main.py` | `"mps"` → set to `"0"` | Inference device (`"0"` = GPU, `"cpu"`) |
| `ALERT_DURATION` | `main.py` | `3` seconds | Duration the alert banner stays on screen |
| `BUFFER_SECONDS` | `video_utils.py` | `5` seconds | Pre-fall rolling buffer duration |
| `POST_FALL_SECS` | `video_utils.py` | `5` seconds | Post-fall recording duration |
| `VELOCITY_WINDOW` | `fall_logic.py` | `8` frames | Frames used to compute hip velocity |

---

## 📤 Output

### Saved Clips — `clips/`

Every detected fall generates an annotated `.mp4` clip:
```
clips/fall_2026-05-08_19-30-00.mp4
```
Contains 5 seconds before + 5 seconds after the fall event, with pose skeleton overlaid.

### Fall Log — `logs/fall_log.csv`

```csv
fall_number, timestamp,           clip_path
1,           2026-05-08 19:30:00, clips/fall_2026-05-08_19-30-00.mp4
2,           2026-05-08 19:45:12, clips/fall_2026-05-08_19-45-12.mp4
```

---

## 🧠 Fall Detection Logic

Fall detection is purely geometry-based — no ML classifier training needed.

**Two conditions must both be true simultaneously:**

1. **High downward hip velocity** — The average hip Y-coordinate moves down by more than **8% of body height** over the last 8 frames.
2. **Collapsed hip-shoulder gap** — The normalized hip-to-shoulder distance drops below **0.3**, indicating the person is close to horizontal.

```
normalized_gap = (avg_hip_y - avg_shoulder_y) / body_height

Fall = (velocity > body_height × 0.08) AND (normalized_gap < 0.3)
```

This two-condition gate eliminates false positives from fast walking or bending.

---

## 📄 License

This project is intended for academic and research purposes.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — pose estimation backbone
- [NVIDIA Jetson](https://developer.nvidia.com/embedded/jetson-nano) — edge AI platform
- [OpenCV](https://opencv.org/) — video capture and rendering
