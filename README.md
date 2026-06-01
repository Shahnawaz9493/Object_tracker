<div align="center">

# 🎯 ObjectTrackX

### Real-Time Object Detection & Multi-Object Tracking

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)](https://ultralytics.com)
[![SORT](https://img.shields.io/badge/Tracker-SORT-orange?style=for-the-badge)](https://arxiv.org/abs/1602.00763)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

<br/>

> **Detect. Track. Identify.** — A production-ready pipeline combining YOLOv8 object detection with a from-scratch SORT tracker (Kalman Filter + Hungarian Algorithm) for real-time multi-object tracking with persistent IDs, motion trails, and a live HUD.

<br/>

```
Video Input ──► YOLOv8 Detector ──► SORT Tracker ──► Annotated Output
 (cam/mp4)       80 COCO classes    Kalman + IoU     IDs · Trails · HUD
```

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Multi-backend detection** | YOLOv8 → MobileNet SSD → Color Seg (auto-fallback) |
| 🧠 **SORT tracker (from scratch)** | Kalman Filter + Hungarian Algorithm, no external tracker lib |
| 🎨 **Rich annotation** | Corner-accent boxes, motion trails, label pills, centroid dots |
| 📊 **Live HUD** | FPS counter, detection count, track count, progress bar |
| 🎥 **Flexible input** | Webcam, video file, or any OpenCV-compatible source |
| 💾 **Output saving** | Export annotated video to MP4 |
| 🖥️ **Headless mode** | Run on servers with no display (`--no-display`) |
| ⚡ **Zero-crash fallback** | Runs without any model weights using color segmentation |

---

## 📁 Project Structure

```
ObjectTrackX/
│
├── object_tracker_main.py   # Main pipeline — video I/O, annotation, HUD
├── sort_tracker.py          # SORT tracker (Kalman Filter + Hungarian matching)
├── detector.py              # Multi-backend detector (YOLO / SSD / Color Seg)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/ObjectTrackX.git
cd ObjectTrackX
pip install -r requirements.txt
```

### 2. Download YOLOv8 Weights (one-time)

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Run

```bash
# Webcam
python object_tracker_main.py --yolo yolov8n.pt

# Video file
python object_tracker_main.py --source video.mp4 --yolo yolov8n.pt

# Save output
python object_tracker_main.py --source video.mp4 --yolo yolov8n.pt --save output.mp4

# Headless (no window)
python object_tracker_main.py --source video.mp4 --no-display --save output.mp4
```

---

## 🎮 All CLI Arguments

```
python object_tracker_main.py [OPTIONS]

Options:
  --source PATH        Video file or camera index (default: 0 = webcam)
  --yolo PATH          YOLOv8 .pt weights file       (best accuracy)
  --proto PATH         MobileNet SSD .prototxt file   (fallback)
  --model PATH         MobileNet SSD .caffemodel file (fallback)
  --conf FLOAT         Detection confidence threshold (default: 0.40)
  --max-age INT        SORT: frames a track lives without detection (default: 5)
  --min-hits INT       SORT: frames before a track is confirmed (default: 2)
  --iou FLOAT          SORT: min IoU for detection→track match (default: 0.30)
  --no-display         Suppress OpenCV window (headless/server mode)
  --save PATH          Save annotated output to MP4
  --max-frames INT     Stop after N frames
```

---

## 🏗️ Architecture Deep Dive

### Detection Pipeline

The detector auto-selects the best available backend at startup:

```
1. YOLOv8 (ultralytics)       ← Best: 80 COCO classes, GPU/CPU
       ↓ not found?
2. MobileNet SSD (OpenCV DNN) ← Good: 20 classes, fast CPU
       ↓ not found?
3. Color Segmentation         ← Fallback: no weights needed, demo only
```

### SORT Tracker Internals

```
Every frame:
  1. Predict  ── all existing tracks run Kalman predict step
  2. Match    ── build IoU matrix (detections × predictions)
                 solve with Hungarian algorithm
  3. Update   ── matched tracks run Kalman update with new measurement
  4. Spawn    ── unmatched detections become new KalmanBoxTracker instances
  5. Kill     ── tracks with time_since_update > max_age are removed
  6. Report   ── tracks with hit_streak ≥ min_hits are returned
```

### Kalman Filter State Vector

```
State:       [ cx,  cy,   w,   h,  vcx,  vcy,  vw,  vh ]
              ↑pos  ↑pos ↑size ↑size ↑vel  ↑vel ↑vel ↑vel

Observation: [ cx,  cy,   w,   h ]     (4D measurement from detector)
```

---

## ⚙️ SORT Hyperparameter Guide

| Parameter | Default | Lower → | Higher → |
|---|---|---|---|
| `--max-age` | `5` | Tracks die faster, less ghosting | Tracks survive occlusion longer |
| `--min-hits` | `2` | New tracks appear immediately | More frames needed to confirm a track |
| `--iou` | `0.30` | More lenient matching | Stricter matching, fewer false links |

**Recommended presets:**

```bash
# Fast-moving objects (sports, traffic)
--max-age 3 --min-hits 1 --iou 0.25

# Slow/dense scenes (crowds, retail)
--max-age 8 --min-hits 3 --iou 0.35

# Minimal latency (real-time demo)
--max-age 5 --min-hits 1 --conf 0.3
```

---

## 🔍 Model Comparison

| Model | Size | Speed (CPU) | Classes | Best For |
|---|---|---|---|---|
| `yolov8n.pt` | 6 MB | ~15 FPS | 80 | Real-time on CPU |
| `yolov8s.pt` | 22 MB | ~10 FPS | 80 | Balanced accuracy |
| `yolov8m.pt` | 50 MB | ~6 FPS | 80 | Higher accuracy |
| `yolov8l.pt` | 87 MB | ~4 FPS | 80 | Maximum accuracy |
| MobileNet SSD | 23 MB | ~20 FPS | 20 | Fastest CPU option |

---

## 🖥️ Output Annotation

Every frame is annotated with:

- **Corner-accent bounding boxes** — each track ID gets a stable unique color
- **Motion trails** — last 40 centroid positions drawn as a fading path
- **Label pills** — `#ID classname` with auto-contrast text (black or white)
- **Centroid dot** — marks the center of each tracked object
- **HUD overlay** — detector name, FPS, detection count, track count, frame number, progress bar

---

## 🐛 Troubleshooting

**Seeing "ColorSegment" instead of YOLOv8?**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python object_tracker_main.py --yolo yolov8n.pt
```

**Wrong camera?**
```bash
python object_tracker_main.py --source 1   # try index 1, 2, etc.
```

**No display / server environment?**
```bash
python object_tracker_main.py --source video.mp4 --no-display --save out.mp4
```

**Too slow on CPU?**
```bash
python object_tracker_main.py --yolo yolov8n.pt --conf 0.3
```

**Tracks flickering / IDs changing too often?**
```bash
python object_tracker_main.py --max-age 8 --min-hits 1 --iou 0.2
```

---

## 📦 Dependencies

```txt
opencv-python >= 4.8.0     # Video I/O, drawing, DNN backend
numpy >= 1.24              # Array ops
scipy >= 1.10              # Hungarian algorithm
ultralytics >= 8.0         # YOLOv8 (optional but recommended)
```

Install all:
```bash
pip install opencv-python numpy scipy ultralytics
```

---

## 📚 References

- [SORT: Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763) — Bewley et al., 2016
- [YOLOv8 by Ultralytics](https://docs.ultralytics.com/)
- [Deep SORT](https://arxiv.org/abs/1703.07402) — appearance-based re-ID (next step)
- [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) — optimal assignment

---

## 🗺️ Roadmap

- [ ] Deep SORT integration (appearance re-ID with ReID embeddings)
- [ ] ByteTrack support
- [ ] GPU acceleration via TensorRT / ONNX
- [ ] Speed estimation (pixels/frame → km/h with calibration)
- [ ] Zone-based counting (draw a line, count crossings)
- [ ] REST API / WebSocket streaming output
- [ ] Streamlit dashboard for live monitoring

---

<div align="center">

Built with ❤️ using OpenCV · YOLOv8 · SORT · Python

</div>
