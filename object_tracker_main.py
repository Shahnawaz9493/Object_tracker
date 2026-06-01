"""
object_tracker_main.py
======================
Real-time Object Detection + SORT Tracking Pipeline

Usage:
    python object_tracker_main.py                  # webcam
    python object_tracker_main.py --source video.mp4
    python object_tracker_main.py --source video.mp4 --yolo yolov8n.pt
    python object_tracker_main.py --source video.mp4 --no-display --save output.mp4

Architecture:
  ┌────────────┐   frames   ┌──────────────┐   detections   ┌──────────────┐
  │  Video     │ ─────────► │   Detector   │ ─────────────► │ SORT Tracker │
  │  Source    │            │  (YOLO/SSD/  │                │  (Kalman +   │
  │ (cam/file) │            │   Color Seg) │                │  Hungarian)  │
  └────────────┘            └──────────────┘                └──────┬───────┘
                                                                   │ tracks
                                                             ┌─────▼──────┐
                                                             │  Annotator │
                                                             │  + Display │
                                                             └────────────┘
"""

import argparse
import time
import sys
import os

import cv2
import numpy as np

# ── add local modules to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from sort_tracker import SORT
from detector import build_detector, COLORS


# ─── Colour palette for track IDs ─────────────────────────────────────────
def id_color(track_id: int):
    """Return a stable, visually distinct BGR colour for a track ID."""
    np.random.seed(track_id * 17 + 3)
    return tuple(int(c) for c in np.random.randint(80, 230, 3))


# ─── Annotation helpers ────────────────────────────────────────────────────
def draw_fancy_box(frame, x1, y1, x2, y2, color, thickness=2, corner_len=15):
    """Bounding box with corner accents instead of full rectangle."""
    # Thin full rect
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
    # Corner L-shapes
    for px, py, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px,py), (px+sx*corner_len,py), color, thickness)
        cv2.line(frame, (px,py), (px,py+sy*corner_len), color, thickness)


def draw_label(frame, text, x, y, color, font_scale=0.55, thickness=1):
    """Draw label with a filled background pill."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    # Background
    cv2.rectangle(frame, (x-pad, y-th-pad), (x+tw+pad, y+baseline+pad), color, -1)
    # Text (white or black for contrast)
    brightness = sum(color) / 3
    txt_color = (0,0,0) if brightness > 150 else (255,255,255)
    cv2.putText(frame, text, (x, y), font, font_scale, txt_color, thickness, cv2.LINE_AA)


def draw_trail(frame, trail, color, max_len=25):
    """Draw motion trail of centroid positions."""
    pts = list(trail)[-max_len:]
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, pts[i-1], pts[i], c, max(1, int(2*alpha)))


def draw_hud(frame, fps, total_dets, total_tracks, detector_name, frame_num, total_frames=0):
    """Draw a semi-transparent HUD overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top-left HUD box
    cv2.rectangle(overlay, (0,0), (320, 130), (20,20,20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    lines = [
        (f"Detector : {detector_name}",         (10, 22)),
        (f"FPS      : {fps:5.1f}",              (10, 46)),
        (f"Detections: {total_dets:3d}",        (10, 70)),
        (f"Tracks    : {total_tracks:3d}",      (10, 94)),
        (f"Frame     : {frame_num}"
         + (f"/{total_frames}" if total_frames else ""), (10, 118)),
    ]
    for text, pos in lines:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (220,220,220), 1, cv2.LINE_AA)

    # Progress bar (if video file)
    if total_frames > 0:
        bar_w = w - 20
        progress = frame_num / total_frames
        cv2.rectangle(frame, (10, h-12), (10+bar_w, h-6), (60,60,60), -1)
        cv2.rectangle(frame, (10, h-12), (10+int(bar_w*progress), h-6), (0,200,100), -1)


# ─── Main pipeline ─────────────────────────────────────────────────────────
def run(source=0,
        yolo_path='yolov8n.pt',
        mobilenet_proto='MobileNetSSD_deploy.prototxt',
        mobilenet_model='MobileNetSSD_deploy.caffemodel',
        conf=0.40,
        max_age=5,
        min_hits=2,
        iou_thresh=0.30,
        display=True,
        save_path=None,
        max_frames=None):

    # ── Detector ──────────────────────────────────────────────────────────
    detector = build_detector(
        yolo_path=yolo_path,
        mobilenet_proto=mobilenet_proto,
        mobilenet_model=mobilenet_model,
        conf=conf,
    )
    print(f"[Pipeline] Using detector: {detector.name}")

    # ── Tracker ───────────────────────────────────────────────────────────
    tracker = SORT(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)
    print(f"[Pipeline] SORT tracker  max_age={max_age} min_hits={min_hits} "
          f"iou={iou_thresh}")

    # ── Video source ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Pipeline] Source {source}  {W}x{H} @ {fps_src:.1f}fps  frames={total_f}")

    # ── Writer ────────────────────────────────────────────────────────────
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps_src, (W, H))
        print(f"[Pipeline] Saving output → {save_path}")

    # ── State ─────────────────────────────────────────────────────────────
    trails    = {}   # track_id → deque of (cx, cy)
    id_labels = {}   # track_id → last seen label
    from collections import deque

    frame_num   = 0
    fps_history = []
    t_start     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_num >= max_frames:
            break
        frame_num += 1

        t0 = time.time()

        # ── Detect ──────────────────────────────────────────────────────
        raw_dets = detector.detect(frame)

        # Build np array [x1,y1,x2,y2,score]
        det_array = np.array(
            [[*d['bbox'], d['conf']] for d in raw_dets],
            dtype=np.float32
        ).reshape(-1, 5) if raw_dets else np.empty((0, 5))

        # Build label lookup  bbox → label
        label_map = {}
        for d in raw_dets:
            key = tuple(int(v) for v in d['bbox'])
            label_map[key] = (d['label'], d['class_id'])

        # ── Track ───────────────────────────────────────────────────────
        tracks = tracker.update(det_array)  # [x1,y1,x2,y2,tid]

        # ── Annotate ────────────────────────────────────────────────────
        for d in raw_dets:
            x1,y1,x2,y2 = [int(v) for v in d['bbox']]
            det_color = detector.color_for(d['label'])
            # Light semi-transparent fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1,y1),(x2,y2), det_color, -1)
            frame = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)

        for track in tracks:
            x1,y1,x2,y2,tid = [int(v) for v in track[:5]]
            cx, cy = (x1+x2)//2, (y1+y2)//2
            color   = id_color(int(tid))

            # Trail
            if tid not in trails:
                trails[tid] = deque(maxlen=40)
            trails[tid].append((cx, cy))
            draw_trail(frame, trails[tid], color)

            # Fancy box
            draw_fancy_box(frame, x1, y1, x2, y2, color, thickness=2)

            # Resolve label
            best_label = id_labels.get(tid, '?')
            for d in raw_dets:
                rx1,ry1,rx2,ry2 = [int(v) for v in d['bbox']]
                if abs(rx1-x1)<20 and abs(ry1-y1)<20:
                    best_label = d['label']
                    id_labels[tid] = best_label
                    break

            label_text = f"#{tid} {best_label}"
            draw_label(frame, label_text, x1, y1-2, color)

            # Centroid dot
            cv2.circle(frame, (cx,cy), 4, color, -1)

        # ── FPS & HUD ───────────────────────────────────────────────────
        elapsed = time.time() - t0
        fps_history.append(1.0 / elapsed if elapsed > 0 else 0)
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps_now = sum(fps_history) / len(fps_history)

        draw_hud(frame, fps_now, len(raw_dets), len(tracks),
                 detector.name, frame_num, total_f)

        # ── Output ──────────────────────────────────────────────────────
        if writer:
            writer.write(frame)

        if display:
            cv2.imshow("Object Detection + SORT Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[Pipeline] Quit by user")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    avg_fps    = frame_num / total_time if total_time > 0 else 0
    print(f"\n[Pipeline] Done  frames={frame_num}  "
          f"time={total_time:.1f}s  avg_fps={avg_fps:.1f}")
    return frame_num, avg_fps


# ─── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Object Detection + SORT Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--source',    default=0,
                   help='Video file path or camera index (default: 0 = webcam)')
    p.add_argument('--yolo',      default='yolov8n.pt',
                   help='Path to YOLOv8 .pt weights')
    p.add_argument('--proto',     default='MobileNetSSD_deploy.prototxt',
                   help='Path to MobileNet SSD prototxt')
    p.add_argument('--model',     default='MobileNetSSD_deploy.caffemodel',
                   help='Path to MobileNet SSD caffemodel')
    p.add_argument('--conf',      type=float, default=0.40,
                   help='Detection confidence threshold (default: 0.40)')
    p.add_argument('--max-age',   type=int,   default=5,
                   help='SORT max_age (default: 5)')
    p.add_argument('--min-hits',  type=int,   default=2,
                   help='SORT min_hits (default: 2)')
    p.add_argument('--iou',       type=float, default=0.30,
                   help='SORT IoU threshold (default: 0.30)')
    p.add_argument('--no-display', action='store_true',
                   help='Suppress OpenCV window (headless mode)')
    p.add_argument('--save',      default=None,
                   help='Path to save annotated output video')
    p.add_argument('--max-frames',type=int,   default=None,
                   help='Stop after N frames')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    run(
        source=source,
        yolo_path=args.yolo,
        mobilenet_proto=args.proto,
        mobilenet_model=args.model,
        conf=args.conf,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_thresh=args.iou,
        display=not args.no_display,
        save_path=args.save,
        max_frames=args.max_frames,
    )