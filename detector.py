"""
detector.py – Multi-backend object detector
Supports:
  1. YOLOv8 via Ultralytics  (best, auto-selected when model file found)
  2. MobileNet SSD via OpenCV DNN  (fallback with caffemodel)
  3. Color/shape segmentation      (fallback demo detector, no weights needed)
"""

import cv2
import numpy as np
import os

# ── Class info ────────────────────────────────────────────────────────────────
COCO_CLASSES = [
    'person','bicycle','car','motorbike','aeroplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis',
    'snowboard','sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle','wine glass','cup',
    'fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa',
    'pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush',
]

MOBILENET_CLASSES = [
    'background','aeroplane','bicycle','bird','boat','bottle','bus','car',
    'cat','chair','cow','diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor',
]

# Distinct colors per class (BGR)
np.random.seed(42)
COLORS = {name: tuple(int(c) for c in np.random.randint(80, 230, 3))
          for name in COCO_CLASSES + MOBILENET_CLASSES}


# ─── Base class ───────────────────────────────────────────────────────────────
class BaseDetector:
    name = "base"

    def detect(self, frame):
        """
        Returns list of dicts:
          { 'bbox': [x1,y1,x2,y2], 'label': str, 'conf': float, 'class_id': int }
        """
        raise NotImplementedError

    def color_for(self, label):
        return COLORS.get(label, (0, 200, 255))


# ─── YOLOv8 (Ultralytics) ────────────────────────────────────────────────────
class YOLOv8Detector(BaseDetector):
    name = "YOLOv8"

    def __init__(self, model_path='yolov8n.pt', conf=0.4, device='cpu'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf  = conf
        self.device = device
        print(f"[YOLOv8] Loaded {model_path}")

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf,
                                     device=self.device, verbose=False)[0]
        dets = []
        for box in results.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            label = results.names[cls]
            dets.append({'bbox':[x1,y1,x2,y2],'label':label,
                         'conf':conf,'class_id':cls})
        return dets


# ─── MobileNet SSD (OpenCV DNN) ───────────────────────────────────────────────
class MobileNetSSDDetector(BaseDetector):
    name = "MobileNetSSD"

    def __init__(self, proto, model, conf=0.4):
        self.net  = cv2.dnn.readNetFromCaffe(proto, model)
        self.conf = conf
        self.classes = MOBILENET_CLASSES
        print(f"[MobileNetSSD] Loaded model")

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        dets = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.conf:
                continue
            cls = int(detections[0, 0, i, 1])
            if cls >= len(self.classes):
                continue
            label = self.classes[cls]
            if label == 'background':
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            dets.append({'bbox':[x1,y1,x2,y2],'label':label,
                         'conf':conf,'class_id':cls})
        return dets


# ─── Color Segmentation Detector (no weights needed) ─────────────────────────
class ColorSegmentDetector(BaseDetector):
    """
    Detects blobs of distinct hue on a known-background scene.
    Works perfectly on our synthetic test video.
    """
    name = "ColorSegment"

    # (label, BGR_mean, tolerance)
    TARGETS = [
        ('car',    np.array([220,100,  0]), 60),
        ('person', np.array([ 30,160, 30]), 60),
        ('truck',  np.array([ 60, 60,180]), 60),
        ('car2',   np.array([  0,150,200]), 60),
        ('person2',np.array([180, 50,180]), 60),
    ]
    DISPLAY = {
        'car':'car', 'car2':'car', 'person':'person', 'person2':'person',
        'truck':'truck',
    }

    def __init__(self, min_area=800, conf=0.85):
        self.min_area = min_area
        self.conf     = conf
        print("[ColorSegment] No model weights needed – using color segmentation")

    def detect(self, frame):
        dets = []
        for idx, (raw_label, target_bgr, tol) in enumerate(self.TARGETS):
            label = self.DISPLAY.get(raw_label, raw_label)
            diff = np.abs(frame.astype(np.float32) - target_bgr[None,None,:])
            mask = (diff.max(axis=2) < tol).astype(np.uint8) * 255
            # morphological cleanup
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                dets.append({'bbox':[x, y, x+w, y+h],
                             'label': label,
                             'conf':  self.conf,
                             'class_id': idx})
        return dets


# ─── Factory ──────────────────────────────────────────────────────────────────
def build_detector(yolo_path=None, mobilenet_proto=None, mobilenet_model=None,
                   conf=0.4):
    """Auto-select the best available detector."""
    if yolo_path and os.path.isfile(yolo_path) and os.path.getsize(yolo_path) > 1e6:
        try:
            return YOLOv8Detector(yolo_path, conf=conf)
        except Exception as e:
            print(f"[YOLOv8] Failed: {e}")

    if mobilenet_proto and mobilenet_model and \
       os.path.isfile(mobilenet_proto) and os.path.isfile(mobilenet_model):
        try:
            return MobileNetSSDDetector(mobilenet_proto, mobilenet_model, conf=conf)
        except Exception as e:
            print(f"[MobileNetSSD] Failed: {e}")

    return ColorSegmentDetector(conf=conf)