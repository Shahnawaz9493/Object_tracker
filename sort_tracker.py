"""
SORT (Simple Online and Realtime Tracking)
Implementation from scratch using Kalman Filter + Hungarian Algorithm
Paper: https://arxiv.org/abs/1602.00763
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ─── Kalman Filter ───────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Tracks a single bounding box using a Kalman Filter.
    State vector: [x, y, w, h, vx, vy, vw, vh]
    Observation:  [x, y, w, h]
    """
    count = 0  # class-level ID counter

    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2]
        """
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.history = []

        # State: [cx, cy, w, h, vcx, vcy, vw, vh]
        self.x = self._bbox_to_state(bbox)

        # Transition matrix (constant velocity model)
        self.F = np.eye(8, 8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Observation matrix
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        # Covariance
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 1000.0  # high uncertainty for velocity

        # Process noise
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0

    @staticmethod
    def _bbox_to_state(bbox):
        """[x1,y1,x2,y2] → [cx,cy,w,h, 0,0,0,0]"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        return np.array([[cx], [cy], [w], [h], [0.], [0.], [0.], [0.]])

    @staticmethod
    def _state_to_bbox(state):
        """[cx,cy,w,h,...] → [x1,y1,x2,y2]"""
        cx, cy, w, h = state[0,0], state[1,0], state[2,0], state[3,0]
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    def predict(self):
        """Run Kalman predict step and return predicted bbox."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox(self.x)

    def update(self, bbox):
        """Run Kalman update step with measurement bbox."""
        z = np.array([[b] for b in [
            (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2,
            bbox[2]-bbox[0],     bbox[3]-bbox[1]
        ]])
        y  = z - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.history = []

    def get_state(self):
        return self._state_to_bbox(self.x)


# ─── IoU helpers ─────────────────────────────────────────────────────────────

def iou(bb_test, bb_gt):
    """Intersection-over-Union of two bboxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area_t = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area_g = (bb_gt[2]-bb_gt[0])   * (bb_gt[3]-bb_gt[1])
    union = area_t + area_g - inter
    return inter / union if union > 0 else 0.0


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Match detections to existing trackers using Hungarian algorithm.
    Returns:
        matches       – list of (det_idx, trk_idx)
        unmatched_det – set of detection indices with no match
        unmatched_trk – set of tracker  indices with no match
    """
    if len(trackers) == 0:
        return [], set(range(len(detections))), set()

    iou_matrix = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches, unmatched_det, unmatched_trk = [], set(), set()

    matched_det = set(row_ind)
    matched_trk = set(col_ind)

    for d in range(len(detections)):
        if d not in matched_det:
            unmatched_det.add(d)

    for t in range(len(trackers)):
        if t not in matched_trk:
            unmatched_trk.add(t)

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c))
        else:
            unmatched_det.add(r)
            unmatched_trk.add(c)

    return matches, unmatched_det, unmatched_trk


# ─── SORT Tracker ────────────────────────────────────────────────────────────

class SORT:
    """
    SORT tracker. Call update() on every frame with a list of
    [[x1, y1, x2, y2, score], ...] detections.
    Returns [[x1, y1, x2, y2, track_id], ...] for confirmed tracks.
    """

    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        """
        max_age      – frames a track survives without a detection
        min_hits     – frames before a new track is reported
        iou_threshold– minimum IoU to accept an assignment
        """
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0
        KalmanBoxTracker.count = 0   # reset IDs between runs

    def update(self, dets):
        """
        dets: np.ndarray of shape (N, 5) → [x1, y1, x2, y2, score]
              or empty array of shape (0, 5).
        Returns np.ndarray of shape (M, 6) → [x1, y1, x2, y2, track_id, class_id]
                (class_id is carried from dets[:,5] if present, else -1)
        """
        self.frame_count += 1
        has_class = dets.shape[1] >= 6 if len(dets) else False

        # Predict from all existing trackers
        trk_preds = []
        dead = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if any(np.isnan(pred)):
                dead.append(i)
            else:
                trk_preds.append(pred)
        for i in reversed(dead):
            self.trackers.pop(i)

        det_boxes = [d[:4] for d in dets] if len(dets) else []
        matches, unmatched_det, unmatched_trk = associate_detections_to_trackers(
            det_boxes, trk_preds, self.iou_threshold)

        # Update matched trackers
        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(dets[d_idx, :4])

        # Create new trackers for unmatched detections
        for d_idx in unmatched_det:
            self.trackers.append(KalmanBoxTracker(dets[d_idx, :4]))

        # Collect results and remove dead tracks
        results = []
        keep = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                keep.append(trk)
                if (trk.time_since_update < 1 and
                        (trk.hit_streak >= self.min_hits or
                         self.frame_count <= self.min_hits)):
                    bbox = trk.get_state()
                    results.append(bbox + [trk.id])
        self.trackers = keep

        return np.array(results) if results else np.empty((0, 5))