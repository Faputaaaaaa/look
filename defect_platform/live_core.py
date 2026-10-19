# -*- coding: utf-8 -*-
"""Live-video primitives. No Qt or model dependency; testable on CPU."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameJob:
    session: int
    sequence: int
    timestamp: float
    image: np.ndarray
    roi: tuple[int, int, int, int] | None
    threshold: float
    topk: int


class LatestFrameMailbox:
    """Bounded queue (capacity one): slow detector never builds a frame backlog."""
    def __init__(self):
        self._condition = Condition()
        self._item = None
        self._closed = False

    def offer(self, item):
        with self._condition:
            if self._closed:
                return False
            self._item = item
            self._condition.notify()
            return True

    def take(self, timeout=0.2):
        with self._condition:
            if self._item is None and not self._closed:
                self._condition.wait(timeout)
            item, self._item = self._item, None
            return item

    def close(self):
        with self._condition:
            self._closed = True
            self._item = None
            self._condition.notify_all()


def bounded_roi(shape, roi, minimum=32):
    """Crop in original camera pixels. A too-small or invalid ROI is rejected."""
    h, w = shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("invalid frame dimensions")
    if roi is None:
        return (0, 0, w, h)
    if len(roi) != 4 or any(not np.isfinite(v) for v in roi):
        raise ValueError("ROI must be four finite coordinates")
    x1, y1, x2, y2 = [int(round(float(v))) for v in roi]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < minimum or y2 - y1 < minimum:
        raise ValueError(f"ROI must be at least {minimum}×{minimum} pixels")
    return (x1, y1, x2, y2)


def global_detections(detections, roi, frame_shape):
    """Translate ROI-local model boxes to frame coords and clip boundaries."""
    x0, y0, _, _ = roi
    h, w = frame_shape[:2]
    converted = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        coords = (max(0, min(w, x1 + x0)), max(0, min(h, y1 + y0)),
                  max(0, min(w, x2 + x0)), max(0, min(h, y2 + y0)))
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        converted.append({**d, "bbox": tuple(float(v) for v in coords)})
    return converted


def to_gray(frame, max_width=640):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w > max_width:
        nw = max_width
        nh = max(1, round(h * nw / w))
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    return gray


class MotionBoxes:
    """Short-lived optical-flow tracking, never treated as a fresh AI detection.

    Frames stored only as 8-bit low-resolution grayscale to bound memory.
    A detection is discarded when too old for credible tracking.
    """
    def __init__(self, max_age=2.5, history_size=90):
        self.max_age = max_age
        self.history = deque(maxlen=history_size)  # (seq, timestamp, gray)
        self.boxes = []
        self.points = []
        self.prev = None
        self.last_detection_at = -1.0
        self.orig_size = None
        self._scales = (1.0, 1.0)
        self._tracking = False

    def clear(self):
        self.history.clear()
        self.boxes = []
        self.points = []
        self.prev = None
        self.orig_size = None
        self._tracking = False
        self.last_detection_at = -1.0

    def _init_boxes(self, gray, detections):
        self.boxes = [dict(d) for d in detections]
        self.points = []
        sx, sy = self._scales
        hh, ww = gray.shape
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            a, b = max(0,int(x1/sx)), max(0,int(y1/sy))
            c, e = min(ww,int(np.ceil(x2/sx))), min(hh,int(np.ceil(y2/sy)))
            pts = None
            if c-a >= 8 and e-b >= 8:
                pts = cv2.goodFeaturesToTrack(
                    gray[b:e, a:c], maxCorners=30, qualityLevel=0.01, minDistance=3)
                if pts is not None:
                    pts = pts.reshape(-1, 2)
                    pts[:, 0] += a
                    pts[:, 1] += b
                    pts = pts.reshape(-1, 1, 2).astype(np.float32)
            self.points.append(pts)
        self.prev = gray
        self._tracking = bool(self.boxes)

    def _advance(self, gray):
        if not self._tracking or self.prev is None:
            return
        sx, sy = self._scales
        for idx, pts in enumerate(self.points):
            if pts is None or len(pts) < 3:
                # No reliable feature points: fail closed rather than hold stale box.
                self.points[idx] = None
                continue
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev, gray, pts, None, winSize=(21,21), maxLevel=2)
            if nxt is None or status is None:
                self.points[idx] = None
                continue
            good = status.reshape(-1).astype(bool)
            old = pts.reshape(-1,2)[good]
            new = nxt.reshape(-1,2)[good]
            if len(new) < 3:
                self.points[idx] = None
                continue
            shift = np.median(new-old, axis=0)
            residual = np.median(np.linalg.norm((new-old)-shift, axis=1))
            if not np.all(np.isfinite(shift)) or residual > 7 or np.linalg.norm(shift) > 60:
                self.points[idx] = None
                continue
            dx,dy = float(shift[0]*sx),float(shift[1]*sy)
            x1,y1,x2,y2 = self.boxes[idx]["bbox"]
            self.boxes[idx]["bbox"] = (x1+dx,y1+dy,x2+dx,y2+dy)
            self.points[idx] = new.reshape(-1,1,2).astype(np.float32)
        self.prev = gray
        # Hide boxes that have lost visual track (instead of falsely freezing them).
        self._tracking = any(p is not None for p in self.points)
        if not self._tracking:
            self.boxes = []

    def ingest(self, sequence, timestamp, frame):
        h, w = frame.shape[:2]
        if self.orig_size != (w,h):
            self.clear()
            self.orig_size = (w,h)
        gray = to_gray(frame)
        self._scales = (w/gray.shape[1], h/gray.shape[0])
        self.history.append((sequence,timestamp,gray))
        if self._tracking:
            self._advance(gray)
        if timestamp - self.last_detection_at > self.max_age:
            self.boxes = []
            self._tracking = False

    def install(self, sequence, timestamp, detections):
        """Replay at most 12 sampled frames from detection image to latest.

        Returns False for stale/missing images. Boxes remain provisional tracking,
        NOT fresh re-detections. The source image must exist in history.
        """
        if not self.history:
            return False
        now = self.history[-1][1]
        if now - timestamp > self.max_age or now < timestamp:
            return False
        index = next((i for i, (seq,_,_) in enumerate(self.history) if seq==sequence), None)
        if index is None:
            return False
        tail = list(self.history)[index:]
        if not tail:
            return False
        self.last_detection_at = timestamp
        self._init_boxes(tail[0][2], detections)
        if self._tracking and len(tail)>1:
            # Bounded catch-up work avoids blocking Qt main thread for 90 LK steps.
            indices = np.unique(np.linspace(1,len(tail)-1,min(12,len(tail)-1)).astype(int))
            for idx in indices:
                self._advance(tail[idx][2])
        return True

    def visible(self, now=None):
        now = time.monotonic() if now is None else now
        if now - self.last_detection_at > self.max_age:
            return []
        return [dict(d) for d,p in zip(self.boxes,self.points) if p is not None]
