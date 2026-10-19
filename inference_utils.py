"""Small, testable inference utilities for the custom bottle-defect detector.

No assumptions about bottle type or number of defect instances.
"""
from torchvision.ops import batched_nms


def unique_sahi_windows(width: int, height: int, size: int, stride: int):
    """Return stable, unique, edge-covering sliding-window coordinates.

    This retains the old inference grid ordering and removes repeats caused by
    clamping windows to the image boundary. Small images are padded by callers.
    """
    if min(width, height, size, stride) <= 0:
        raise ValueError("image dimensions, window size, and stride must be positive")
    if stride > size:
        raise ValueError("stride must not exceed window size (would leave gaps)")

    out = []
    seen = set()
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            x2 = min(x + size, width)
            y2 = min(y + size, height)
            x1 = max(0, x2 - size)
            y1 = max(0, y2 - size)
            rect = (x1, y1, x2, y2)
            if rect not in seen:
                seen.add(rect)
                out.append(rect)
    return out


def class_aware_keep(boxes, scores, labels, iou_threshold: float, max_dets: int):
    """Keep overlapping detections of DIFFERENT classes; suppress same-class duplicates.

    This is class-aware NMS, not multi-label classification of one ROI.
    """
    if max_dets <= 0:
        raise ValueError("max_dets must be positive")
    if not (0 <= iou_threshold <= 1):
        raise ValueError("iou_threshold must be within [0, 1]")
    return batched_nms(boxes, scores, labels, iou_threshold)[:max_dets]
