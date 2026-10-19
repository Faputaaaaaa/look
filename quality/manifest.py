"""Validated, explicit multi-defect ground-truth manifest contract.

Do not infer an unlabelled image is defect-free. An empty defects list is a
positive statement from a labeler, not a missing label file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _bbox(raw, context):
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"{context}: bbox must contain [x1, y1, x2, y2]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in raw):
        raise ValueError(f"{context}: bbox coordinates must be finite numbers")
    x1, y1, x2, y2 = raw
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"{context}: invalid bounding box {raw}")
    return tuple(float(v) for v in raw)


def validate_manifest(data):
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError("manifest must be an object with an images array")
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes or len(classes) != len(set(classes)) or not all(isinstance(x, str) and x.strip() for x in classes):
        raise ValueError("manifest.classes must be a unique nonempty list of class names")
    seen = set()
    for row in data["images"]:
        if not isinstance(row, dict):
            raise ValueError("images entries must be objects")
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip() or image_id in seen:
            raise ValueError(f"missing/duplicate image_id: {image_id!r}")
        seen.add(image_id)
        for field in ("sku_id", "capture_group", "path"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{image_id}: explicit {field} required")
        if not isinstance(row.get("defects"), list):
            raise ValueError(f"{image_id}: defects list required (use [] only for confirmed normal images)")
        for idx, inst in enumerate(row["defects"]):
            if not isinstance(inst, dict) or inst.get("class_name") not in classes:
                raise ValueError(f"{image_id}[{idx}]: unknown/missing class_name")
            _bbox(inst.get("bbox"), f"{image_id}[{idx}]")
    return data


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as f:
        return validate_manifest(json.load(f))
