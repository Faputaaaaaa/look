"""Class-aware, multi-instance IoU matching and VOC all-point AP@0.5.

The operational threshold controls P/R/FP and image-level acceptance. AP uses
all stored predictions and should be exported using a low score floor.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .manifest import _bbox, validate_manifest


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def _ap_all_point(tps, fps, n_positive):
    if not n_positive:
        return None
    cum_t, cum_f = 0, 0
    rec, prec = [0.0], [0.0]
    for tp, fp in zip(tps, fps):
        cum_t += tp
        cum_f += fp
        rec.append(cum_t / n_positive)
        prec.append(cum_t / (cum_t + cum_f))
    rec.append(1.0)
    prec.append(0.0)
    for i in range(len(prec)-2, -1, -1):
        prec[i] = max(prec[i], prec[i+1])
    return sum((rec[i]-rec[i-1])*prec[i] for i in range(1, len(rec)) if rec[i] != rec[i-1])


def _match_class(images, pred_by_image, cls, iou_threshold, score_threshold):
    gt = {r["image_id"]: [d["bbox"] for d in r["defects"] if d["class_name"] == cls] for r in images}
    predictions = []
    for r in images:
        image_id = r["image_id"]
        for i, p in enumerate(pred_by_image[image_id]):
            if p["class_name"] == cls and p["confidence"] >= score_threshold:
                predictions.append((p["confidence"], image_id, i, p["bbox"]))
    predictions.sort(key=lambda x: (-x[0], x[1], x[2]))
    matched = defaultdict(set)
    tps, fps = [], []
    for score, image_id, _, box in predictions:
        options = [( _iou(box, g), j) for j, g in enumerate(gt[image_id]) if j not in matched[image_id]]
        best_iou, best_j = max(options, default=(0.0, -1))
        ok = best_j >= 0 and best_iou >= iou_threshold
        if ok:
            matched[image_id].add(best_j)
        tps.append(int(ok))
        fps.append(int(not ok))
    count_gt = sum(map(len, gt.values()))
    return {"tp": sum(tps), "fp": sum(fps), "fn": count_gt - sum(tps),
            "n_gt": count_gt, "tps": tps, "fps": fps,
            "matched": matched}


def evaluate(manifest, predictions, *, iou_threshold=0.5, operating_score=0.5):
    validate_manifest(manifest)
    if not 0 < iou_threshold <= 1 or not 0 <= operating_score <= 1:
        raise ValueError("invalid IoU threshold or operating score")
    images = manifest["images"]
    image_ids = {r["image_id"] for r in images}
    pred_by_image = {}
    rows = predictions.get("predictions") if isinstance(predictions, dict) else predictions
    if not isinstance(rows, list):
        raise ValueError("prediction JSON must contain predictions array")
    for row in rows:
        if not isinstance(row, dict) or row.get("image_id") not in image_ids or row["image_id"] in pred_by_image:
            raise ValueError(f"prediction references unknown/duplicate image: {row!r}")
        dets = row.get("detections")
        if not isinstance(dets, list):
            raise ValueError("detections must be an array (use [] for no detections)")
        for i, d in enumerate(dets):
            if not isinstance(d, dict) or d.get("class_name") not in manifest["classes"]:
                raise ValueError(f"unknown class in prediction for {row['image_id']}")
            _bbox(d.get("bbox"), f"{row['image_id']} detection {i}")
            s = d.get("confidence")
            if isinstance(s, bool) or not isinstance(s, (float, int)) or not math.isfinite(s) or not 0 <= s <= 1:
                raise ValueError("confidence must be finite in [0,1]")
        pred_by_image[row["image_id"]] = dets
    if set(pred_by_image) != image_ids:
        raise ValueError(f"missing predictions for {len(image_ids-set(pred_by_image))} images; refusing to silently treat them as empty")
    per_class = {}
    all_tp, all_fp, all_fn = 0, 0, 0
    matched_count = defaultdict(int)
    fp_count = defaultdict(int)
    ap_vals = []
    for cls in manifest["classes"]:
        raw = _match_class(images, pred_by_image, cls, iou_threshold, 0.0)
        op = _match_class(images, pred_by_image, cls, iou_threshold, operating_score)
        ap = _ap_all_point(raw["tps"], raw["fps"], raw["n_gt"])
        if ap is not None:
            ap_vals.append(ap)
        tp, fp, fn = op["tp"], op["fp"], op["fn"]
        all_tp += tp; all_fp += fp; all_fn += fn
        for image_id, indices in op["matched"].items():
            matched_count[image_id] += len(indices)
        # An image's false alarms are counted as all detections at operational
        # threshold minus class-specific matched true positives.
        for image_id in image_ids:
            n = sum(d["class_name"] == cls and d["confidence"] >= operating_score for d in pred_by_image[image_id])
            fp_count[image_id] += n - len(op["matched"].get(image_id, ()))
        per_class[cls] = {"AP50": ap, "tp": tp, "fp": fp, "fn": fn,
                          "precision": tp / (tp+fp) if tp+fp else 0.0,
                          "recall": tp / (tp+fn) if tp+fn else None,
                          "gt_instances": raw["n_gt"]}
    defective = [r for r in images if r["defects"]]
    normals = [r for r in images if not r["defects"]]
    multi_instance = [r for r in defective if len(r["defects"]) >= 2]
    multi_class = [r for r in defective if len({d["class_name"] for d in r["defects"]}) >= 2]

    def all_found(row):
        return matched_count[row["image_id"]] == len(row["defects"])

    def exact_match(row):
        return all_found(row) and fp_count[row["image_id"]] == 0

    complete = sum(all_found(r) for r in defective)
    exact = sum(exact_match(r) for r in defective)
    multi_instance_complete = sum(all_found(r) for r in multi_instance)
    multi_instance_exact = sum(exact_match(r) for r in multi_instance)
    multi_class_complete = sum(all_found(r) for r in multi_class)
    multi_class_exact = sum(exact_match(r) for r in multi_class)
    normal_alarm = sum(fp_count[r["image_id"]] > 0 for r in normals)
    return {
        "n_images": len(images), "n_defective": len(defective), "n_normal": len(normals),
        "n_multi_instance": len(multi_instance), "n_multi_class": len(multi_class),
        "iou_threshold": iou_threshold, "operating_score": operating_score,
        "mAP50": sum(ap_vals)/len(ap_vals) if ap_vals else None,
        "micro_tp": all_tp, "micro_fp": all_fp, "micro_fn": all_fn,
        "micro_precision": all_tp/(all_tp+all_fp) if all_tp+all_fp else 0.0,
        "micro_recall": all_tp/(all_tp+all_fn) if all_tp+all_fn else None,
        "all_defects_found_rate": complete/len(defective) if defective else None,
        "exact_defect_set_rate": exact/len(defective) if defective else None,
        "multi_instance_all_found_rate": multi_instance_complete/len(multi_instance) if multi_instance else None,
        "multi_instance_exact_set_rate": multi_instance_exact/len(multi_instance) if multi_instance else None,
        "multi_class_all_found_rate": multi_class_complete/len(multi_class) if multi_class else None,
        "multi_class_exact_set_rate": multi_class_exact/len(multi_class) if multi_class else None,
        "normal_false_alarm_rate": normal_alarm/len(normals) if normals else None,
        "per_class": per_class,
    }
