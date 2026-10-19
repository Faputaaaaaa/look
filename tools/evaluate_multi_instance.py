"""Evaluate offline class-aware multi-instance predictions, no torch required."""
import argparse
import json
from pathlib import Path
from quality.manifest import load_manifest
from quality.split import check_group_isolation
from quality.metrics import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--split-json", type=Path, help="optional; evaluate just one named partition")
    p.add_argument("--partition", default="val", choices=["train", "val", "test_unseen_sku"])
    p.add_argument("--score", type=float, default=0.5, help="operating confidence threshold; AP always uses all stored predictions")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=Path("multi_instance_metrics.json"))
    a = p.parse_args()
    manifest = load_manifest(a.manifest)
    preds = json.loads(a.predictions.read_text(encoding="utf-8"))
    rows = preds["predictions"] if isinstance(preds, dict) else preds
    if a.split_json:
        split = json.loads(a.split_json.read_text(encoding="utf-8"))
        check_group_isolation(manifest, split)
        selected = set(split[a.partition])
        if not selected:
            p.error(f"split partition {a.partition!r} is empty")
        manifest = dict(manifest, images=[r for r in manifest["images"] if r["image_id"] in selected])
        rows = [r for r in rows if r["image_id"] in selected]
        if len(manifest["images"]) != len(selected):
            p.error("split includes image_ids absent from manifest")
    summary = evaluate(manifest, rows, iou_threshold=a.iou, operating_score=a.score)
    by_sku = {}
    for sku in sorted({r["sku_id"] for r in manifest["images"]}):
        part = [r for r in manifest["images"] if r["sku_id"] == sku]
        ids = {r["image_id"] for r in part}
        by_sku[sku] = evaluate(dict(manifest, images=part), [r for r in rows if r["image_id"] in ids], iou_threshold=a.iou, operating_score=a.score)
    result = {"overall": summary, "per_sku": by_sku, "note": "AP uses saved detections (export at low score floor); P/R and bottle-level measures use operating threshold"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
