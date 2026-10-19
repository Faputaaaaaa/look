"""Export raw (low score-floor) SAHI detections for each selected manifest image.

Uses the project's existing sahi_inference_image implementation: no competing
inference pipeline. Requires torch, real checkpoint, and actual images.
"""
import argparse
import json
from pathlib import Path

from quality.manifest import load_manifest
from quality.split import check_group_isolation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--split-json", required=True, type=Path)
    p.add_argument("--partition", default="val", choices=["train", "val", "test_unseen_sku"])
    p.add_argument("--data-root", default=".", type=Path)
    p.add_argument("--model", type=Path, help="model .pth; defaults to project config")
    p.add_argument("--score-floor", type=float, default=0.05)
    p.add_argument("--max-dets", type=int, default=30)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if not 0 <= a.score_floor <= 1 or a.max_dets < 1:
        p.error("invalid score floor or max detections")
    manifest = load_manifest(a.manifest)
    split = json.loads(a.split_json.read_text(encoding="utf-8"))
    check_group_isolation(manifest, split)
    selected = set(split[a.partition])
    if not selected:
        p.error(f"split {a.partition!r} has no images")
    # Import heavy packages only when the actual export is executed.
    from PIL import Image
    import sahi_inference as si
    from config import config
    model = si.load_model(str(a.model) if a.model else None)
    rows = []
    for entry in manifest["images"]:
        if entry["image_id"] not in selected:
            continue
        img = Path(entry["path"])
        if not img.is_absolute():
            img = a.data_root / img
        if not img.is_file():
            raise FileNotFoundError(f"image absent: {img}")
        with Image.open(img) as im:
            predictions = si.sahi_inference_image(model, im.convert("RGB"), score_thresh=a.score_floor, max_dets=a.max_dets)
        rows.append({"image_id": entry["image_id"], "detections": [
            {"bbox": [float(x1), float(y1), float(x2), float(y2)],
             "confidence": float(conf), "class_name": config.class_names[int(cls)-1]}
            for x1, y1, x2, y2, conf, cls in predictions]})
        print(f"{entry['image_id']}: {len(rows[-1]['detections'])} detections")
    if {r["image_id"] for r in rows} != selected:
        raise ValueError("split contains IDs not in manifest")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"predictions": rows, "score_floor": a.score_floor, "max_dets": a.max_dets,
                                 "model": str(a.model) if a.model else "project_default", "partition": a.partition}, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
