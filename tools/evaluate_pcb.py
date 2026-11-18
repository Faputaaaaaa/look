"""PCB 基准评估：统一走项目推理口径 → quality/metrics.py 类别感知评估。

两种模式（对应 PCB_PLAN.md 的消融矩阵）：
- resize（Phase 1 基线）：整图 letterbox config.img_size → model.inference →
  反 letterbox 回原图坐标。letterbox 公式与 dataset._process_resize 完全一致
  （scale=min(S/iw,S/ih)，灰色 128 画布，BILINEAR）。
- sahi（Phase 2 消融）：走 sahi_inference.sahi_inference_image 单一口径
  （窗口枚举/全局 NMS/top-k 与平台一致）。

评估口径：quality/metrics.py —— 类别感知多实例匹配 + VOC all-point AP50
（锚定 recall=0）+ 每类 AP/P/R + micro P/R；另有相对面积分档
[<0.05%, 0.05~0.10%, >0.10%] 的每类 AP（PCB 不硬套 COCO AP_small）。
操作阈值 P/R 用 qm.evaluate 的 operating_score 固定口径。

注意：
1. import 顺序敏感：必须先 config.set_profile('pcb')，再 import model_fpn/
   sahi_inference（后者在 import 时捕获 window/img_size/类映射常量）。
2. 模型内部 inference 有 score>0.1 的硬过滤（model_fpn.py），故 AP 曲线的
   有效 score floor = 0.1（PR2 建议的 0.05 floor 被此上限约束，已记录）。
3. 只评估 val 划分（板 10/11/12）——PCB_PLAN 铁律：评估禁止用训练图。

用法：
    D:/Python3.14/python.exe tools/evaluate_pcb.py --model <ckpt> [--mode resize|sahi]
"""

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---- import 顺序敏感：先切 profile，再 import 依赖 config 常量的模块 ----
from config import config
config.set_profile("pcb")

import torch
from PIL import Image
import torchvision.transforms as transforms

from quality.metrics import evaluate as qm_evaluate
from quality.metrics import _ap_all_point, _iou  # 分档 AP 复用同一插值实现
import sahi_inference as si


NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

SIZE_BINS = [("tiny(<0.05%)", 0.0, 0.0005),
             ("small(0.05-0.1%)", 0.0005, 0.001),
             ("medium(>0.1%)", 0.001, float("inf"))]


def letterbox(img: Image.Image):
    """整图 letterbox（与 dataset._process_resize 同公式），返回 (tensor, scale, ox, oy)。"""
    iw, ih = img.size
    size = config.img_size
    scale = min(size / iw, size / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (ox, oy))
    t = NORMALIZE(transforms.ToTensor()(canvas))
    return t.unsqueeze(0).to(si.DEVICE), scale, ox, oy


def invert_letterbox(box, scale, ox, oy, iw, ih):
    """letterbox 坐标 → 原图坐标，clip 到 [0, iw]×[0, ih]。"""
    x1 = max(0.0, min(float(iw), (box[0] - ox) / scale))
    y1 = max(0.0, min(float(ih), (box[1] - oy) / scale))
    x2 = max(0.0, min(float(iw), (box[2] - ox) / scale))
    y2 = max(0.0, min(float(ih), (box[3] - oy) / scale))
    return (x1, y1, x2, y2)


@torch.no_grad()
def infer_resize(model, img: Image.Image, thr):
    """resize 模式单图推理 → [(x1,y1,x2,y2,score,cls_name)] 原图坐标。"""
    t, scale, ox, oy = letterbox(img)
    out = model.inference(t)
    dets = si._parse_inference_out(out, 0, thr)
    iw, ih = img.size
    result = []
    for x1, y1, x2, y2, s, c in dets:
        box = invert_letterbox((x1, y1, x2, y2), scale, ox, oy, iw, ih)
        if box[2] - box[0] <= 1 or box[3] - box[1] <= 1:
            continue
        result.append((box[0], box[1], box[2], box[3], s, config.class_names[c - 1]))
    return result


def infer_sahi(model, img: Image.Image):
    """SAHI 模式单图推理 → [(x1,y1,x2,y2,score,cls_name)] 原图坐标（平台同口径）。"""
    dets = si.sahi_inference_image(model, img, max_dets=config.max_dets_per_image)
    return [(x1, y1, x2, y2, s, config.class_names[c - 1]) for x1, y1, x2, y2, s, c in dets]


def build_manifest(split_data, split_name):
    """由 pcb_split.json 构建 quality 契约 manifest（capture_group=板 ID）。"""
    classes = split_data["classes"]
    images = []
    for r in split_data[split_name]:
        defects = []
        for cls_id, cx, cy, w, h in r["boxes"]:
            x1 = (cx - w / 2) * r["width"]
            y1 = (cy - h / 2) * r["height"]
            x2 = (cx + w / 2) * r["width"]
            y2 = (cy + h / 2) * r["height"]
            defects.append({
                "class_name": classes[cls_id],
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            })
        images.append({
            "image_id": r["file"],
            "sku_id": r["board"],
            "capture_group": r["board"],
            "path": f"images/{r['file']}",
            "width": r["width"],
            "height": r["height"],
            "defects": defects,
        })
    return {"classes": classes, "images": images}


def bin_ap(manifest, pred_by_image, cls, lo, hi):
    """某个 (类别 × 面积档) 的 AP50：只匹配档内 GT（其余 GT 视为不存在）。

    复用 quality/metrics.py 的插值实现；bbox 面积占比用 manifest 里 image 宽高算。
    """
    from collections import defaultdict

    gt = {}
    n_pos = 0
    for row in manifest["images"]:
        img_area = row["width"] * row["height"]
        boxes = []
        for d in row["defects"]:
            if d["class_name"] != cls:
                continue
            w = d["bbox"][2] - d["bbox"][0]
            h = d["bbox"][3] - d["bbox"][1]
            frac = w * h / img_area
            if lo <= frac < hi:
                boxes.append(d["bbox"])
        gt[row["image_id"]] = boxes
        n_pos += len(boxes)
    if n_pos == 0:
        return None
    all_preds = sorted(
        (d["confidence"], image_id, d["bbox"])
        for image_id, dets in pred_by_image.items()
        for d in dets if d["class_name"] == cls)
    matched = defaultdict(set)
    tps, fps = [], []
    for _conf, image_id, box in all_preds:
        options = [(_iou(box, g), j)
                   for j, g in enumerate(gt[image_id]) if j not in matched[image_id]]
        best = max(options, default=(0.0, -1))
        ok = best[1] >= 0 and best[0] >= 0.5
        if ok:
            matched[image_id].add(best[1])
        tps.append(int(ok))
        fps.append(int(not ok))
    return _ap_all_point(tps, fps, n_pos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="PCB checkpoint 路径")
    parser.add_argument("--split-json", default=config.pcb_split_json)
    parser.add_argument("--split", default="val", choices=["val"],
                        help="只允许 val（评估禁止用训练图）")
    parser.add_argument("--mode", default="resize", choices=["resize", "sahi"])
    parser.add_argument("--max-imgs", type=int, default=None, help="冒烟：只评估前 N 张")
    parser.add_argument("--out", default=None, help="评估结果 JSON 输出路径（默认 data/pcb/eval_<mode>.json）")
    args = parser.parse_args()

    split_path = Path(args.split_json)
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    manifest = build_manifest(split_data, args.split)
    rows = manifest["images"]
    if args.max_imgs:
        rows = rows[:args.max_imgs]
        manifest = {"classes": manifest["classes"], "images": rows}

    model = si.load_model(args.model)
    print(f"[eval] mode={args.mode} img_size={config.img_size} "
          f"images={len(rows)} model={args.model}")

    data_root = Path(config.data_dir)
    predictions = []
    for i, row in enumerate(rows):
        if i % 20 == 0:
            print(f"  进度: {i}/{len(rows)}")
        img = Image.open(data_root / row["path"]).convert("RGB")
        if args.mode == "sahi":
            dets = infer_sahi(model, img)
        else:
            dets = infer_resize(model, img, thr=0.05)
        predictions.append({
            "image_id": row["image_id"],
            "detections": [
                {"class_name": c, "confidence": round(float(s), 4),
                 "bbox": [round(float(v), 2) for v in box]}
                for box, s, c in [(d[:4], d[4], d[5]) for d in dets]
            ],
        })

    result = qm_evaluate(manifest, predictions, iou_threshold=0.5, operating_score=0.5)

    # ---- 控制台表格 ----
    print(f"\n{'='*72}")
    print(f"PCB 评估结果（{args.mode} / {len(rows)} 张 / IoU@0.5）")
    print(f"{'='*72}")
    print(f"{'class':<16} {'AP50':>7} {'P@0.5':>7} {'R@0.5':>7} {'GT':>5} {'TP':>5} {'FP':>5} {'FN':>5}")
    for cls, d in result["per_class"].items():
        ap = d["AP50"]
        ap_s = f"{ap:.4f}" if ap is not None else "  -"
        rec = d["recall"]
        rec_s = f"{rec:.4f}" if rec is not None else "  -"
        print(f"{cls:<16} {ap_s:>7} {d['precision']:>7.4f} {rec_s:>7} "
              f"{d['gt_instances']:>5} {d['tp']:>5} {d['fp']:>5} {d['fn']:>5}")
    mAP = result["mAP50"]
    print(f"{'-'*72}")
    print(f"{'mAP50':<16} {mAP:>7.4f}    micro P={result['micro_precision']:.4f} "
          f"R={result['micro_recall']:.4f}")
    print(f"{'='*72}")

    # ---- 相对面积分档（不硬套 COCO AP_small） ----
    pred_by_image = {p["image_id"]: p["detections"] for p in predictions}
    print(f"\n相对面积分档 AP50（按缺陷面积占图比例）")
    header = f"{'class':<16}" + "".join(f"{name:>18}" for name, _, _ in SIZE_BINS)
    print(header)
    for cls in split_data["classes"]:
        line = f"{cls:<16}"
        for name, lo, hi in SIZE_BINS:
            ap = bin_ap(manifest, pred_by_image, cls, lo, hi)
            line += f"{ap:>18.4f}" if ap is not None else f"{'-':>18}"
        print(line)

    # ---- 落盘 ----
    out_path = Path(args.out) if args.out else split_path.parent / f"eval_{args.mode}.json"
    payload = {
        "mode": args.mode,
        "model": args.model,
        "split": args.split,
        "n_images": len(rows),
        "img_size": config.img_size,
        "eval": result,
        "predictions": predictions,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out_path} 已生成")


if __name__ == "__main__":
    main()
