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
import time
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
# 注意：sahi_inference 在 import 时捕获 window/img_size/类映射常量，
# 因此延迟到 main() 里、CLI 覆盖参数生效之后再 import。

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    return t.unsqueeze(0).to(DEVICE), scale, ox, oy


def invert_letterbox(box, scale, ox, oy, iw, ih):
    """letterbox 坐标 → 原图坐标，clip 到 [0, iw]×[0, ih]。"""
    x1 = max(0.0, min(float(iw), (box[0] - ox) / scale))
    y1 = max(0.0, min(float(ih), (box[1] - oy) / scale))
    x2 = max(0.0, min(float(iw), (box[2] - ox) / scale))
    y2 = max(0.0, min(float(ih), (box[3] - oy) / scale))
    return (x1, y1, x2, y2)


def _parse_out(out, thr):
    """解析 model.inference 的输出（batch 第 0 张）。

    与 sahi_inference._parse_inference_out 同逻辑，但该函数在 PR4 分支才存在
    （exp/pcb-benchmark 从 main 开出），工具自持一份，避免跨分支依赖。
    """
    boxes_list = out.get("boxes", [])
    scores_list = out.get("scores", [])
    labels_list = out.get("labels", [])

    if isinstance(boxes_list, list) and len(boxes_list) > 0:
        b_boxes = boxes_list[0]
        b_scores = scores_list[0] if isinstance(scores_list, list) else scores_list
        b_labels = labels_list[0] if isinstance(labels_list, list) else labels_list
    elif isinstance(boxes_list, torch.Tensor):
        b_boxes = boxes_list[0] if boxes_list.dim() == 3 else boxes_list
        b_scores = scores_list[0] if isinstance(scores_list, torch.Tensor) and scores_list.dim() >= 2 else scores_list
        b_labels = labels_list[0] if isinstance(labels_list, torch.Tensor) and labels_list.dim() >= 2 else labels_list
    else:
        return []

    results = []
    if isinstance(b_boxes, torch.Tensor) and b_boxes.numel() > 0:
        for i in range(b_boxes.size(0)):
            s = float(b_scores[i].item()) if isinstance(b_scores, torch.Tensor) else float(b_scores)
            if s < thr:
                continue
            c = int(b_labels[i].item()) if isinstance(b_labels, torch.Tensor) else int(b_labels)
            if c == 0:
                continue
            box = b_boxes[i].cpu().numpy()
            results.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), s, c))
    return results


@torch.no_grad()
def infer_resize(model, img: Image.Image, thr):
    """resize 模式单图推理 → [(x1,y1,x2,y2,score,cls_name)] 原图坐标。"""
    t, scale, ox, oy = letterbox(img)
    out = model.inference(t)
    dets = _parse_out(out, thr)
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


def bin_stats(manifest, pred_by_image, cls, lo, hi, op=0.5):
    """某个 (类别 × 面积档) 的统计：n_gt / recall@op / AP50。

    只匹配档内 GT（其余 GT 视为不存在）。注意：AP50 用共用预测池
    （COCO per-area 语义）——匹配档外 GT 的预测在这里记 FP，跨档竞争
    会压低各档 AP，因此 recall@op 是分档表的主读数，AP 是辅助列。
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
        return {"n_gt": 0, "recall_op": None, "ap": None}

    def _match(score_floor):
        matched = defaultdict(set)
        tps, fps = [], []
        for _conf, image_id, box in sorted(
                ((d["confidence"], image_id, d["bbox"])
                 for image_id, dets in pred_by_image.items()
                 for d in dets if d["class_name"] == cls and d["confidence"] >= score_floor),
                key=lambda x: -x[0]):
            options = [(_iou(box, g), j)
                       for j, g in enumerate(gt[image_id]) if j not in matched[image_id]]
            best = max(options, default=(0.0, -1))
            ok = best[1] >= 0 and best[0] >= 0.5
            if ok:
                matched[image_id].add(best[1])
            tps.append(int(ok))
            fps.append(int(not ok))
        return matched, tps, fps

    matched_op, tps_op, fps_op = _match(op)
    matched_raw, tps_raw, fps_raw = _match(0.0)
    recall_op = sum(len(v) for v in matched_op.values()) / n_pos
    ap = _ap_all_point(tps_raw, fps_raw, n_pos)
    return {"n_gt": n_pos, "recall_op": recall_op, "ap": ap}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="PCB checkpoint 路径")
    parser.add_argument("--split-json", default=config.pcb_split_json)
    parser.add_argument("--split", default="val", choices=["val"],
                        help="只允许 val（评估禁止用训练图）")
    parser.add_argument("--mode", default="resize", choices=["resize", "sahi"])
    parser.add_argument("--max-imgs", type=int, default=None, help="冒烟：只评估前 N 张")
    parser.add_argument("--out", default=None, help="评估结果 JSON 输出路径（默认 data/pcb/eval_<mode>.json）")
    # [PCB SAHI 消融] 窗口尺寸/步长覆盖：必须在 import sahi_inference 之前生效
    parser.add_argument("--img-size", type=int, default=None,
                        help="覆盖 config.img_size（SAHI 模式须 == window_size，不额外缩放）")
    parser.add_argument("--window-size", type=int, default=None,
                        help="覆盖 config.window_size（512/640/800）")
    parser.add_argument("--window-stride", type=int, default=None,
                        help="覆盖 config.window_stride（窗口的一半）")
    args = parser.parse_args()

    if args.img_size:
        config.img_size = args.img_size
    if args.window_size:
        config.window_size = args.window_size
    if args.window_stride:
        config.window_stride = args.window_stride
    # 覆盖生效后再 import（sahi 常量在 import 时捕获 config）
    import sahi_inference as si

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
    infer_times = []  # 每图推理耗时（秒）
    for i, row in enumerate(rows):
        if i % 20 == 0:
            print(f"  进度: {i}/{len(rows)}")
        img = Image.open(data_root / row["path"]).convert("RGB")
        t0 = time.perf_counter()
        if args.mode == "sahi":
            dets = infer_sahi(model, img)
        else:
            dets = infer_resize(model, img, thr=0.05)
        infer_times.append(time.perf_counter() - t0)
        predictions.append({
            "image_id": row["image_id"],
            "detections": [
                {"class_name": c, "confidence": round(float(s), 4),
                 "bbox": [round(float(v), 2) for v in box]}
                for box, s, c in [(d[:4], d[4], d[5]) for d in dets]
            ],
        })

    vram_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
               if torch.cuda.is_available() else None)
    print(f"[eval] 推理耗时: 总 {sum(infer_times):.1f}s / "
          f"每图均值 {sum(infer_times) / len(infer_times):.2f}s "
          f"(min {min(infer_times):.2f} / max {max(infer_times):.2f})")
    if vram_mb is not None:
        print(f"[eval] GPU 峰值显存: {vram_mb:.0f} MB")

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

    # ---- 相对面积分档（不硬套 COCO AP_small；主读数是 recall@op，AP 为辅助） ----
    pred_by_image = {p["image_id"]: p["detections"] for p in predictions}
    print(f"\n相对面积分档（操作点 conf>=0.5；格式 = n_gt / recall@op / AP50）")
    header = f"{'class':<16}" + "".join(f"{name:>26}" for name, _, _ in SIZE_BINS)
    print(header)
    for cls in split_data["classes"]:
        line = f"{cls:<16}"
        for name, lo, hi in SIZE_BINS:
            s = bin_stats(manifest, pred_by_image, cls, lo, hi)
            if s["ap"] is None:
                line += f"{'-':>26}"
            else:
                line += f"{s['n_gt']:>4} / {s['recall_op']:>5.2f} / {s['ap']:>6.3f}".rjust(26)
        print(line)

    # ---- 落盘 ----
    out_path = Path(args.out) if args.out else split_path.parent / f"eval_{args.mode}.json"
    payload = {
        "mode": args.mode,
        "model": args.model,
        "split": args.split,
        "n_images": len(rows),
        "img_size": config.img_size,
        "window_size": config.window_size if args.mode == "sahi" else None,
        "window_stride": config.window_stride if args.mode == "sahi" else None,
        "infer_time_s": {"total": round(sum(infer_times), 2),
                         "mean_per_image": round(sum(infer_times) / len(infer_times), 3)},
        "gpu_peak_vram_mb": round(vram_mb, 0) if vram_mb is not None else None,
        "eval": result,
        "predictions": predictions,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out_path} 已生成")


if __name__ == "__main__":
    main()
