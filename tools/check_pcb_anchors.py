"""PCB 锚框匹配率检查：为 PCB 线选定 img_size 与 anchor 集（用数据说话）。

对 train 划分全部 GT，计算在给定「输入模式 × anchor 集」下每个 GT 能达到的最大
IoU。位置无关：锚框密布特征图，最佳匹配等价于锚框与 GT 中心对齐，故用闭式解
IoU = min(wa,wg)*min(ha,hg) / (wa*ha + wg*hg - min(wa,wg)*min(ha,hg))。

输入模式：
- resize_<n>: 整图 letterbox 到 n×n（scale = min(n/iw, n/ih)，与
  dataset._process_resize 的公式一致）—— PCB 干净基线（无 SAHI）的输入口径。
- sahi_<n>:   n×n 原生窗口（窗口内即原图分辨率，无缩放）—— SAHI 消融列的口径。

候选 anchor 集（shape 与所在 FPN 层无关，匹配只看 尺度×比例 的并集）：
- xinpaide_cur: 瓶身线现役配置（32~512px，ratio 0.5~3）
- pcb_cand:     P2[16,24,32,48,64] P3[32,48,64,96,128] P4[64,96,128,192,256]，
                ratio [0.5,1,2] —— 按实测中位 66.7px 设计
- pcb_cand_plus: pcb_cand + ratio 3.0（检验细长框是否有收益）

输出：控制台表格 + data/pcb/anchor_match.json。
纯标准库，无 torch/numpy 依赖。运行：
    D:/Python3.14/python.exe -I tools/check_pcb_anchors.py --split-json data/pcb/pcb_split.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Windows GBK 控制台会把 UTF-8 中文打成乱码（项目 HANDOFF 有记录），强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


ANCHOR_SETS = {
    "xinpaide_cur": {
        "P2": ([32, 48, 64, 96, 128], [0.5, 1.0, 2.0, 3.0]),
        "P3": ([64, 96, 128, 192, 256], [0.5, 1.0, 2.0, 3.0]),
        "P4": ([128, 192, 256, 384, 512], [0.5, 1.0, 2.0, 3.0]),
    },
    "pcb_cand": {
        "P2": ([16, 24, 32, 48, 64], [0.5, 1.0, 2.0]),
        "P3": ([32, 48, 64, 96, 128], [0.5, 1.0, 2.0]),
        "P4": ([64, 96, 128, 192, 256], [0.5, 1.0, 2.0]),
    },
    "pcb_cand_plus": {
        "P2": ([16, 24, 32, 48, 64], [0.5, 1.0, 2.0, 3.0]),
        "P3": ([32, 48, 64, 96, 128], [0.5, 1.0, 2.0, 3.0]),
        "P4": ([64, 96, 128, 192, 256], [0.5, 1.0, 2.0, 3.0]),
    },
}

SIZE_BINS = [("tiny", 0.0, 0.0005), ("small", 0.0005, 0.001),
             ("medium", 0.001, float("inf"))]
POS_IOU = 0.3     # 项目 RPN pos_iou_thresh（FPN_CONFIG['rpn_config']）
STRICT_IOU = 0.5  # VOC/COCO 常见正样本口径


def anchor_shapes(anchor_set):
    """展平 anchor 集为 (w, h) 形状列表（层无关）。"""
    shapes = []
    for scales, ratios in anchor_set.values():
        for s in scales:
            for r in ratios:
                shapes.append((s * (r ** 0.5), s / (r ** 0.5)))
    return shapes


def best_iou(gt_w, gt_h, shapes):
    """GT 与全部 anchor shape 中心对齐的最大 IoU。"""
    best = 0.0
    for wa, ha in shapes:
        inter = min(wa, gt_w) * min(ha, gt_h)
        union = wa * ha + gt_w * gt_h - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best:
            best = iou
    return best


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def evaluate(mode, img_size, set_name, rows, classes):
    """对一种「模式 × anchor 集」组合统计全部 train GT 的匹配率。"""
    shapes = anchor_shapes(ANCHOR_SETS[set_name])
    ious = []
    per_cls = {i: [] for i in range(len(classes))}
    per_bin = {name: [] for name, _, _ in SIZE_BINS}
    gt_px_sizes = []
    for r in rows:
        iw, ih = r["width"], r["height"]
        scale = min(img_size / iw, img_size / ih) if mode == "resize" else 1.0
        for cls_id, cx, cy, w, h in r["boxes"]:
            gt_w = w * iw * scale
            gt_h = h * ih * scale
            gt_px_sizes.append((gt_w * gt_h) ** 0.5)
            frac = w * h  # 归一化面积占比（与 scale 无关）
            iou = best_iou(gt_w, gt_h, shapes)
            ious.append(iou)
            per_cls[cls_id].append(iou)
            for name, lo, hi in SIZE_BINS:
                if lo <= frac < hi:
                    per_bin[name].append(iou)
                    break
    srt = sorted(ious)
    return {
        "mode": mode,
        "img_size": img_size,
        "anchor_set": set_name,
        "n_gt": len(ious),
        "n_anchor_shapes": len(shapes),
        "cov_pos_iou_0.3": round(sum(1 for x in ious if x >= POS_IOU) / len(ious), 4),
        "cov_strict_iou_0.5": round(sum(1 for x in ious if x >= STRICT_IOU) / len(ious), 4),
        "best_iou_median": round(percentile(srt, 0.5), 4),
        "best_iou_mean": round(sum(ious) / len(ious), 4),
        "best_iou_p10": round(percentile(srt, 0.1), 4),
        "gt_sqrt_px_median": round(percentile(sorted(gt_px_sizes), 0.5), 1),
        "gt_sqrt_px_p10": round(percentile(sorted(gt_px_sizes), 0.1), 1),
        "gt_sqrt_px_p90": round(percentile(sorted(gt_px_sizes), 0.9), 1),
        "per_class": {
            classes[i]: {
                "n": len(per_cls[i]),
                "cov_0.3": round(sum(1 for x in per_cls[i] if x >= POS_IOU) / len(per_cls[i]), 4),
                "median_iou": round(percentile(sorted(per_cls[i]), 0.5), 4),
            }
            for i in range(len(classes))
        },
        "per_size_bin": {
            name: {
                "n": len(per_bin[name]),
                "cov_0.3": round(sum(1 for x in per_bin[name] if x >= POS_IOU) / len(per_bin[name]), 4)
                if per_bin[name] else None,
                "median_iou": round(percentile(sorted(per_bin[name]), 0.5), 4)
                if per_bin[name] else None,
            }
            for name, _, _ in SIZE_BINS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", default="data/pcb/pcb_split.json")
    parser.add_argument("--img-sizes", type=int, nargs="+", default=[640, 1024],
                        help="resize 模式候选输入尺寸")
    parser.add_argument("--sahi-size", type=int, default=640,
                        help="sahi 模式窗口尺寸（原生分辨率）")
    args = parser.parse_args()

    split_path = Path(args.split_json)
    if not split_path.exists():
        print(f"[FAIL] 找不到 {split_path}，先运行 tools/prepare_pcb.py")
        sys.exit(1)
    data = json.loads(split_path.read_text(encoding="utf-8"))
    classes = data["classes"]
    train_rows = data["train"]

    results = []
    for img_size in args.img_sizes:
        for set_name in ANCHOR_SETS:
            results.append(evaluate("resize", img_size, set_name, train_rows, classes))
    for set_name in ANCHOR_SETS:
        results.append(evaluate("sahi", args.sahi_size, set_name, train_rows, classes))

    # ---- 控制台表格 ----
    print("\n锚框匹配率（train 划分全部 GT，位置无关最大 IoU）")
    print(f"{'mode':<10} {'img':>4} {'set':<14} {'n_shp':>5} {'gt_px':>8} {'cov>=0.3':>9} "
          f"{'cov>=0.5':>9} {'IoU中位':>8} {'IoU均值':>8}")
    for r in results:
        print(f"{r['mode']:<10} {r['img_size']:>4} {r['anchor_set']:<14} "
              f"{r['n_anchor_shapes']:>5} {r['gt_sqrt_px_median']:>7.0f}px "
              f"{r['cov_pos_iou_0.3']:>9.3f} {r['cov_strict_iou_0.5']:>9.3f} "
              f"{r['best_iou_median']:>8.3f} {r['best_iou_mean']:>8.3f}")

    # 每类细节（只打印最关心的两档：resize_1024 与 sahi_640 的 pcb_cand）
    for key in [("resize", 1024, "pcb_cand"), ("sahi", args.sahi_size, "pcb_cand")]:
        r = next(x for x in results
                 if x["mode"] == key[0] and x["img_size"] == key[1]
                 and x["anchor_set"] == key[2])
        print(f"\n每类匹配（{r['mode']}_{r['img_size']} / {r['anchor_set']}，"
              f"GT像素中位={r['gt_sqrt_px_median']}px, p10={r['gt_sqrt_px_p10']}, "
              f"p90={r['gt_sqrt_px_p90']}）")
        for cls, d in r["per_class"].items():
            print(f"  {cls:<16} n={d['n']:>4}  cov>=0.3={d['cov_0.3']:.3f}  "
                  f"IoU中位={d['median_iou']:.3f}")
        print("  尺寸分档: " + "  ".join(
            f"{name}[n={d['n']},cov0.3={d['cov_0.3']}]"
            for name, d in r["per_size_bin"].items()))

    # ---- 落盘 ----
    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "split_json": str(split_path),
        "pos_iou_thresh": POS_IOU,
        "results": results,
    }
    out_path = split_path.parent / "anchor_match.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out_path} 已生成")


if __name__ == "__main__":
    main()
