"""PCB SAHI 窗口尺寸统计：512/640/800 的窗口数、背景比例、GT 截断率、训练样本量。

口径与训练/推理一致：窗口枚举用 inference_utils.unique_sahi_windows（推理同源），
步长 = 窗口一半；GT 归属按中心在窗口内 + clip 到窗口边界（dataset._get_bboxes_in_window
同规则，clip 后 <4px 丢弃）。

输出指标（train / val 分别统计）：
- windows_per_image: 每图窗口数 min / median / max；total = 全部窗口数
- pos_windows: 含 ≥1 个 GT 的窗口数；neg_windows: 纯背景窗口数 + 占比
- window_clip_rate: 窗口级截断率 = 被 clip 的 (GT×窗口) 出现次数 / 全部出现次数
- gt_fully_covered: 每个 GT 至少在一个窗口内完整可见（不被 clip）的比例
- samples_per_epoch: 训练样本/epoch = pos 窗口 + max_neg_windows_per_img × 图数
  （背景窗口限量统一策略；None=不限时按全部 neg 窗口计）

用法：
    D:/Python3.14/python.exe -m tools.pcb_window_stats --sizes 512 640 800
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

from inference_utils import unique_sahi_windows


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def window_stats(rows, size, stride, neg_limit):
    """对一个划分的全部图统计窗口指标。"""
    n_windows_per_img = []
    total_windows = 0
    total_pos = 0
    total_neg = 0
    occ_total = 0       # (GT × 窗口) 出现次数
    occ_clipped = 0     # 其中被 clip 的次数
    occ_dropped = 0     # clip 后 <4px 被丢弃的次数
    n_gt = 0
    n_gt_fully_covered = 0
    per_img_pos = []    # 每图含 GT 的窗口数（算背景限量用）

    for r in rows:
        iw, ih = r["width"], r["height"]
        rects = unique_sahi_windows(iw, ih, size, stride)
        n_windows_per_img.append(len(rects))
        total_windows += len(rects)

        n_pos_this = 0
        for (x1, y1, x2, y2) in rects:
            has_gt = False
            for cls_id, cx, cy, w, h in r["boxes"]:
                cx_px, cy_px = cx * iw, cy * ih
                if not (x1 <= cx_px <= x2 and y1 <= cy_px <= y2):
                    continue
                has_gt = True
                # clip 口径与 dataset._get_bboxes_in_window 一致
                bx1 = cx_px - w * iw / 2 - x1
                by1 = cy_px - h * ih / 2 - y1
                bx2 = cx_px + w * iw / 2 - x1
                by2 = cy_px + h * ih / 2 - y1
                occ_total += 1
                if bx1 < 0 or by1 < 0 or bx2 > (x2 - x1) or by2 > (y2 - y1):
                    occ_clipped += 1
                clip_w = min(bx2, x2 - x1) - max(bx1, 0.0)
                clip_h = min(by2, y2 - y1) - max(by1, 0.0)
                if clip_w < 4 or clip_h < 4:
                    occ_dropped += 1
            if has_gt:
                n_pos_this += 1
        per_img_pos.append(n_pos_this)
        total_pos += n_pos_this
        total_neg += len(rects) - n_pos_this

        # GT 级：每个 GT 至少在一个窗口内完整可见（不被 clip）
        for cls_id, cx, cy, w, h in r["boxes"]:
            n_gt += 1
            gx1 = (cx - w / 2) * iw
            gy1 = (cy - h / 2) * ih
            gx2 = (cx + w / 2) * iw
            gy2 = (cy + h / 2) * ih
            for (x1, y1, x2, y2) in rects:
                if x1 <= gx1 and y1 <= gy1 and gx2 <= x2 and gy2 <= y2:
                    n_gt_fully_covered += 1
                    break

    n_imgs = len(rows)
    if neg_limit is None or neg_limit < 0:
        samples_per_epoch = total_windows  # 不限：全部窗口
    else:
        neg_samples = sum(min(w - p, neg_limit)
                          for w, p in zip(n_windows_per_img, per_img_pos))
        samples_per_epoch = total_pos + neg_samples

    return {
        "n_images": n_imgs,
        "windows_per_image": {
            "min": min(n_windows_per_img), "max": max(n_windows_per_img),
            "median": percentile(sorted(n_windows_per_img), 0.5),
        },
        "total_windows": total_windows,
        "pos_windows": total_pos,
        "neg_windows": total_neg,
        "neg_ratio": round(total_neg / total_windows, 4) if total_windows else None,
        "window_clip_rate": round(occ_clipped / occ_total, 4) if occ_total else None,
        "clip_dropped_rate": round(occ_dropped / occ_total, 4) if occ_total else None,
        "gt_fully_covered_rate": round(n_gt_fully_covered / n_gt, 4) if n_gt else None,
        "n_gt": n_gt,
        "samples_per_epoch": samples_per_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", default="data/pcb/pcb_split.json")
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 640, 800])
    parser.add_argument("--neg-limit", type=int, default=2,
                        help="背景窗口每图限量（统一策略；-1=不限）")
    args = parser.parse_args()

    split_path = Path(args.split_json)
    data = json.loads(split_path.read_text(encoding="utf-8"))

    print(f"{'size/stride':<12} {'split':<6} {'win/img':>9} {'总窗口':>8} "
          f"{'pos':>6} {'neg':>6} {'neg%':>7} {'窗截断%':>8} {'GT完整%':>8} "
          f"{'clip丢%':>7} {'样本/ep':>9}")
    for size in args.sizes:
        stride = size // 2
        for name in ("train", "val"):
            s = window_stats(data[name], size, stride, args.neg_limit)
            print(f"{size}/{stride:<4} {'':>3} {name:<6} "
                  f"{s['windows_per_image']['median']:>5.0f}~"
                  f"{s['windows_per_image']['max']:<3} "
                  f"{s['total_windows']:>8} {s['pos_windows']:>6} "
                  f"{s['neg_windows']:>6} "
                  f"{s['neg_ratio'] * 100:>6.1f}% "
                  f"{s['window_clip_rate'] * 100:>7.1f}% "
                  f"{s['gt_fully_covered_rate'] * 100:>7.1f}% "
                  f"{s['clip_dropped_rate'] * 100:>6.2f}% "
                  f"{s['samples_per_epoch']:>9}")


if __name__ == "__main__":
    main()
