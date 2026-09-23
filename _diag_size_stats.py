"""统计 xinpaide 数据集 dirty/missing 的实际尺寸分布。

输出：
1. 每类样本数、w/h 像素分布（min/median/max/mean）
2. 占原图面积百分比分布
3. 在不同 window_size 下的相对占比（评估上下文是否充足）
4. 推荐切片尺寸
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# 直接读取，避免依赖 config
DATA_DIR = Path(r"d:\deeplearning\learn\xinpaide")
ORIG_W, ORIG_H = 4096, 2736

# YOLO id → 类名
YOLO_ID_TO_NAME = {
    0: 'ghosting', 1: 'faint', 2: 'fuzziness',
    3: 'missing', 4: 'dirty', 5: 'scratch', 6: 'body_scratch'
}
TARGET_CLASSES = ['missing', 'dirty']


def collect_stats():
    stats = defaultdict(list)  # class_name → list of (w_px, h_px, area_ratio)
    raw_records = defaultdict(list)  # class_name → list of (file, cx, cy, w_norm, h_norm)

    for cls in TARGET_CLASSES:
        label_dir = DATA_DIR / cls / "labels"
        if not label_dir.exists():
            continue
        for txt in sorted(label_dir.glob("frame_*.txt")):
            with open(txt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    yolo_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    name = YOLO_ID_TO_NAME.get(yolo_id)
                    if name not in TARGET_CLASSES:
                        continue
                    w_px = w * ORIG_W
                    h_px = h * ORIG_H
                    area_ratio = w_px * h_px / (ORIG_W * ORIG_H)
                    stats[name].append((w_px, h_px, area_ratio))
                    raw_records[name].append((txt.stem, cx, cy, w, h))
    return stats, raw_records


def print_stats(stats):
    print("=" * 80)
    print(f"原图尺寸: {ORIG_W} × {ORIG_H}  (总面积 = {ORIG_W*ORIG_H:,} px²)")
    print("=" * 80)
    summary = {}
    for cls in TARGET_CLASSES:
        arr = np.array(stats[cls])  # [N, 3]
        if len(arr) == 0:
            print(f"\n[{cls}] 无样本")
            continue
        w_px = arr[:, 0]
        h_px = arr[:, 1]
        area = arr[:, 2]

        print(f"\n[{cls}]  样本数: {len(arr)}")
        print(f"  宽度 w_px:  min={w_px.min():.0f}  median={np.median(w_px):.0f}  mean={w_px.mean():.0f}  max={w_px.max():.0f}")
        print(f"  高度 h_px:  min={h_px.min():.0f}  median={np.median(h_px):.0f}  mean={h_px.mean():.0f}  max={h_px.max():.0f}")
        print(f"  面积占比%:  min={area.min()*100:.3f}  median={np.median(area)*100:.3f}  mean={area.mean()*100:.3f}  max={area.max()*100:.3f}")
        print(f"  长宽比 w/h: min={(w_px/h_px).min():.2f}  median={np.median(w_px/h_px):.2f}  max={(w_px/h_px).max():.2f}")

        # 分位数
        for q in [10, 25, 50, 75, 90, 95]:
            print(f"  P{q:>2}: w={np.percentile(w_px,q):.0f}  h={np.percentile(h_px,q):.0f}  area%={np.percentile(area,q)*100:.3f}")

        summary[cls] = {
            'w_median': float(np.median(w_px)),
            'h_median': float(np.median(h_px)),
            'w_p90': float(np.percentile(w_px, 90)),
            'h_p90': float(np.percentile(h_px, 90)),
            'w_max': float(w_px.max()),
            'h_max': float(h_px.max()),
            'n': len(arr),
        }

    return summary


def evaluate_window_sizes(summary):
    """评估不同 window_size 下缺陷占切片的比例。

    经验法则：缺陷占切片 5%~25% 时上下文充足（context-rich）；
    >40% 时上下文不足（context-poor），ROI Align 学不到边界对比。
    """
    print("\n" + "=" * 80)
    print("切片尺寸适配性评估（缺陷占切片宽度比 = w_px / window_size）")
    print("目标：缺陷占切片 5%~25% 为最佳上下文区间")
    print("=" * 80)

    for ws in [320, 448, 512, 640, 768, 1024]:
        print(f"\n--- window_size = {ws} ---")
        for cls in TARGET_CLASSES:
            if cls not in summary:
                continue
            s = summary[cls]
            ratio_median = s['w_median'] / ws * 100
            ratio_p90 = s['w_p90'] / ws * 100
            ratio_max = s['w_max'] / ws * 100
            verdict = "✓ 上下文充足" if ratio_p90 <= 30 else ("△ 偏大" if ratio_p90 <= 50 else "✗ 上下文不足")
            print(f"  [{cls}] median={ratio_median:5.1f}%  P90={ratio_p90:5.1f}%  max={ratio_max:5.1f}%  → {verdict}")


def recommend(summary):
    """基于 P90 给出推荐切片尺寸。"""
    print("\n" + "=" * 80)
    print("推荐策略")
    print("=" * 80)

    # 经验：缺陷 P90 占切片 15%-20% 为甜点区
    # window_size ≈ w_p90 / 0.18
    for cls in TARGET_CLASSES:
        if cls not in summary:
            continue
        s = summary[cls]
        # 取 w 和 h 较大者做基准，保证两个维度都装得下
        max_dim_p90 = max(s['w_p90'], s['h_p90'])
        max_dim_max = max(s['w_max'], s['h_max'])
        recommended = int(np.ceil(max_dim_p90 / 0.20 / 32) * 32)  # 对齐到32
        recommended = max(256, recommended)  # 下限256
        print(f"\n  [{cls}]")
        print(f"    P90 最大维度: {max_dim_p90:.0f}px  极值最大维度: {max_dim_max:.0f}px")
        print(f"    推荐单类切片: {recommended}px  (该尺寸下P90缺陷占比≈{max_dim_p90/recommended*100:.1f}%)")

    # 综合两类
    if all(cls in summary for cls in TARGET_CLASSES):
        all_max_p90 = max(summary[c]['w_p90'] for c in TARGET_CLASSES)
        # 统一尺寸方案
        unified = int(np.ceil(all_max_p90 / 0.20 / 32) * 32)
        unified = max(256, unified)
        print(f"\n  [统一切片方案]")
        print(f"    综合P90基准: {all_max_p90:.0f}px")
        print(f"    推荐 window_size = {unified}px")

        # 评估此方案下两类缺陷的实际占比
        for cls in TARGET_CLASSES:
            s = summary[cls]
            ratio_p90 = max(s['w_p90'], s['h_p90']) / unified * 100
            ratio_max = max(s['w_max'], s['h_max']) / unified * 100
            print(f"    [{cls}] P90占比={ratio_p90:.1f}%  最大占比={ratio_max:.1f}%")


if __name__ == "__main__":
    stats, raw = collect_stats()
    summary = print_stats(stats)
    evaluate_window_sizes(summary)
    recommend(summary)
