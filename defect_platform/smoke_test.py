# -*- coding: utf-8 -*-
"""[平台] 无界面冒烟测试：跑几张图，直接产出"能看的效果图"。

用途：
1. 不装 / 不想开 PyQt 时，先确认整条链路（config → checkpoint → SAHI → 画框）是通的；
2. 每次改了模型或后处理参数，快速肉眼核对框准不准。

产物（默认 defect_platform/outputs/smoke_<时间戳>/）：
- <name>_full.jpg  全图：红/黄框=预测，绿框=GT
- <name>_crop.jpg  缺陷区域放大裁剪
- summary.txt      每张图的类别/置信度/框/IoU/耗时

用法：
    python defect_platform/smoke_test.py                    # val 集 missing/dirty 各 1 张
    python defect_platform/smoke_test.py --n 3              # 各 3 张
    python defect_platform/smoke_test.py --dir D:/my_photos # 对某个文件夹跑
    python defect_platform/smoke_test.py --topk 3           # 一图保留 3 个框
"""
import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from defect_platform.detector import FasterRCNNDetector  # noqa: E402
from defect_platform.viz import (  # noqa: E402
    IMG_EXTS, OUT_DIR, imread_u, imwrite_u, load_gts, best_gt_iou, draw_vis, crop_with_dets,
)


def collect_from_split(split, n_per_class):
    import sahi_inference as si
    sp = si.load_split()
    ids = []
    for cls in ('missing', 'dirty'):
        ids += [i for i in sp[split] if i.startswith(f'{cls}/')][:n_per_class]
    out = []
    for img_id in ids:
        img_file, _, _ = si.resolve_image(img_id)
        out.append((img_id, Path(img_file)))
    return out


def collect_from_dir(d, limit=0):
    files = sorted([p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS])
    if limit:
        files = files[:limit]
    return [(p.name, p) for p in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None, help='checkpoint（默认 config.default_ckpt）')
    ap.add_argument('--split', default='val', choices=['val', 'train'])
    ap.add_argument('--n', type=int, default=1, help='每个类别取几张（split 模式）')
    ap.add_argument('--dir', default=None, help='直接对某文件夹跑（优先于 split）')
    ap.add_argument('--limit', type=int, default=0, help='--dir 模式下最多几张')
    ap.add_argument('--topk', type=int, default=None)
    ap.add_argument('--thresh', type=float, default=None)
    ap.add_argument('--no-gt', action='store_true', help='不画 GT 绿框')
    ap.add_argument('--expand', action='store_true',
                    help=f'启用 A11 检测框扩张（missing 1.3x / dirty 1.1x，默认关）')
    a = ap.parse_args()

    if a.expand:
        from config import config
        config.use_bbox_expansion = True
        print('已启用 A11 扩张:', config.expand_missing, '/', config.expand_dirty)

    items = collect_from_dir(a.dir, a.limit) if a.dir else collect_from_split(a.split, a.n)
    if not items:
        print('没有可跑的图片')
        return 1

    t0 = time.time()
    det = FasterRCNNDetector(a.model, score_thresh=a.thresh, max_dets=a.topk)
    print(f'checkpoint : {det.model_path}')
    print(f'后处理     : {det.postproc_state()}')
    print(f'待检测     : {len(items)} 张\n')

    out_dir = OUT_DIR / f"smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["图片\t类别\t置信度\t预测框[x1,y1,x2,y2]\tGT框\tIoU\t耗时s"]
    ious, times = [], []

    for img_id, p in items:
        img = imread_u(p)
        if img is None:
            print(f'[跳过] 读不了 {p}')
            continue

        def _prog(done, total):
            if done % 24 == 0 or done == total:
                print(f'    ...切片 {done}/{total}', end='\r')

        t1 = time.time()
        dets = det.detect_bgr(img, score_thresh=a.thresh, max_dets=a.topk, progress=_prog)
        sec = time.time() - t1
        times.append(sec)
        gts = load_gts(str(p))

        vis = draw_vis(img, dets, gts, show_gt=not a.no_gt)
        stem = Path(p).stem
        imwrite_u(out_dir / f'{stem}_full.jpg', vis)
        # 缩略全图：原图 4096 宽缩到 1400，方便直接肉眼扫一眼
        tw = 1400
        th_ = max(1, int(img.shape[0] * tw / img.shape[1]))
        imwrite_u(out_dir / f'{stem}_thumb.jpg', cv2.resize(vis, (tw, th_)))

        print(f'[{img_id}] {sec:.1f}s  检出 {len(dets)} 个')
        if not dets:
            lines.append(f"{p.name}\t-\t-\t-\t{len(gts)}个GT\t-\t{sec:.1f}")
            continue
        for d in dets:
            iou = best_gt_iou(d['bbox'], gts, d['class_name'])
            if iou is not None:
                ious.append(iou)
            print(f"    {d['class_name']:8s} conf={d['confidence']:.3f} "
                  f"box={list(d['bbox'])}  IoU={'-' if iou is None else f'{iou:.3f}'}")
            if gts:
                print(f"    {'GT':8s} {[ (round(g[0]),round(g[1]),round(g[2]),round(g[3]),g[4]) for g in gts ]}")
            lines.append(f"{p.name}\t{d['class_name']}\t{d['confidence']:.4f}\t{list(d['bbox'])}\t"
                         f"{[(round(g[0]),round(g[1]),round(g[2]),round(g[3]),g[4]) for g in gts]}\t"
                         f"{'-' if iou is None else f'{iou:.3f}'}\t{sec:.1f}")
            # 缺陷区域放大图（框坐标平移到裁剪坐标系后按裁剪尺寸重画）
            imwrite_u(out_dir / f'{stem}_crop.jpg',
                      crop_with_dets(img, dets, gts, d['bbox'], show_gt=not a.no_gt))

    lines.append(f"\n平均耗时 {np.mean(times):.1f}s/张"
                 + (f" | 平均IoU {np.mean(ious):.3f}（{len(ious)} 个可比对框）" if ious else ""))
    (out_dir / 'summary.txt').write_text('\n'.join(lines), encoding='utf-8')

    print(f'\n总耗时 {time.time()-t0:.1f}s')
    if ious:
        print(f'平均 IoU {np.mean(ious):.3f}（{len(ious)} 个框）')
    print(f'结果图目录: {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
