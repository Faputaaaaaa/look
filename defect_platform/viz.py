# -*- coding: utf-8 -*-
"""[平台] 图像 IO / 画框 / IoU 工具。

单独成模块的目的：不依赖 Qt。
- platform_ui.py（GUI）与 smoke_test.py（无界面冒烟）共用同一套画框逻辑，
  保证"界面看到的框"和"脚本产出的图"完全一致。
"""
from pathlib import Path

import cv2
import numpy as np

CLASS_COLORS = {
    'missing': (0, 200, 255),   # 黄（BGR）
    'dirty': (0, 80, 255),      # 红（BGR）
}
GT_COLOR = (60, 230, 60)        # 绿（BGR）
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
OUT_DIR = Path(__file__).resolve().parent / 'outputs'


def imread_u(path):
    """支持中文路径的读图，强制 3 通道 BGR（避免 RGBA 花屏）。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(path, img, ext='.jpg'):
    """支持中文路径的写图。"""
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def find_label(img_path):
    """按数据集常见布局猜 YOLO 标签路径，找不到返回 None。"""
    p = Path(img_path)
    for c in (
        p.parent.parent / 'labels' / f'{p.stem}.txt',   # <root>/<subset>/images/x.jpg
        p.parent / 'labels' / f'{p.stem}.txt',
        p.with_suffix('.txt'),
    ):
        if c.exists():
            return c
    return None


def load_gts(img_path):
    """读 GT（复用 sahi_inference.load_gt，YOLO 归一化 → 原图像素）。"""
    lp = find_label(img_path)
    if lp is None:
        return []
    import sahi_inference as si
    try:
        return si.load_gt(str(img_path), lp)
    except Exception:
        return []


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def best_gt_iou(bbox, gts, class_name=None):
    """与同类 GT 的最大 IoU（无同类 GT 时对全部 GT 求）。"""
    pool = [g for g in gts if class_name is None or g[4] == class_name] or gts
    if not pool:
        return None
    return max(iou_xyxy(bbox, (g[0], g[1], g[2], g[3])) for g in pool)


def draw_vis(bgr, dets, gts=None, show_gt=True):
    """画检测框（类别色）+ 可选 GT 框（绿）。线宽随原图尺寸自适应。"""
    vis = bgr.copy()
    h, w = bgr.shape[:2]
    f = max(0.45, min(1.0, w / 4096.0))
    th = max(2, int(round(8 * f)))
    fs = max(0.6, 3.0 * f)
    ft = max(2, int(round(6 * f)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if show_gt:
        for g in (gts or []):
            x1, y1, x2, y2, name = int(g[0]), int(g[1]), int(g[2]), int(g[3]), g[4]
            cv2.rectangle(vis, (x1, y1), (x2, y2), GT_COLOR, max(1, th // 2))
            txt = f'GT {name}'
            (tw, tth), bl = cv2.getTextSize(txt, font, fs * 0.7, max(1, ft // 2))
            ty = min(h - 2, y2 + tth + bl + 6)
            cv2.rectangle(vis, (x1, y2), (x1 + tw, ty), GT_COLOR, -1)
            cv2.putText(vis, txt, (x1, ty - bl), font, fs * 0.7, (20, 40, 20), max(1, ft // 2))

    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        color = CLASS_COLORS.get(d['class_name'], (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, th)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (tw, tth), bl = cv2.getTextSize(label, font, fs, ft)
        top = max(0, y1 - tth - bl - 6)
        cv2.rectangle(vis, (x1, top), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - bl - 3), font, fs, (255, 255, 255), ft)
    return vis


def crop_around(bgr, bbox, pad_ratio=0.8, min_pad=120):
    """裁一个以 bbox 为中心、带 padding 的放大区域（便于人眼看细节）。"""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(min_pad, max(bw, bh) * pad_ratio))
    cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad); cy2 = min(h, y2 + pad)
    return bgr[cy1:cy2, cx1:cx2].copy()


def crop_with_dets(bgr, dets, gts, focus_bbox, pad_ratio=0.8, min_pad=120, show_gt=True):
    """裁剪缺陷区域并在裁剪图上重画框。

    直接裁 draw_vis 的输出会带上一堆按 4096 宽图定的粗线/大字；
    这里先把裁剪区域算好，把框坐标平移到裁剪坐标系后按裁剪尺寸重画，
    这样放大图看着正常，也方便判断"框贴合度"。
    """
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = focus_bbox
    pad = int(max(min_pad, max(x2 - x1, y2 - y1) * pad_ratio))
    cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad); cy2 = min(h, y2 + pad)

    sub = bgr[cy1:cy2, cx1:cx2].copy()
    d2 = [{
        'bbox': (d['bbox'][0] - cx1, d['bbox'][1] - cy1, d['bbox'][2] - cx1, d['bbox'][3] - cy1),
        'class_name': d['class_name'],
        'confidence': d['confidence'],
        'class_id': d.get('class_id', 0),
    } for d in dets]
    g2 = [(g[0] - cx1, g[1] - cy1, g[2] - cx1, g[3] - cy1, g[4]) for g in (gts or [])]
    return draw_vis(sub, d2, g2, show_gt=show_gt)
