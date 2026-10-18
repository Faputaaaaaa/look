"""[A15] dirty 标注质量核查（纯标注统计，不跑模型）。

统计全部 dirty GT（train+val，100 张）的：
- 框宽高/长宽比/面积，找尺寸离群（可疑标注之一：框画得不像）
- 生成每张图 GT 区域裁剪图（含上下文），供人工目检框是否盖住真实脏污

输出：
- experiments/label_check/dirty_gt_stats.txt  统计表
- experiments/label_check/crop_*.jpg           GT 区域裁剪图
"""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import sahi_inference as si

OUT = Path("experiments/label_check")
OUT.mkdir(parents=True, exist_ok=True)

MARGIN = 300  # 裁剪上下文边距（px）


def main():
    sp = si.load_split()
    dirty_ids = [i for i in sp['train'] + sp['val'] if i.startswith('dirty/')]
    print(f"dirty 总数: {len(dirty_ids)} (train 80 + val 20)")

    rows = []  # (img_id, w, h, aspect, area)
    for img_id in dirty_ids:
        img_file, lbl_file, _ = si.resolve_image(img_id)
        gts = si.load_gt(img_file, lbl_file)
        dg = [g for g in gts if g[4] == 'dirty']
        if not dg:
            rows.append((img_id, None, None, None, None, len(gts)))
            continue
        # 每图假定 1 个 dirty GT（与 top-1 假设一致），多于 1 个记下来
        g = dg[0]
        w, h = g[2] - g[0], g[3] - g[1]
        rows.append((img_id, w, h, w / h, w * h, len(dg)))

    valid = [r for r in rows if r[1] is not None]
    W = np.array([r[1] for r in valid])
    H = np.array([r[2] for r in valid])
    A = np.array([r[3] for r in valid])  # aspect
    AREA = np.array([r[4] for r in valid])

    med = lambda x: float(np.median(x))
    mad = lambda x: float(np.median(np.abs(x - np.median(x))))  # 中位绝对偏差
    print(f"w: median={med(W):.0f} px, MAD={mad(W):.0f}")
    print(f"h: median={med(H):.0f} px, MAD={mad(H):.0f}")
    print(f"aspect(w/h): median={med(A):.2f}, MAD={mad(A):.2f}")
    print(f"area: median={med(AREA):.0f}, MAD={mad(AREA):.0f}")
    print(f"多 GT 图: {[r[0] for r in rows if r[5] and r[5] > 1] or '无'}")
    print(f"无 dirty GT 图: {[r[0] for r in rows if r[1] is None] or '无'}")

    # 离群判定：robust z-score（|x-med|/MAD > 3.5 任意一项即标记）
    lines = []
    lines.append(f"dirty GT 标注统计（n={len(valid)}）")
    lines.append(f"w  : median={med(W):.0f} MAD={mad(W):.0f}")
    lines.append(f"h  : median={med(H):.0f} MAD={mad(H):.0f}")
    lines.append(f"asp: median={med(A):.2f} MAD={mad(A):.2f}")
    lines.append(f"area: median={med(AREA):.0f} MAD={mad(AREA):.0f}")
    lines.append("")
    lines.append("按 aspect 排序（最瘦高/最扁的在前 12 + 后 6）：")
    order = sorted(valid, key=lambda r: r[3])
    lines.append(f"{'img':32s} {'w':>5s} {'h':>5s} {'w/h':>5s} {'area':>9s} {'z_asp':>6s} {'z_area':>6s}")
    for r in order[:12] + order[-6:]:
        za = abs(r[3] - med(A)) / max(mad(A), 1e-6)
        zar = abs(r[4] - med(AREA)) / max(mad(AREA), 1e-6)
        lines.append(f"{r[0]:32s} {r[1]:5.0f} {r[2]:5.0f} {r[3]:5.2f} {r[4]:9.0f} {za:6.1f} {zar:6.1f}")

    # 可疑 = 尺寸离群（z>3.5）
    suspicious = []
    for r in valid:
        zas = max(abs(r[1] - med(W)) / max(mad(W), 1e-6),
                  abs(r[2] - med(H)) / max(mad(H), 1e-6),
                  abs(r[3] - med(A)) / max(mad(A), 1e-6),
                  abs(r[4] - med(AREA)) / max(mad(AREA), 1e-6))
        if zas > 3.5:
            suspicious.append((r[0], zas))

    # 已知"预测偏移大/形状异常"的重点核查图（A14 + val_vis 发现）
    known = ['dirty/frame_0037.jpg', 'dirty/frame_0041.jpg', 'dirty/frame_0042.jpg',
             'dirty/frame_0045.jpg', 'dirty/frame_0048.jpg', 'dirty/frame_0050.jpg',
             'dirty/frame_0065.jpg', 'dirty/frame_0070.jpg', 'dirty/frame_0078.jpg',
             'dirty/frame_0091.jpg', 'dirty/frame_0010.jpg', 'dirty/frame_0015.jpg']

    lines.append("")
    lines.append(f"尺寸离群 (z>3.5): {len(suspicious)} 张 -> {[s[0] for s in suspicious]}")
    lines.append(f"重点核查（偏移大/形状异常）: {len(known)} 张")

    review_set = sorted(set([s[0] for s in suspicious] + known))
    lines.append(f"待目检合计: {len(review_set)} 张 -> {review_set}")
    print("\n".join(lines))

    # ===== 生成裁剪图 =====
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except Exception:
        font = ImageFont.load_default()

    for img_id in review_set:
        img_file, lbl_file, _ = si.resolve_image(img_id)
        img = Image.open(img_file).convert('RGB')
        iw, ih = img.size
        gts = [g for g in si.load_gt(img_file, lbl_file) if g[4] == 'dirty']
        if not gts:
            continue
        g = gts[0]
        cx1 = int(max(0, g[0] - MARGIN))
        cy1 = int(max(0, g[1] - MARGIN))
        cx2 = int(min(iw, g[2] + MARGIN))
        cy2 = int(min(ih, g[3] + MARGIN))
        crop = img.crop((cx1, cy1, cx2, cy2)).copy()
        draw = ImageDraw.Draw(crop)
        # GT 框（绿，原图坐标-裁剪偏移）
        draw.rectangle([g[0] - cx1, g[1] - cy1, g[2] - cx1, g[3] - cy1],
                       outline=(0, 255, 0), width=6)
        draw.text((g[0] - cx1 + 5, max(0, g[1] - cy1 - 50)), "GT dirty",
                  fill=(0, 255, 0), font=font)
        # GT 中心十字（看框中心是否在脏污上）
        gx, gy = (g[0] + g[2]) / 2 - cx1, (g[1] + g[3]) / 2 - cy1
        draw.line([gx - 30, gy, gx + 30, gy], fill=(255, 255, 0), width=3)
        draw.line([gx, gy - 30, gx, gy + 30], fill=(255, 255, 0), width=3)
        out_path = OUT / f"crop_{Path(img_id).stem}.jpg"
        crop.save(out_path, quality=90)
        print(f"  saved {out_path} (crop {crop.size[0]}x{crop.size[1]})")

    (OUT / "dirty_gt_stats.txt").write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f"\n[OK] 统计 -> {OUT / 'dirty_gt_stats.txt'}")


if __name__ == '__main__':
    main()
