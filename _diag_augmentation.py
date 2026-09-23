"""[R42 审查] 滑动窗口路径增强适用性分析
分析切片内缺陷的位置/尺寸分布，评估每个增强操作的风险
"""
import numpy as np
from pathlib import Path
from PIL import Image
import random
from config import config

DATA_DIR = Path(config.data_dir)
SUBSETS = ['missing', 'dirty']
WINDOW_SIZE = config.window_size  # 640
STRIDE = config.window_stride    # 320

# 收集所有含缺陷窗口的统计信息
all_windows_info = []

for subset in SUBSETS:
    img_dir = DATA_DIR / subset / "images"
    lbl_dir = DATA_DIR / subset / "labels"

    for img_file in sorted(img_dir.glob("*.jpg"))[:20]:
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        if not lbl_file.exists():
            continue

        try:
            img = Image.open(img_file)
            img_w, img_h = img.size
        except:
            img_w, img_h = config.original_img_width, config.original_img_height

        # 读取GT
        gts = []
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in [3, 4]:  # missing=3, dirty=4
                        cls_name = 'missing' if cls_id == 3 else 'dirty'
                        cx = float(parts[1]) * img_w
                        cy = float(parts[2]) * img_h
                        w = float(parts[3]) * img_w
                        h = float(parts[4]) * img_h
                        gts.append((cx, cy, w, h, cls_name))

        if not gts:
            continue

        # 生成窗口
        windows = []
        for y in range(0, img_h - WINDOW_SIZE + 1, STRIDE):
            for x in range(0, img_w - WINDOW_SIZE + 1, STRIDE):
                windows.append({'x': x, 'y': y})
        if img_w > WINDOW_SIZE:
            x = img_w - WINDOW_SIZE
            for y in range(0, img_h - WINDOW_SIZE + 1, STRIDE):
                windows.append({'x': x, 'y': y})
        if img_h > WINDOW_SIZE:
            y = img_h - WINDOW_SIZE
            for x in range(0, img_w - WINDOW_SIZE + 1, STRIDE):
                windows.append({'x': x, 'y': y})

        # 找含缺陷的窗口
        for window in windows:
            for cx, cy, w, h, cls_name in gts:
                if (window['x'] <= cx <= window['x'] + WINDOW_SIZE and
                    window['y'] <= cy <= window['y'] + WINDOW_SIZE):

                    # 切片内归一化坐标
                    local_cx = (cx - window['x']) / WINDOW_SIZE
                    local_cy = (cy - window['y']) / WINDOW_SIZE
                    local_w = w / WINDOW_SIZE
                    local_h = h / WINDOW_SIZE

                    # 计算bbox边缘到切片边缘的距离（归一化）
                    left_dist = local_cx - local_w/2
                    right_dist = 1 - (local_cx + local_w/2)
                    top_dist = local_cy - local_h/2
                    bottom_dist = 1 - (local_cy + local_h/2)

                    all_windows_info.append({
                        'cls': cls_name,
                        'local_cx': local_cx,
                        'local_cy': local_cy,
                        'local_w': local_w,
                        'local_h': local_h,
                        'left_dist': left_dist,
                        'right_dist': right_dist,
                        'top_dist': top_dist,
                        'bottom_dist': bottom_dist,
                        'min_dist': min(left_dist, right_dist, top_dist, bottom_dist),
                        'touched_edges': sum([left_dist < 0.05, right_dist < 0.05,
                                              top_dist < 0.05, bottom_dist < 0.05]),
                    })
                    break  # 每个窗口只记录第一个缺陷

print(f"\n{'='*70}")
print(f"滑动窗口路径增强适用性分析")
print(f"总含缺陷窗口数: {len(all_windows_info)}")
print(f"{'='*70}")

# ============ 1. 缺陷在切片内的位置分布 ============
print(f"\n--- 1. 缺陷中心在切片内的位置分布 ---")
cx_arr = np.array([w['local_cx'] for w in all_windows_info])
cy_arr = np.array([w['local_cy'] for w in all_windows_info])

for cls in ['missing', 'dirty']:
    cls_cx = np.array([w['local_cx'] for w in all_windows_info if w['cls'] == cls])
    cls_cy = np.array([w['local_cy'] for w in all_windows_info if w['cls'] == cls])
    print(f"\n  {cls} (n={len(cls_cx)}):")
    print(f"    切片内cx: mean={cls_cx.mean():.3f}, median={np.median(cls_cx):.3f}, std={cls_cx.std():.3f}")
    print(f"    切片内cy: mean={cls_cy.mean():.3f}, median={np.median(cls_cy):.3f}, std={cls_cy.std():.3f}")
    print(f"    cx范围: [{cls_cx.min():.3f}, {cls_cx.max():.3f}]")
    print(f"    cy范围: [{cls_cy.min():.3f}, {cls_cy.max():.3f}]")

    # 分象限统计
    q1 = np.sum((cls_cx < 0.5) & (cls_cy < 0.5))
    q2 = np.sum((cls_cx >= 0.5) & (cls_cy < 0.5))
    q3 = np.sum((cls_cx < 0.5) & (cls_cy >= 0.5))
    q4 = np.sum((cls_cx >= 0.5) & (cls_cy >= 0.5))
    total = len(cls_cx)
    print(f"    象限分布: Q1(左上)={q1}({q1/total*100:.0f}%), Q2(右上)={q2}({q2/total*100:.0f}%), "
          f"Q3(左下)={q3}({q3/total*100:.0f}%), Q4(右下)={q4}({q4/total*100:.0f}%)")

# ============ 2. 缺陷在切片内的尺寸分布 ============
print(f"\n--- 2. 缺陷在切片内的尺寸分布 ---")
for cls in ['missing', 'dirty']:
    cls_w = np.array([w['local_w'] for w in all_windows_info if w['cls'] == cls])
    cls_h = np.array([w['local_h'] for w in all_windows_info if w['cls'] == cls])
    print(f"\n  {cls}:")
    print(f"    宽度占比: mean={cls_w.mean():.3f}, median={np.median(cls_w):.3f}")
    print(f"    高度占比: mean={cls_h.mean():.3f}, median={np.median(cls_h):.3f}")
    print(f"    最大宽度占比: {cls_w.max():.3f} (像素: {cls_w.max()*640:.0f})")
    print(f"    最大高度占比: {cls_h.max():.3f} (像素: {cls_h.max()*640:.0f})")

# ============ 3. 缺陷到切片边缘的距离 ============
print(f"\n--- 3. 缺陷到切片边缘的距离（归一化）---")
for cls in ['missing', 'dirty']:
    cls_info = [w for w in all_windows_info if w['cls'] == cls]
    min_dists = np.array([w['min_dist'] for w in cls_info])
    touched = np.array([w['touched_edges'] for w in cls_info])
    print(f"\n  {cls}:")
    print(f"    最小边缘距离: mean={min_dists.mean():.3f}, median={np.median(min_dists):.3f}")
    print(f"    最小边缘距离<0.1（贴边）: {np.sum(min_dists < 0.1)}/{len(min_dists)} ({np.mean(min_dists<0.1)*100:.1f}%)")
    print(f"    最小边缘距离<0.05（严重贴边）: {np.sum(min_dists < 0.05)}/{len(min_dists)} ({np.mean(min_dists<0.05)*100:.1f}%)")
    print(f"    贴边数≥2的窗口: {np.sum(touched >= 2)}/{len(touched)} ({np.mean(touched>=2)*100:.1f}%)")

# ============ 4. 各增强操作风险评估 ============
print(f"\n{'='*70}")
print(f"各增强操作在滑动窗口路径下的风险评估")
print(f"{'='*70}")

# 4.1 水平翻转评估
print(f"\n--- 4.1 水平翻转 (p=0.5) ---")
print(f"  操作: 切片内cx → 1 - cx")
print(f"  影响范围: bbox位置同步调整")
print(f"  风险评估:")
print(f"    - 缺陷是否会被裁出切片? 否（镜像变换保持缺陷在切片内）")
print(f"    - 翻转后缺陷是否合理? 是（瓶身水平对称）")
print(f"    - 推理一致性? 一致（推理时不翻转，训练时翻转只增加样本多样性）")
print(f"  结论: ✅ 零风险，强烈推荐")

# 4.2 垂直翻转评估
print(f"\n--- 4.2 垂直翻转 (p=0.3) ---")
print(f"  操作: 切片内cy → 1 - cy")
print(f"  影响范围: bbox位置同步调整")
print(f"  风险评估:")
# 检查cy偏置
for cls in ['missing', 'dirty']:
    cls_cy = np.array([w['local_cy'] for w in all_windows_info if w['cls'] == cls])
    bias = cls_cy.mean() - 0.5
    print(f"    - {cls}类cy偏置: mean={cls_cy.mean():.3f}, 偏移={bias:+.3f}", end="")
    if abs(bias) > 0.05:
        print(f" → 有显著偏置，垂直翻转可有效缓解")
    else:
        print(f" → 偏置较小")
print(f"    - 翻转后缺陷是否合理? 需讨论（瓶身有方向性）")
print(f"    - 推理一致性? 一致")
print(f"  结论: ⚠️ 低风险，需确认瓶身方向性")

# 4.3 ColorJitter评估
print(f"\n--- 4.3 ColorJitter ---")
print(f"  操作: 亮度±15%, 对比度±15%")
print(f"  影响范围: 仅像素值，不改变位置")
print(f"  风险评估:")
print(f"    - 是否影响bbox? 否")
print(f"    - 是否影响坐标映射? 否")
print(f"    - 是否影响推理? 否")
print(f"  结论: ✅ 零风险，强烈推荐")

# 4.4 RandomCrop评估
print(f"\n--- 4.4 RandomCrop (p=0.35, crop_ratio=0.7-1.0) ---")
print(f"  操作: 切片内再裁剪70%-100%区域，再resize回640")
print(f"  影响范围: bbox位置和尺度同步变化")
# 计算crop后缺陷可能被裁掉的概率
for cls in ['missing', 'dirty']:
    cls_info = [w for w in all_windows_info if w['cls'] == cls]
    # crop_ratio=0.7时，保留区域是0.7×0.7=49%
    # 缺陷中心在保留区域内的概率
    in_area = sum(1 for w in cls_info
                  if 0.15 < w['local_cx'] < 0.85 and 0.15 < w['local_cy'] < 0.85)
    total = len(cls_info)
    print(f"  {cls}类:")
    print(f"    - crop_ratio=0.7时，缺陷中心在保留区(0.15-0.85)内的比例: {in_area}/{total} ({in_area/total*100:.1f}%)")
    print(f"    - 缺陷可能被裁掉的概率: {(1-in_area/total)*100:.1f}%")
print(f"  风险评估:")
print(f"    - 可能丢失部分缺陷样本（需补丁处理）")
print(f"    - 缩小视野后缺陷相对变大，但切片已640×640，再缩小可能丢失上下文")
print(f"  结论: ⚠️ 中等风险，需要复杂的bbox处理逻辑")

# 4.5 马赛克评估
print(f"\n--- 4.5 马赛克 (p=0.5) ---")
print(f"  操作: 4个不同切片拼接到一张640×640")
print(f"  影响范围: 完全重构图像结构")
print(f"  风险评估:")
print(f"    - 实现复杂: 需要从4个不同原图/窗口采样")
print(f"    - 坐标映射: 需要重新计算每个窗口的偏移")
print(f"    - 背景真实性: 4个不同位置的背景拼接可能产生不真实纹理")
print(f"    - 缺陷组合: 可能产生不合理的多缺陷组合")
print(f"  结论: ❌ 高风险，实现复杂，收益不确定")

print(f"\n{'='*70}")
print(f"最终建议")
print(f"{'='*70}")
print(f"1. 立即启用: 水平翻转 + ColorJitter（零风险）")
print(f"2. 谨慎评估: 垂直翻转（需确认瓶身方向性）")
print(f"3. 暂缓: RandomCrop（中等风险，需复杂处理）")
print(f"4. 不推荐: 马赛克（高风险，实现复杂）")
