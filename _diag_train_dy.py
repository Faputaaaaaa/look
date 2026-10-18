"""根因检查：训练集 dirty 图跑 SAHI 推理，量 dy 偏移，与验证集对比。

判断逻辑：
- 训练图 dy 也 ≈ +20px → 系统性偏差（模型/标注习惯），后处理补偿合法。
- 训练图 dy ≈ 0 → 偏差只出现在验证集（标注口径不一致或过拟合），需另行决策。

用法: SAHI_MODEL_PATH=... python _diag_train_dy.py
"""
import numpy as np

import sahi_inference as si

CLS_ORDER = [('dirty', 2)]  # 只查 dirty

def measure(img_ids, tag):
    stats = {'tp': 0, 'fp': 0, 'fn': 0, 'dy': [], 'dx': [], 'iou': []}
    for img_id in img_ids:
        img_file, lbl_file, _ = si.resolve_image(img_id)
        dets = si.sahi_inference(model, str(img_file))
        gts = si.load_gt(img_file, lbl_file)
        for cls_name, cls_id in CLS_ORDER:
            cd = [d for d in dets if d[5] == cls_id]
            cg = [g for g in gts if g[4] == cls_name]
            matched = set()
            for d in cd:
                best, bi = 0.0, -1
                for j, g in enumerate(cg):
                    if j in matched:
                        continue
                    iou = si.compute_iou(d[:4], g[:4])
                    if iou > best:
                        best, bi = iou, j
                if best >= 0.5 and bi >= 0:
                    matched.add(bi)
                    stats['tp'] += 1
                    g = cg[bi]
                    gcx, gcy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
                    pcx, pcy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                    stats['dy'].append(pcy - gcy)
                    stats['dx'].append(pcx - gcx)
                    stats['iou'].append(best)
                else:
                    stats['fp'] += 1
            stats['fn'] += len(cg) - len(matched)
        print(f"  [{tag}] {img_id}: done (TP={stats['tp']} FP={stats['fp']} FN={stats['fn']})", flush=True)
    return stats


if __name__ == '__main__':
    model = si.load_model()
    sp = si.load_split()

    # 训练集前 10 张 dirty
    train_ids = [i for i in sp['train'] if i.startswith('dirty/')][:10]
    print(f"=== 训练集 dirty 前 {len(train_ids)} 张 ===", flush=True)
    s = measure(train_ids, 'train')

    dy = np.array(s['dy']) if s['dy'] else np.array([0.0])
    dx = np.array(s['dx']) if s['dx'] else np.array([0.0])
    iou = np.array(s['iou']) if s['iou'] else np.array([0.0])
    print()
    print("=" * 60)
    print(f"训练集 dirty: TP={s['tp']} FP={s['fp']} FN={s['fn']}")
    print(f"  dy(px): mean={dy.mean():+.1f} median={np.median(dy):+.1f} std={dy.std():.1f} n={len(s['dy'])}")
    print(f"  dx(px): mean={dx.mean():+.1f} median={np.median(dx):+.1f} std={dx.std():.1f}")
    print(f"  IoU:   mean={iou.mean():.3f} median={np.median(iou):.3f}")
    print(f"  dy>0 占比: {(dy > 0).mean() * 100:.0f}%")
    print("=" * 60)
