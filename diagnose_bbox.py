"""B1/B8 诊断：验证集 bbox 回归误差统计（中心偏移 / 宽高偏差），按类分层。

只做诊断，不修改核心代码，不删除 1.3x/1.1x 后处理。
输出：
1. 每类 center offset（px）、w/h 比例（pred/GT）、IoU 的统计。
2. 前 10 张可视化（GT 绿 + 原始预测框红）到 experiments/diagnosis/。
"""
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

import sahi_inference as si

OUT_DIR = Path("experiments/diagnosis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sahi_raw(model, img_path):
    """SAHI 推理，返回原始检测框（不应用 1.3x/1.1x 扩张），top-1。"""
    img = Image.open(img_path).convert('RGB')
    iw, ih = img.size
    all_dets = []
    for y in range(0, ih, si.STRIDE):
        for x in range(0, iw, si.STRIDE):
            x2 = min(x + si.WINDOW, iw)
            y2 = min(y + si.WINDOW, ih)
            x1 = max(0, x2 - si.WINDOW)
            y1 = max(0, y2 - si.WINDOW)
            if x2 - x1 < 64 or y2 - y1 < 64:
                continue
            crop = img.crop((x1, y1, x2, y2))
            if crop.size != (si.IMG_SIZE, si.IMG_SIZE):
                padded = Image.new('RGB', (si.IMG_SIZE, si.IMG_SIZE), (128, 128, 128))
                padded.paste(crop, (0, 0))
                crop = padded
            for dx1, dy1, dx2, dy2, s, c in si.infer_single_crop(model, si.preprocess_crop(crop)):
                edges = (dx1 < 5) + (dy1 < 5) + (dx2 > si.IMG_SIZE - 5) + (dy2 > si.IMG_SIZE - 5)
                if edges >= 3:
                    continue
                all_dets.append((dx1 + x1, dy1 + y1, dx2 + x1, dy2 + y1, s, c))
    if not all_dets:
        return []
    boxes = torch.tensor([[d[0], d[1], d[2], d[3]] for d in all_dets], dtype=torch.float32)
    scores = torch.tensor([d[4] for d in all_dets])
    classes = torch.tensor([d[5] for d in all_dets], dtype=torch.long)
    final = []
    for cls in classes.unique().tolist():
        m = classes == cls
        if m.sum() == 0:
            continue
        keep = nms(boxes[m], scores[m], si.NMS_THRESH)
        bm, sm = boxes[m], scores[m]
        for i in keep:
            final.append((bm[i, 0].item(), bm[i, 1].item(), bm[i, 2].item(), bm[i, 3].item(),
                          sm[i].item(), cls))
        if len(final) >= si.MAX_DET * 2:
            break
    final.sort(key=lambda d: -d[4])
    return final[:1]


def draw_compare(img_path, gts, dets, out_path, title):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    for x1, y1, x2, y2, name in gts:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=8)
        draw.text((x1, max(0, y1 - 55)), f"GT {name}", fill=(0, 255, 0), font=font)
    for x1, y1, x2, y2, s, c in dets:
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)
        draw.text((x1, y2 + 5), f"pred {s:.2f}", fill=(255, 0, 0), font=font)
    draw.text((20, 20), title, fill=(255, 255, 0), font=font)
    img.save(out_path)


def main():
    model = si.load_model()
    sp = si.load_split()
    val_ids = sp['val']

    per_class = 10
    chosen = []
    for cls_name in ['missing', 'dirty']:
        chosen.extend([i for i in val_ids if i.startswith(f"{cls_name}/")][:per_class])

    stats = {1: {'dx': [], 'dy': [], 'wr': [], 'hr': [], 'iou': []},
             2: {'dx': [], 'dy': [], 'wr': [], 'hr': [], 'iou': []}}
    tp = {1: 0, 2: 0}
    fn = {1: 0, 2: 0}
    fp = {1: 0, 2: 0}
    vis_count = 0

    for img_id in chosen:
        img_file, lbl_file, subset = si.resolve_image(img_id)
        dets = sahi_raw(model, str(img_file))
        gts = si.load_gt(img_file, lbl_file)

        for cls_name, cls_id in [('missing', 1), ('dirty', 2)]:
            cd = [d for d in dets if d[5] == cls_id]
            cg = [g for g in gts if g[4] == cls_name]
            matched = set()
            for d in cd:
                best, bi = 0, -1
                for j, g in enumerate(cg):
                    if j in matched:
                        continue
                    iou = si.compute_iou(d[:4], g[:4])
                    if iou > best:
                        best, bi = iou, j
                if best >= 0.5 and bi >= 0:
                    matched.add(bi)
                    tp[cls_id] += 1
                    g = cg[bi]
                    gcx, gcy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
                    gw, gh = g[2] - g[0], g[3] - g[1]
                    pcx, pcy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                    pw, ph = d[2] - d[0], d[3] - d[1]
                    stats[cls_id]['dx'].append(pcx - gcx)
                    stats[cls_id]['dy'].append(pcy - gcy)
                    stats[cls_id]['wr'].append(pw / gw)
                    stats[cls_id]['hr'].append(ph / gh)
                    stats[cls_id]['iou'].append(best)
                else:
                    fp[cls_id] += 1
            fn[cls_id] += len(cg) - len(matched)

        if vis_count < 10:
            vis_count += 1
            det_lines = [f"{si.MODEL_CLS_TO_NAME.get(d[5], d[5])} {d[4]:.2f}" for d in dets]
            title = f"{img_id}  pred={det_lines}"
            draw_compare(str(img_file), gts, dets, str(OUT_DIR / f"diag_{Path(img_id).name}"), title)

    # 汇总
    lines = []
    lines.append("=" * 70)
    lines.append("B1/B8 诊断：原始框回归误差（未应用 1.3x/1.1x 扩张）")
    lines.append(f"评估 val 原图 {len(chosen)} 张（missing/dirty 各 {per_class}）")
    lines.append("=" * 70)
    for cls_name, cls_id in [('missing', 1), ('dirty', 2)]:
        s = stats[cls_id]
        lines.append(f"\n[{cls_name}]  TP={tp[cls_id]} FP={fp[cls_id]} FN={fn[cls_id]}")
        if len(s['dx']) == 0:
            lines.append("  无 TP，无法统计回归误差")
            continue
        dx = np.array(s['dx']); dy = np.array(s['dy'])
        wr = np.array(s['wr']); hr = np.array(s['hr'])
        iou = np.array(s['iou'])
        lines.append(f"  中心偏移 dx(px): mean={dx.mean():+.1f} median={np.median(dx):+.1f} std={dx.std():.1f}")
        lines.append(f"  中心偏移 dy(px): mean={dy.mean():+.1f} median={np.median(dy):+.1f} std={dy.std():.1f}")
        lines.append(f"  宽度比 w_pred/w_gt: mean={wr.mean():.3f} median={np.median(wr):.3f}")
        lines.append(f"  高度比 h_pred/h_gt: mean={hr.mean():.3f} median={np.median(hr):.3f}")
        lines.append(f"  IoU: mean={iou.mean():.3f} median={np.median(iou):.3f}")
        # 扩张补偿因子 = 1/median_ratio
        lines.append(f"  -> 需要扩张补偿 ≈ 1/{np.median(wr):.3f} (w) / 1/{np.median(hr):.3f} (h)")
    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "bbox_regression.txt").write_text(report + "\n", encoding='utf-8')
    print(f"\n[OK] 报告已保存: {OUT_DIR / 'bbox_regression.txt'}")
    print(f"[OK] 可视化已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
