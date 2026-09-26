"""C5 验证结果图生成：读 split.json 的 val 原图，SAHI 推理，画 TP/FP/FN 对比图。

输出（experiments/val_vis/）：
- 原图 + GT（绿）+ 预测（红=TP / 品红=FP），GT 未匹配则标黄并加 FN。
- 每张图标类别、置信度、IoU、TP/FP/FN 判定。

用法：
    python evaluate_and_visualize.py --n 10   # 每类 10 张（全量）
    python evaluate_and_visualize.py --n 5    # 每类 5 张（冒烟测试）
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import sahi_inference as si

OUT_DIR = Path("experiments/val_vis")

GT_COLOR = (0, 255, 0)
GT_FN_COLOR = (255, 255, 0)
PRED_TP_COLOR = (255, 0, 0)
PRED_FP_COLOR = (255, 0, 255)


def _font():
    try:
        return ImageFont.truetype("arial.ttf", 48), ImageFont.truetype("arial.ttf", 36)
    except Exception:
        f = ImageFont.load_default()
        return f, f


def match(dets, gts):
    """把 dets 与 gts 匹配，返回 (tp, fp, fn)。"""
    tp, fp = [], []
    matched_gt = set()
    for d in dets:
        best, bi = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched_gt:
                continue
            iou = si.compute_iou(d[:4], g[:4])
            if iou > best:
                best, bi = iou, j
        if best >= 0.5 and bi >= 0:
            matched_gt.add(bi)
            tp.append((d, gts[bi], best))
        else:
            fp.append(d)
    fn = [g for j, g in enumerate(gts) if j not in matched_gt]
    return tp, fp, fn


def draw_one(model, img_path, gts, out_path, title):
    dets = si.sahi_inference(model, str(img_path))
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font, font_s = _font()
    tp, fp, fn = match(dets, gts)

    # 未匹配 GT（FN）：黄色
    fn_gts = set(id(g) for g in fn)
    for g in gts:
        x1, y1, x2, y2, name = g
        color = GT_FN_COLOR if id(g) in fn_gts else GT_COLOR
        draw.rectangle([x1, y1, x2, y2], outline=color, width=8)
        label = f"GT {name}" + (" [FN]" if id(g) in fn_gts else "")
        draw.text((x1, max(0, y1 - 60)), label, fill=color, font=font_s)

    # 预测框
    for d in dets:
        x1, y1, x2, y2, s, c = d
        name = si.MODEL_CLS_TO_NAME.get(c, str(c))
        is_tp = any(d is dd for dd, _, _ in tp)
        color = PRED_TP_COLOR if is_tp else PRED_FP_COLOR
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.text((x1, y2 + 5), f"{name} {s:.2f}", fill=color, font=font_s)

    # 每对 TP 标 IoU
    for d, g, iou in tp:
        x1, y1, x2, y2, s, c = d
        draw.text((x2 - 260, max(0, y1 - 60)), f"TP IoU={iou:.2f}", fill=PRED_TP_COLOR, font=font_s)

    draw.text((20, 20), title, fill=(255, 255, 255), font=font_s)
    img.save(out_path)
    return len(tp), len(fp), len(fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10, help='每类可视化张数')
    args = ap.parse_args()

    model = si.load_model()
    sp = si.load_split()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chosen = []
    for cls_name in ['missing', 'dirty']:
        chosen.extend([i for i in sp['val'] if i.startswith(f"{cls_name}/")][:args.n])

    total_tp = total_fp = total_fn = 0
    for img_id in chosen:
        img_file, lbl_file, subset = si.resolve_image(img_id)
        gts = si.load_gt(img_file, lbl_file)
        out = OUT_DIR / f"val_{Path(img_id).name}"
        tp, fp, fn = draw_one(model, img_file, gts, out, img_id)
        total_tp += tp; total_fp += fp; total_fn += fn
        print(f"  {img_id}: TP={tp} FP={fp} FN={fn}")

    print(f"\n[C5] 汇总 TP={total_tp} FP={total_fp} FN={total_fn} (共 {len(chosen)} 张)")
    print(f"[OK] 可视化已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
