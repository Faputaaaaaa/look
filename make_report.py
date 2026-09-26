"""生成 TRAIN_REPORT.md + val_metrics.json（训练后汇总）。

用法: python make_report.py --exp <name> --log <train.log>
模型从 SAHI_MODEL_PATH 环境变量或 sahi_inference.MODEL_PATH 加载。
"""
import argparse, json, re, os, csv
from pathlib import Path
import numpy as np

import sahi_inference as si

CLS_ORDER = [('missing', 1), ('dirty', 2)]


def evaluate(model, val_ids):
    """对 val 原图跑 SAHI 推理，统计 TP/FP/FN + dx/dy 偏移 + AP。"""
    stats = {1: {'tp': 0, 'fp': 0, 'fn': 0, 'dx': [], 'dy': [], 'scores': [], 'tps': []},
             2: {'tp': 0, 'fp': 0, 'fn': 0, 'dx': [], 'dy': [], 'scores': [], 'tps': []}}
    for img_id in val_ids:
        img_file, lbl_file, subset = si.resolve_image(img_id)
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
                stats[cls_id]['scores'].append(d[4])
                if best >= 0.5 and bi >= 0:
                    matched.add(bi)
                    stats[cls_id]['tp'] += 1
                    stats[cls_id]['tps'].append(1)
                    g = cg[bi]
                    gcx, gcy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
                    pcx, pcy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                    stats[cls_id]['dx'].append(pcx - gcx)
                    stats[cls_id]['dy'].append(pcy - gcy)
                else:
                    stats[cls_id]['fp'] += 1
                    stats[cls_id]['tps'].append(0)
            stats[cls_id]['fn'] += len(cg) - len(matched)
    aps = {}
    for cls_name, cls_id in CLS_ORDER:
        s = stats[cls_id]
        n_pos = s['tp'] + s['fn']
        if s['scores'] and n_pos > 0:
            scores = np.array(s['scores'])
            tps = np.array(s['tps'])
            order = np.argsort(-scores)
            tps = tps[order]
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(1 - tps)
            rec = cum_tp / n_pos
            prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)
            aps[cls_name] = float(si.voc_ap(rec, prec))
        else:
            aps[cls_name] = 0.0
    return stats, aps


def parse_a13(log_path):
    last = None
    try:
        with open(log_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '[A13]' in line:
                    m = re.search(r'正anchor=([\d,]+)\s+负anchor=([\d,]+).*?含缺陷窗口=([\d,]+), 背景窗口=([\d,]+)', line)
                    if m:
                        last = tuple(int(g.replace(',', '')) for g in m.groups())
    except Exception:
        pass
    return last


def parse_lr(csv_path):
    rows = []
    try:
        with open(csv_path, encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f)):
                rows.append((i + 1, float(row['LR'])))
    except Exception:
        pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--log', default=None)
    args = ap.parse_args()

    model = si.load_model()
    sp = si.load_split()
    val_ids = sp['val']

    stats, aps = evaluate(model, val_ids)
    mAP = float(np.mean(list(aps.values())))

    a13 = parse_a13(args.log) if args.log else None
    exp_dir = Path('experiments') / args.exp
    lr_rows = parse_lr(exp_dir / 'training_log.csv')

    # 异常日志
    exc_lines = []
    exc_file = exp_dir / 'exceptions.log'
    if exc_file.exists():
        exc_lines = exc_file.read_text(encoding='utf-8', errors='ignore').strip().splitlines()

    L = []
    L.append('# TRAIN_REPORT')
    L.append('')
    L.append(f'- 实验: `{args.exp}`')
    L.append(f'- 模型: `{os.environ.get("SAHI_MODEL_PATH", si.MODEL_PATH)}`')
    L.append(f'- **mAP@0.5: {mAP:.4f} ({mAP * 100:.1f}%)**')
    L.append('')
    L.append('## 每类指标')
    L.append('')
    L.append('| 类 | AP | Precision | Recall | TP | FP | FN |')
    L.append('|---|---|---|---|---|---|---|')
    metrics = {}
    for cls_name, cls_id in CLS_ORDER:
        s = stats[cls_id]
        p = s['tp'] / max(s['tp'] + s['fp'], 1)
        r = s['tp'] / max(s['tp'] + s['fn'], 1)
        L.append(f'| {cls_name} | {aps[cls_name]:.3f} | {p:.3f} | {r:.3f} | {s["tp"]} | {s["fp"]} | {s["fn"]} |')
        metrics[cls_name] = {'AP': aps[cls_name], 'P': round(p, 3), 'R': round(r, 3),
                             'TP': s['tp'], 'FP': s['fp'], 'FN': s['fn']}
    L.append('')
    L.append('## bbox 回归偏移（pred_cy - gt_cy，原图像素）')
    L.append('')
    for cls_name, cls_id in CLS_ORDER:
        s = stats[cls_id]
        if not s['dy']:
            L.append(f'- {cls_name}: 无 TP')
            continue
        dy = np.array(s['dy'])
        dx = np.array(s['dx'])
        L.append(f'- **{cls_name}**: dy mean={dy.mean():+.1f} median={np.median(dy):+.1f} std={dy.std():.1f} n={len(dy)}')
        L.append(f'  - dx mean={dx.mean():+.1f} median={np.median(dx):+.1f} std={dx.std():.1f}')
        metrics[cls_name]['dy'] = {'mean': round(float(dy.mean()), 1), 'median': round(float(np.median(dy)), 1),
                                   'std': round(float(dy.std()), 1), 'n': len(dy)}
    L.append('')
    L.append('## 背景窗口负 anchor 占比')
    L.append('')
    if a13:
        pos, neg, pos_win, bg = a13
        ratio = bg / max(pos_win + bg, 1)
        L.append(f'- 正 anchor: {pos:,}')
        L.append(f'- 负 anchor: {neg:,}（含缺陷窗口 {pos_win:,} / 背景窗口 {bg:,}）')
        L.append(f'- 背景窗口占比: {ratio * 100:.1f}%')
        metrics['negative'] = {'pos': pos, 'neg': neg, 'pos_win': pos_win, 'bg': bg, 'bg_ratio': round(ratio, 4)}
    else:
        L.append('- 未从日志解析到 [A13] 数据')
    L.append('')
    L.append('## loss 曲线可比性')
    L.append('')
    L.append('- train/val loss 已用同一规则计算（validate 已对齐背景窗口），可直接比较。')
    L.append('')
    L.append('## 每 epoch lr')
    L.append('')
    if lr_rows:
        L.append('| epoch | lr |')
        L.append('|---|---|')
        for e, lr in lr_rows:
            L.append(f'| {e} | {lr:.6f} |')
    else:
        L.append('- 无 lr 记录')
    L.append('')
    L.append('## 异常')
    L.append('')
    if exc_lines:
        for line in exc_lines:
            L.append(f'- {line}')
    else:
        L.append('- 无')

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'TRAIN_REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    (exp_dir / 'val_metrics.json').write_text(json.dumps({'mAP': mAP, 'per_class': metrics}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f"[OK] TRAIN_REPORT.md -> {exp_dir / 'TRAIN_REPORT.md'}")
    print(f"[OK] val_metrics.json -> {exp_dir / 'val_metrics.json'}")


if __name__ == '__main__':
    main()
