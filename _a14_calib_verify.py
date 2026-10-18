"""[A14] dy 补偿标定 + 验证（诚实调参协议）。

协议（防"考前看答案"）：
- 标定：val dirty 前 10 张 → 量原始预测框相对 GT 的 dy（不做 IoU 门槛，
  每图恰好 1 GT + top-1 预测，取该对 dy）→ 补偿值 = -median(dy)。
- 验证：val dirty 后 10 张 → 同一推理结果，对比"原始"与"补偿后"两套指标。
  两套均只算 dirty 类 TP/FP/FN/P/R + IoU。

补偿方式与 sahi_inference.py [A14] 代码路径数学等价（对 dirty 框 y1/y2 平移），
本脚本不改 config 开关（默认关），验证通过后再启用。

用法: SAHI_MODEL_PATH=... python -u _a14_calib_verify.py
"""
import numpy as np

import sahi_inference as si

COMP_APPLY_CLS = 2  # dirty


def match_dy(dets, gts):
    """单图 top-1 预测 vs 唯一 dirty GT，返回 (dy, iou)；无预测/无 GT 返回 None。"""
    cd = [d for d in dets if d[5] == COMP_APPLY_CLS]
    cg = [g for g in gts if g[4] == 'dirty']
    if not cd or not cg:
        return None
    d = cd[0]
    best_iou, best_g = 0.0, None
    for g in cg:
        iou = si.compute_iou(d[:4], g[:4])
        if iou > best_iou:
            best_iou, best_g = iou, g
    gcx, gcy = (best_g[0] + best_g[2]) / 2, (best_g[1] + best_g[3]) / 2
    pcx, pcy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
    return pcy - gcy, best_iou


def eval_dets(dets_by_img, gts_by_img, comp):
    """对 {img: dets} 评估 dirty 类 TP/FP/FN/P/R + TP 的 IoU 列表。"""
    tp = fp = fn = 0
    ious = []
    for img_id, dets in dets_by_img.items():
        gts = gts_by_img[img_id]
        if comp != 0.0:
            dets = [(x1, y1 + comp, x2, y2 + comp, s, c) if c == COMP_APPLY_CLS
                    else (x1, y1, x2, y2, s, c) for x1, y1, x2, y2, s, c in dets]
        cd = [d for d in dets if d[5] == COMP_APPLY_CLS]
        cg = [g for g in gts if g[4] == 'dirty']
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
                tp += 1
                ious.append(best)
            else:
                fp += 1
        fn += len(cg) - len(matched)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return tp, fp, fn, p, r, ious


if __name__ == '__main__':
    assert not si.config.use_dy_compensation, "标定期 config 开关必须为关"
    model = si.load_model()
    sp = si.load_split()
    val_dirty = [i for i in sp['val'] if i.startswith('dirty/')]
    calib_ids, verify_ids = val_dirty[:10], val_dirty[10:]
    print(f"标定集: {len(calib_ids)} 张 | 验证集: {len(verify_ids)} 张", flush=True)

    # ===== 标定 =====
    dys = []
    calib_dets, calib_gts = {}, {}
    for img_id in calib_ids:
        img_file, lbl_file, _ = si.resolve_image(img_id)
        dets = si.sahi_inference(model, str(img_file))
        gts = si.load_gt(img_file, lbl_file)
        calib_dets[img_id], calib_gts[img_id] = dets, gts
        m = match_dy(dets, gts)
        if m:
            dys.append(m[0])
        print(f"  [calib] {img_id}: dy={m[0]:+.1f} iou={m[1]:.3f}" if m
              else f"  [calib] {img_id}: 无预测或无GT", flush=True)

    dys = np.array(dys)
    comp = -float(np.median(dys))
    print(f"\n标定 dy: mean={dys.mean():+.1f} median={np.median(dys):+.1f} "
          f"std={dys.std():.1f} n={len(dys)}", flush=True)
    print(f"==> 补偿值 dy_comp_dirty = {comp:+.1f}px", flush=True)

    # ===== 验证（后 10 张，原始 vs 补偿） =====
    ver_dets, ver_gts = {}, {}
    for img_id in verify_ids:
        img_file, lbl_file, _ = si.resolve_image(img_id)
        dets = si.sahi_inference(model, str(img_file))
        gts = si.load_gt(img_file, lbl_file)
        ver_dets[img_id], ver_gts[img_id] = dets, gts
        m = match_dy(dets, gts)
        print(f"  [verify] {img_id}: dy={m[0]:+.1f} iou={m[1]:.3f}" if m
              else f"  [verify] {img_id}: 无预测或无GT", flush=True)

    tp0, fp0, fn0, p0, r0, iou0 = eval_dets(ver_dets, ver_gts, 0.0)
    tp1, fp1, fn1, p1, r1, iou1 = eval_dets(ver_dets, ver_gts, comp)

    print()
    print("=" * 62)
    print(f"[A14] 验证（val dirty 后 10 张，未参与标定）")
    print(f"  补偿前: TP={tp0} FP={fp0} FN={fn0} P={p0:.3f} R={r0:.3f} "
          f"IoU_mean={np.mean(iou0) if iou0 else 0:.3f}")
    print(f"  补偿后: TP={tp1} FP={fp1} FN={fn1} P={p1:.3f} R={r1:.3f} "
          f"IoU_mean={np.mean(iou1) if iou1 else 0:.3f}  (comp={comp:+.1f}px)")
    print("=" * 62)

    # 标定集自身也报一下（参考，不作成绩）
    tpc, fpc, fnc, pc, rc, iouc = eval_dets(calib_dets, calib_gts, comp)
    print(f"[参考] 标定集补偿后: TP={tpc} FP={fpc} FN={fnc} P={pc:.3f} R={rc:.3f} "
          f"IoU_mean={np.mean(iouc) if iouc else 0:.3f}")
