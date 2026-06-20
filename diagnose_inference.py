"""诊断：逐层追踪inference的每一步输出"""
import torch
import sys

print("=" * 70)
print("Inference 逐层诊断 - 精确追踪每一步数值")
print("=" * 70)

from model_fpn import TwoStageDetector, INFERENCE_CONFIG, FPN_CONFIG
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n设备: {device}")

# 创建模型（随机权重 = 模拟未训练状态）
model = TwoStageDetector().to(device)
model.eval()

# 用固定随机种子确保可复现
torch.manual_seed(42)
x = torch.randn(1, 3, 768, 768).to(device)

with torch.no_grad():
    # ===== Step 1: Forward =====
    print("\n" + "=" * 70)
    print("[Step 1] 前向传播")
    features = model.extract_features(x)
    fpn_features = model.fpn(features)
    rpn_cls_preds, rpn_bbox_preds = model.rpn(fpn_features, (768, 768))

    for i, name in enumerate(['P2', 'P3', 'P4']):
        f = fpn_features[i]
        cls_p = rpn_cls_preds[i]
        bbox_p = rpn_bbox_preds[i]
        print(f"\n  {name}: feat={f.shape}, "
              f"cls_pred range=[{cls_p.min().item():.4f}, {cls_p.max().item():.4f}], "
              f"cls_pred mean={cls_p.mean().item():.4f}")

    # ===== Step 2: Anchors + Decode =====
    print("\n" + "=" * 70)
    print("[Step 2] 锚框生成 + 解码")
    feature_sizes = [(f.size(2), f.size(3)) for f in fpn_features]
    anchors_dict = model.rpn.get_anchors(feature_sizes, device)

    total_anchors = sum(a.size(0) for a in anchors_dict.values())
    print(f"  总锚框数: {total_anchors:,}")
    for name in ['P2', 'P3', 'P4']:
        anc = anchors_dict[name]
        sizes = (anc[:, 2] - anc[:, 0]) * (anc[:, 3] - anc[:, 1])
        print(f"  {name}: {anc.size(0):,}个锚框, 尺寸范围=[{sizes.min():.0f}, {sizes.max():.0f}]px")

    # ===== Step 3: RPN Proposal 提取 =====
    print("\n" + "=" * 70)
    print(f"[Step 3] RPN Proposal 提取 (pre_nms={INFERENCE_CONFIG['rpn_pre_nms_top_n']}, "
          f"post_nms={INFERENCE_CONFIG['rpn_post_nms_top_n']}, nms={INFERENCE_CONFIG['rpn_nms_thresh']})")

    from torchvision.ops import nms as tv_nms
    all_proposals = []
    all_prop_scores = []

    for i, layer_name in enumerate(['P2', 'P3', 'P4']):
        anchors = anchors_dict[layer_name]
        cls_pred = rpn_cls_preds[i][0].flatten()   # [B=1, 15, H, W] -> flatten
        bbox_pred = rpn_bbox_preds[i][0].permute(1, 2, 0).contiguous().reshape(-1, 15, 4).reshape(-1, 4)

        # 解码
        boxes = model.decode_boxes(anchors, bbox_pred, img_w=768, img_h=768)

        # 【关键】RPN score threshold 过滤（与 inference() 保持一致）
        score_thresh = INFERENCE_CONFIG['rpn_score_thresh']
        valid_mask = cls_pred > score_thresh
        print(f"\n  {layer_name} ({cls_pred.size(0):,} anchors):")
        print(f"    RPN分数范围: min={cls_pred.min().item():.4f}, max={cls_pred.max().item():.4f}, mean={cls_pred.mean().item():.4f}")
        print(f"    >{score_thresh} 的数量: {valid_mask.sum().item()} / {cls_pred.size(0)}")

        if valid_mask.sum() == 0:
            print(f"    [PASS] 该层全部被过滤! (随机权重的预期行为)")
            continue

        cls_pred = cls_pred[valid_mask]
        boxes = boxes[valid_mask]

        # TopK
        top_n = min(INFERENCE_CONFIG['rpn_pre_nms_top_n'], cls_pred.size(0))
        top_scores, top_inds = cls_pred.topk(top_n)
        top_boxes = boxes[top_inds]

        print(f"\n  {layer_name} ({cls_pred.size(0):,} anchors):")
        print(f"    Top-{top_n} RPN分数: min={top_scores.min().item():.4f}, "
              f"max={top_scores.max().item():.4f}, mean={top_scores.mean().item():.4f}")
        print(f"    >0.5 的数量: {(top_scores > 0.5).sum().item()}, "
              f">0.3 的数量: {(top_scores > 0.3).sum().item()}")

        # NMS
        keep = tv_nms(top_boxes, top_scores.to(top_boxes.dtype), INFERENCE_CONFIG['rpn_nms_thresh'])
        post_nms = min(INFERENCE_CONFIG['rpn_post_nms_top_n'], keep.size(0))
        keep = keep[:post_nms]

        print(f"    NMS后保留: {keep.size(0)} 个")
        print(f"    NMS后分数: min={top_scores[keep].min().item():.4f}, "
              f"max={top_scores[keep].max().item():.4f}")

        all_proposals.append(top_boxes[keep])
        all_prop_scores.append(top_scores[keep])

    # 合并
    proposals = torch.cat(all_proposals, dim=0)
    prop_scores = torch.cat(all_prop_scores, dim=0)
    print(f"\n  各层合并: {proposals.size(0)} 个proposal")

    # 全局NMS
    if proposals.size(0) > 0:
        g_keep = tv_nms(proposals, prop_scores.to(proposals.dtype), 0.7)
        g_keep = g_keep[:min(600, g_keep.size(0))]
        proposals = proposals[g_keep]
        prop_scores = prop_scores[g_keep]
        print(f"  全局NMS后: {proposals.size(0)} 个proposal")

    # ===== Step 4: ROI Head =====
    print("\n" + "=" * 70)
    print(f"[Step 4] ROI Head 分类 (score_thresh={INFERENCE_CONFIG['score_thresh']})")

    from torchvision.ops import roi_align as tv_roi_align

    roi_indices = torch.full((proposals.size(0), 1), 0, dtype=torch.float32, device=device)
    roi_boxes = torch.cat([roi_indices, proposals], dim=1)

    box_sizes = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    sqrt_sizes = torch.sqrt(box_sizes)

    p2_mask = sqrt_sizes < 64
    p3_mask = (sqrt_sizes >= 64) & (sqrt_sizes < 128)
    p4_mask = sqrt_sizes >= 128

    roi_feats = []
    roi_idx = []
    for mask, feat, scale, name in [
        (p2_mask, fpn_features[0], 1/4, 'P2'),
        (p3_mask, fpn_features[1], 1/8, 'P3'),
        (p4_mask, fpn_features[2], 1/16, 'P4'),
    ]:
        if mask.sum() > 0:
            layer_boxes = roi_boxes[mask]
            rf = tv_roi_align(feat, layer_boxes, output_size=(7,7), spatial_scale=scale)
            roi_feats.append(rf)
            roi_idx.append(torch.where(mask)[0])
            print(f"  {name}层ROI: {mask.sum().item()} 个proposal")

    if len(roi_feats) == 0:
        print("\n  [!] 无ROI特征!")
        sys.exit(1)

    roi_feat_cat = torch.cat(roi_feats, dim=0)
    roi_orig_idx = torch.cat(roi_idx, dim=0)

    cls_logits, bbox_deltas = model.roi_head(roi_feat_cat)
    cls_probs = torch.softmax(cls_logits, dim=1)

    scores, labels = cls_probs[:, 1:].max(dim=1)
    labels = labels + 1

    print(f"\n  ROI Head 输出:")
    print(f"    cls_logits 范围: [{cls_logits.min().item():.4f}, {cls_logits.max().item():.4f}]")
    print(f"    softmax概率范围: [{cls_probs.min().item():.4f}, {cls_probs.max().item():.4f}]")
    print(f"    前景类最大概率(scores): min={scores.min().item():.4f}, "
          f"max={scores.max().item():.4f}, mean={scores.mean().item():.4f}")
    print(f"    >0.3 的数量: {(scores > 0.3).sum().item()} / {scores.size(0)}")
    print(f"    >0.5 的数量: {(scores > 0.5).sum().item()} / {scores.size(0)}")

    # Score过滤
    valid_mask = scores > INFERENCE_CONFIG['score_thresh']
    print(f"\n  [score_thresh={INFERENCE_CONFIG['score_thresh']}] 过滤后: {valid_mask.sum().item()} 个")

    if valid_mask.sum() == 0:
        print("\n  >>> 随机权重模型被正确过滤! 这是预期行为 <<<")
        print("      如果训练后的模型也输出大量框，说明模型没有学到有效特征")
    else:
        refined = model.decode_boxes(proposals[roi_orig_idx], bbox_deltas, img_w=768, img_h=768)
        final_boxes = refined[valid_mask]
        final_scores = scores[valid_mask]
        final_labels = labels[valid_mask]

        print(f"\n  最终NMS前: {final_boxes.size(0)} 个检测框")
        print(f"  分数分布: min={final_scores.min().item():.4f}, max={final_scores.max().item():.4f}")

        # 最终NMS
        f_keep = tv_nms(final_boxes, final_scores.to(final_boxes.dtype), INFERENCE_CONFIG['nms_thresh'])
        f_keep = f_keep[:INFERENCE_CONFIG['max_detections']]
        print(f"  最终NMS后: {f_keep.size(0)} 个检测框")

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)
