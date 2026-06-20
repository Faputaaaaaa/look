"""
RPN Proposal诊断脚本 - 深度分析定位精度差的根因
目标：
1. 检查RPN生成的proposal尺寸分布
2. 分析proposal与GT框的IoU
3. 检查ROI Head的bbox回归效果
4. 定位精度问题的根本原因
"""
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from model_fpn import TwoStageDetector, FPN_CONFIG
from dataset import DefectDataset

config.setup_experiment()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {device}")

# ============================================================
# 1. 加载模型
# ============================================================
print("\n" + "="*70)
print("加载模型...")
print("="*70)

model = TwoStageDetector().to(device)
MODEL_PATH = "experiments/exp_20260608_223914/models/best_model.pth"
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
print(f"[OK] 模型加载成功: {MODEL_PATH}")

# ============================================================
# 2. 加载测试集
# ============================================================
print("\n" + "="*70)
print("加载测试集...")
print("="*70)

test_dataset = DefectDataset(
    img_dir=config.test_img_dir,
    label_dir=config.test_label_dir,
    mode='test'
)
print(f"[OK] 测试集: {len(test_dataset)} 张图片")

# ============================================================
# 3. 诊断函数
# ============================================================

def diagnose_rpn_proposals(model, dataset, num_samples=10, max_proposals_per_layer=1000):
    """诊断RPN生成的proposal质量和分布"""

    print("\n" + "="*70)
    print("【诊断1】RPN Proposal尺寸与分数分布")
    print("="*70)

    all_proposal_widths = []
    all_proposal_heights = []
    all_proposal_areas = []
    all_proposal_scores = []

    all_gt_widths = []
    all_gt_heights = []
    all_gt_areas = []

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        image_tensor = sample[0].unsqueeze(0).to(device)  # [1, 3, H, W]

        # 获取GT信息
        num_objects = sample[1].item()
        if num_objects > 0 and len(sample) > 2:
            gt_bboxes = sample[2]  # [N, 5] normalized (cx, cy, w, h, class_id)
            img_size = config.img_size
            for i in range(num_objects):
                cx, cy, w, h, class_id = gt_bboxes[i].tolist()
                w_px = w * img_size
                h_px = h * img_size
                all_gt_widths.append(w_px)
                all_gt_heights.append(h_px)
                all_gt_areas.append(w_px * h_px)

        # 前向传播获取RPN proposals
        with torch.no_grad():
            # 提取特征
            features = model.extract_features(image_tensor)
            fpn_features = model.fpn(features)  # [P2, P3, P4]

            # RPN前向传播
            img_h, img_w = image_tensor.shape[-2:]
            rpn_cls_preds, rpn_bbox_preds = model.rpn(fpn_features, (img_h, img_w))

            # 生成anchors
            feature_sizes = [(f.shape[-2], f.shape[-1]) for f in fpn_features]
            anchors = model.rpn.anchor_generator.generate_all_anchors(feature_sizes)

            # 对每层生成proposals
            layer_names = ['P2', 'P3', 'P4']
            for idx, level in enumerate(layer_names):
                cls_score = rpn_cls_preds[idx]  # [B, 15, H, W]
                bbox_pred = rpn_bbox_preds[idx]  # [B, 60, H, W]
                anchor = anchors[level]  # [A*H*W, 4]

                B, _, H, W = cls_score.shape

                # Reshape: [B, 15, H, W] -> [B*H*W, 15]
                cls_score_reshaped = cls_score.permute(0, 2, 3, 1).reshape(-1, 15)  # [A*H*W, 15]
                bbox_pred_reshaped = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 60)   # [A*H*W, 60]

                num_anchors_per_loc = 15  # 5 scales * 3 ratios

                # Objectness分数 (sigmoid后的前景概率)
                obj_probs = torch.sigmoid(cls_score_reshaped)  # [A*H*W, 15]

                # 筛选高置信度的proposals (使用较高的阈值以减少数量)
                thresh = 0.5
                valid_mask = obj_probs > thresh

                if valid_mask.sum() == 0:
                    continue

                # 获取所有有效位置的索引
                valid_flat_mask = valid_mask.reshape(-1)  # [total_positions * 15]
                valid_indices = valid_flat_mask.nonzero(as_tuple=True)[0]  # 全局索引

                # 限制每层的最大proposal数量以提高性能
                if len(valid_indices) > max_proposals_per_layer:
                    # 选择置信度最高的proposals
                    valid_scores_temp = obj_probs.reshape(-1)[valid_indices]
                    topk_scores, topk_indices = torch.topk(valid_scores_temp, max_proposals_per_layer)
                    valid_indices = valid_indices[topk_indices]

                # 计算每个有效索引对应的位置和anchor
                position_idx = valid_indices // num_anchors_per_loc  # 位置索引
                anchor_loc_idx = valid_indices % num_anchors_per_loc   # anchor局部索引

                # 确保anchor在正确的设备上
                anchor = anchor.to(device)
                valid_anchors = anchor[position_idx]      # [N, 4]

                # 获取对应的bbox predictions
                position_bbox_preds = bbox_pred_reshaped[position_idx]  # [N, 60]

                # 对于每个位置，选择对应anchor的4个坐标
                coord_start_idx = anchor_loc_idx.unsqueeze(1) * 4  # [N, 1]
                coord_indices = coord_start_idx + torch.arange(4, device=device).unsqueeze(0)  # [N, 4]
                valid_bbox_preds = position_bbox_preds.gather(1, coord_indices)  # [N, 4]

                valid_scores = obj_probs.reshape(-1)[valid_indices]    # [N]

                # 解码proposals (anchor + delta)
                proposals = model.decode_boxes(valid_anchors, valid_bbox_preds)  # [N, 4]

                # 向量化计算宽高 (避免循环)
                x1 = proposals[:, 0]
                y1 = proposals[:, 1]
                x2 = proposals[:, 2]
                y2 = proposals[:, 3]

                widths = (x2 - x1).abs().cpu().numpy()
                heights = (y2 - y1).abs().cpu().numpy()
                areas = widths * heights
                scores = valid_scores.cpu().numpy()

                all_proposal_widths.extend(widths.tolist())
                all_proposal_heights.extend(heights.tolist())
                all_proposal_areas.extend(areas.tolist())
                all_proposal_scores.extend(scores.tolist())

    # 统计分析
    print(f"\n样本数: {num_samples}")
    print(f"\n{'='*50}")
    print("【RPN Proposal统计】")
    print(f"{'='*50}")
    print(f"总Proposal数: {len(all_proposal_areas)}")
    if len(all_proposal_areas) > 0:
        widths = np.array(all_proposal_widths)
        heights = np.array(all_proposal_heights)
        areas = np.array(all_proposal_areas)
        scores = np.array(all_proposal_scores)

        print(f"\n【尺寸分布 - 像素】(输入图像: {config.img_size}x{config.img_size})")
        print(f"  宽度:  mean={widths.mean():.1f}, std={widths.std():.1f}, "
              f"min={widths.min():.1f}, max={widths.max():.1f}")
        print(f"  高度:  mean={heights.mean():.1f}, std={heights.std():.1f}, "
              f"min={heights.min():.1f}, max={heights.max():.1f}")
        print(f"  面积:  mean={areas.mean():.0f}, std={areas.std():.0f}, "
              f"min={areas.min():.0f}, max={areas.max():.0f}")

        print(f"\n【Objectness分数分布】")
        print(f"  分数:  mean={scores.mean():.3f}, std={scores.std():.3f}, "
              f"min={scores.min():.3f}, max={scores.max():.3f}")

        # 尺寸分桶分析
        print(f"\n【Proposal尺寸分桶】")
        size_buckets = {
            '小 (<64px)': (areas < 64*64).sum(),
            '中 (64-128px)': ((areas >= 64*64) & (areas < 128*128)).sum(),
            '大 (128-256px)': ((areas >= 128*256) & (areas < 256*256)).sum(),
            '超大 (>256px)': (areas >= 256*256).sum(),
        }
        total = len(areas)
        for bucket_name, count in size_buckets.items():
            pct = count / total * 100
            print(f"  {bucket_name}: {count} ({pct:.1f}%)")

    if len(all_gt_areas) > 0:
        gt_widths = np.array(all_gt_widths)
        gt_heights = np.array(all_gt_heights)
        gt_areas = np.array(all_gt_areas)

        print(f"\n{'='*50}")
        print("【GT Ground Truth框统计】")
        print(f"{'='*50}")
        print(f"总GT框数: {len(gt_areas)}")
        print(f"\n【GT尺寸分布 - 像素】")
        print(f"  宽度:  mean={gt_widths.mean():.1f}, std={gt_widths.std():.1f}, "
              f"min={gt_widths.min():.1f}, max={gt_widths.max():.1f}")
        print(f"  高度:  mean={gt_heights.mean():.1f}, std={gt_heights.std():.1f}, "
              f"min={gt_heights.min():.1f}, max={gt_heights.max():.1f}")
        print(f"  面积:  mean={gt_areas.mean():.0f}, std={gt_areas.std():.0f}, "
              f"min={gt_areas.min():.0f}, max={gt_areas.max():.0f}")

        # Proposal vs GT对比
        print(f"\n{'='*50}")
        print("【关键发现: Proposal vs GT 尺寸对比】")
        print(f"{'='*50}")
        if len(all_proposal_areas) > 0 and len(gt_areas) > 0:
            prop_mean = areas.mean()
            gt_mean = gt_areas.mean()
            ratio = prop_mean / gt_mean if gt_mean > 0 else float('inf')
            print(f"  Proposal平均面积: {prop_mean:.0f} px²")
            print(f"  GT平均面积:       {gt_mean:.0f} px²")
            print(f"  倍数关系:         Proposal是GT的 {ratio:.1f} 倍")
            if ratio > 10:
                print(f"  ⚠️  [严重] Proposal比GT大{ratio:.0f}倍! 这就是定位不准的根因!")
            elif ratio > 5:
                print(f"  ⚠️  [警告] Proposal偏大{ratio:.1f}倍")


def diagnose_roi_head(model, dataset, num_samples=5):
    """诊断ROI Head的bbox回归效果"""

    print("\n" + "="*70)
    print("【诊断2】ROI Head Bbox回归效果")
    print("="*70)

    final_detections = []

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        image_tensor = sample[0].unsqueeze(0).to(device)

        with torch.no_grad():
            results = model.inference(image_tensor)

        boxes = results['boxes'][0]      # [N, 4] 最终检测结果
        scores = results['scores'][0]    # [N]

        if boxes.size(0) > 0:
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            widths = (x2 - x1).abs().cpu().numpy()
            heights = (y2 - y1).abs().cpu().numpy()
            areas = widths * heights
            scores_np = scores.cpu().numpy()

            for i in range(len(widths)):
                final_detections.append({
                    'width': widths[i],
                    'height': heights[i],
                    'area': areas[i],
                    'score': scores_np[i]
                })

    if len(final_detections) > 0:
        areas = [d['area'] for d in final_detections]
        widths = [d['width'] for d in final_detections]
        heights = [d['height'] for d in final_detections]
        scores = [d['score'] for d in final_detections]

        print(f"\n【最终检测框统计】({len(final_detections)}个检测)")
        print(f"  宽度:  mean={np.mean(widths):.1f}, std={np.std(widths):.1f}")
        print(f"  高度:  mean={np.mean(heights):.1f}, std={np.std(heights):.1f}")
        print(f"  面积:  mean={np.mean(areas):.0f}, std={np.std(areas):.0f}")
        print(f"  置信度: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")

        # 面积分桶
        print(f"\n【最终检测尺寸分桶】")
        img_area = config.img_size ** 2
        small = sum(1 for a in areas if a < img_area * 0.05)  # <5%图像
        medium = sum(1 for a in areas if img_area * 0.05 <= a < img_area * 0.2)  # 5-20%
        large = sum(1 for a in areas if a >= img_area * 0.2)  # >20%
        total = len(areas)
        print(f"  小框 (<5%图像): {small} ({small/total*100:.1f}%)")
        print(f"  中框 (5-20%图像): {medium} ({medium/total*100:.1f}%)")
        print(f"  大框 (>20%图像): {large} ({large/total*100:.1f}%)")

        if large / total > 0.7:
            print(f"\n⚠️  [严重] {large/total*100:.0f}%的检测框覆盖>20%图像区域!")
            print(f"   → ROI Head的bbox回归几乎失效!")


def diagnose_anchor_config():
    """诊断Anchor配置是否合理"""

    print("\n" + "="*70)
    print("【诊断3】Anchor配置分析")
    print("="*70)

    img_size = config.img_size
    print(f"\n输入图像尺寸: {img_size}x{img_size}")

    for layer, cfg in FPN_CONFIG['anchor_configs'].items():
        stride = cfg['stride']
        scales = cfg['scales']
        ratios = cfg['ratios']

        print(f"\n[{layer}] stride={stride}")
        print(f"  Anchor scales: {scales}")
        print(f"  Anchor ratios: {ratios}")

        # 计算实际anchor尺寸（像素）
        base_size = stride * 4  # 基准anchor尺寸
        for scale in scales:
            for ratio in ratios:
                ws = base_size * scale * np.sqrt(ratio)
                hs = base_size * scale / np.sqrt(ratio)
                area = ws * hs
                pct_of_img = area / (img_size ** 2) * 100
                print(f"    Anchor: {ws:.0f}x{hs:.0f} (面积:{area:.0f}px², 占图像{pct_of_img:.1f}%)")


# ============================================================
# 4. 执行所有诊断
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("开始深度诊断 - 定位精度根因分析")
    print("="*70)

    # 诊断1: RPN Proposal分布
    diagnose_rpn_proposals(model, test_dataset, num_samples=10, max_proposals_per_layer=1000)

    # 诊断2: ROI Head回归效果
    diagnose_roi_head(model, test_dataset, num_samples=10)

    # 诊断3: Anchor配置
    diagnose_anchor_config()

    print("\n" + "="*70)
    print("诊断完成!")
    print("="*70)
