"""
两阶段目标检测器（FPN + RPN + Anchor）

架构：
1. Backbone: ResNet18
2. FPN: P2 + P3 + P4 多尺度特征
3. RPN: 区域提议网络
4. ROI Head: 检测头（分类+回归精修）

损失函数：
- Focal Loss (分类)
- GIoU Loss (回归)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
from typing import List, Tuple, Dict, Optional
from config import config
from utils import logger


# ============================================================
# 损失函数
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss用于分类。
    
    FL(p) = -α(1-p)^γ * log(p)    (正样本)
    FL(p) = -(1-α)p^γ * log(1-p)  (负样本)
    
    Args:
        alpha: 正样本权重
        gamma: 聚焦参数
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [N,] 预测概率 (0-1)
            target: [N,] 目标标签 (0或1)
        
        Returns:
            loss: 标量损失
        """
        pred = pred.clamp(1e-6, 1 - 1e-6)
        
        # 正样本损失
        pos_mask = target == 1
        pos_loss = -self.alpha * (1 - pred[pos_mask]) ** self.gamma * torch.log(pred[pos_mask])
        
        # 负样本损失
        neg_mask = target == 0
        neg_loss = -(1 - self.alpha) * pred[neg_mask] ** self.gamma * torch.log(1 - pred[neg_mask])
        
        loss = pos_loss.sum() + neg_loss.sum()
        num_samples = pos_mask.sum() + neg_mask.sum()
        
        return loss / max(num_samples, 1)


class GIoULoss(nn.Module):
    """GIoU Loss用于边界框回归。
    
    GIoU = IoU - (C - (A∪B)) / C
    Loss = 1 - GIoU
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [N, 4] 预测框 (x1, y1, x2, y2)
            target: [N, 4] 目标框 (x1, y1, x2, y2)
        
        Returns:
            loss: 标量损失
        """
        if pred.size(0) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 计算交集
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # 计算并集
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / union_area.clamp(min=1e-6)
        
        # 最小闭包区域C
        c_x1 = torch.min(pred[:, 0], target[:, 0])
        c_y1 = torch.min(pred[:, 1], target[:, 1])
        c_x2 = torch.max(pred[:, 2], target[:, 2])
        c_y2 = torch.max(pred[:, 3], target[:, 3])
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
        
        # GIoU
        giou = iou - (c_area - union_area) / c_area.clamp(min=1e-6)
        
        # Loss
        loss = 1 - giou
        
        return loss.mean()


# ============================================================
# FPN模块
# ============================================================

class FPN(nn.Module):
    """特征金字塔网络 (Feature Pyramid Network)。
    
    输出 P2 + P3 + P4 三层特征
    - P2: stride=4 (高分辨率，适合小目标)
    - P3: stride=8 (中等分辨率)
    - P4: stride=16 (低分辨率，适合大目标)
    """
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        
        # 横向连接（1x1卷积降维）
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])
        
        # 平滑卷积（3x3）
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
        
        self.out_channels = out_channels
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [C2, C3, C4] 从ResNet提取的特征
        
        Returns:
            [P2, P3, P4] FPN特征
        """
        # 横向连接
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
        
        # 平滑卷积
        fpn_features = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        
        return fpn_features


# ============================================================
# Anchor生成器
# ============================================================

class AnchorGenerator:
    """锚框生成器（K-means优化配置）。

    配置:
    - P2(stride=4): 尺度[16,24,32,48,64], 比率[0.5,1.0,2.0] → 15个锚框/位置
    - P3(stride=8): 尺度[32,48,64,96,128], 比率[0.5,1.0,2.0] → 15个锚框/位置
    - P4(stride=16): 尺度[64,128,192,256,384], 比率[0.5,1.0,2.0] → 15个锚框/位置
    """
    
    def __init__(self):
        self.anchor_configs = {
            'P2': {
                'stride': 4,
                'scales': [16, 24, 32, 48, 64],  # K-means优化: 覆盖11-90px，匹配小目标如chongkong(41px)
                'ratios': [0.5, 1.0, 2.0]
            },
            'P3': {
                'stride': 8,
                'scales': [32, 48, 64, 96, 128],  # K-means优化: 覆盖23-181px
                'ratios': [0.5, 1.0, 2.0]
            },
            'P4': {
                'stride': 16,
                'scales': [64, 128, 192, 256, 384],  # K-means优化: 覆盖45-543px
                'ratios': [0.5, 1.0, 2.0]
            }
        }
    
    def generate_anchors_for_layer(
        self, 
        feature_size: Tuple[int, int],
        stride: int,
        scales: List[float],
        ratios: List[float]
    ) -> torch.Tensor:
        """为一层特征图生成所有锚框。
        
        Args:
            feature_size: (H, W) 特征图尺寸
            stride: 下采样步长
            scales: 锚框尺度列表
            ratios: 宽高比列表
        
        Returns:
            anchors: [N, 4] 所有锚框 (x1, y1, x2, y2)
        """
        H, W = feature_size
        
        anchors = []
        for scale in scales:
            for ratio in ratios:
                w = scale * np.sqrt(ratio)
                h = scale / np.sqrt(ratio)
                anchors.append([-w/2, -h/2, w/2, h/2])
        anchors = torch.tensor(anchors, dtype=torch.float32)
        num_anchors_per_loc = anchors.size(0)
        
        shift_x = (torch.arange(0, W, dtype=torch.float32) + 0.5) * stride
        shift_y = (torch.arange(0, H, dtype=torch.float32) + 0.5) * stride
        
        shift_xx, shift_yy = torch.meshgrid(shift_x, shift_y, indexing='xy')
        shifts = torch.stack([
            shift_xx.flatten(),
            shift_yy.flatten(),
            shift_xx.flatten(),
            shift_yy.flatten()
        ], dim=1)
        
        all_anchors = anchors.view(1, num_anchors_per_loc, 4) + shifts.view(-1, 1, 4)
        all_anchors = all_anchors.view(-1, 4)
        
        return all_anchors
    
    def generate_all_anchors(
        self, 
        feature_sizes: List[Tuple[int, int]]
    ) -> Dict[str, torch.Tensor]:
        """生成所有层的锚框。
        
        Args:
            feature_sizes: [(H2,W2), (H3,W3), (H4,W4)]
        
        Returns:
            {'P2': anchors, 'P3': anchors, 'P4': anchors}
        """
        all_anchors = {}
        layer_names = ['P2', 'P3', 'P4']
        
        for name, feat_size in zip(layer_names, feature_sizes):
            cfg = self.anchor_configs[name]
            anchors = self.generate_anchors_for_layer(
                feat_size,
                cfg['stride'],
                cfg['scales'],
                cfg['ratios']
            )
            all_anchors[name] = anchors
        
        return all_anchors


# ============================================================
# RPN模块
# ============================================================

class RPN(nn.Module):
    """区域提议网络。
    
    输入: FPN特征
    输出: 候选框（每层200个，汇总300个）
    
    正负样本匹配:
    - 正样本: IoU >= 0.5
    - 负样本: IoU < 0.3
    - 正负比例: 1:3
    """
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        
        # 3x3卷积
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        self.cls_conv = nn.Conv2d(in_channels, 15, 1)  # 5尺度×3宽高比=15个锚框
        
        self.bbox_conv = nn.Conv2d(in_channels, 15 * 4, 1)
        
        # 锚框生成器
        self.anchor_generator = AnchorGenerator()
    
    def forward(
        self, 
        fpn_features: List[torch.Tensor],
        img_size: Tuple[int, int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            fpn_features: [P2, P3, P4]
            img_size: (H, W) 原图尺寸
        
        Returns:
            cls_preds: 每层的分类预测
            bbox_preds: 每层的边界框预测
        """
        cls_preds = []
        bbox_preds = []
        
        for feat in fpn_features:
            x = F.relu(self.conv(feat))
            
            # 分类 [B, 15, H, W] — 输出原始logits（不在此处sigmoid）
            # 训练时直接送入BCEWithLogitsLoss，推理时再手动sigmoid
            cls_pred = self.cls_conv(x)
            cls_preds.append(cls_pred)
            
            # 回归 [B, 60, H, W]
            bbox_pred = self.bbox_conv(x)
            bbox_preds.append(bbox_pred)
        
        return cls_preds, bbox_preds
    
    def get_anchors(
        self, 
        feature_sizes: List[Tuple[int, int]],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """获取所有锚框。"""
        anchors = self.anchor_generator.generate_all_anchors(feature_sizes)
        return {k: v.to(device) for k, v in anchors.items()}


# ============================================================
# ROI Head（检测头）
# ============================================================

class ROIHead(nn.Module):
    """检测头。
    
    对RPN输出的候选框进行分类和边界框精修。
    """
    
    def __init__(self, in_channels: int = 256, num_classes: int = 4):
        super().__init__()
        
        self.num_classes = num_classes
        
        # ROI池化后的特征处理
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.dropout1 = nn.Dropout(0.3)  # 添加dropout防止过拟合
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.3)  # 添加dropout防止过拟合
        
        # 分类头（含背景类：5前景类 + 1背景类 = 6维）
        # 标准Faster R-CNN设计：softmax后类间竞争，BG提供"退出选项"
        self.cls_head = nn.Linear(1024, num_classes + 1)

        # 回归头（共享bbox回归，所有类别共用一组delta）
        # 共享回归避免per-class delta选错导致的框膨胀问题
        self.bbox_head = nn.Linear(1024, 16)

    def forward(self, roi_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            roi_features: [N, C, 7, 7] ROI池化后的特征

        Returns:
            cls_pred: [N, num_classes+1] 分类预测（索引0=背景, 1-5=前景）
            bbox_pred: [N, 16] 边界框精修（共享回归，4组delta各4维）
        """
        x = roi_features.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # 应用dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # 应用dropout
        
        cls_pred = self.cls_head(x)
        bbox_pred = self.bbox_head(x)
        
        return cls_pred, bbox_pred


# ============================================================
# 完整的两阶段检测器
# ============================================================

class TwoStageDetector(nn.Module):
    """两阶段目标检测器。
    
    架构:
    1. Backbone: ResNet18
    2. FPN: P2 + P3 + P4
    3. RPN: 区域提议
    4. ROI Head: 分类+回归精修
    
    损失函数:
    - RPN: Focal Loss (α=0.25, γ=2.0) + GIoU Loss
    - ROI: Focal Loss (α=0.5, γ=2.0) + GIoU Loss
    """
    
    def __init__(self, num_classes: int = None):
        super().__init__()
        
        self.num_classes = num_classes or len(config.class_names)
        
        # Backbone: ResNet18
        resnet = resnet18(weights=None)
        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        # FPN
        self.fpn = FPN([64, 128, 256], out_channels=256)
        
        # RPN
        self.rpn = RPN(in_channels=256)
        
        # ROI Head
        self.roi_head = ROIHead(in_channels=256, num_classes=self.num_classes)
        
        # 损失函数
        self.rpn_cls_loss = FocalLoss(alpha=0.25, gamma=2.0)  # RPN: 标准参数
        self.rpn_bbox_loss = GIoULoss()
        self.roi_cls_loss = FocalLoss(alpha=0.5, gamma=2.0)   # ROI: 调整参数
        self.roi_bbox_loss = GIoULoss()
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """提取ResNet多层特征。"""
        c2 = self.layer1(x)   # [B, 64, H/4, W/4]
        c3 = self.layer2(c2)  # [B, 128, H/8, W/8]
        c4 = self.layer3(c3)  # [B, 256, H/16, W/16]
        
        return [c2, c3, c4]
    
    def forward(self, x: torch.Tensor) -> Dict:
        """
        Args:
            x: [B, 3, H, W] 输入图像
        
        Returns:
            dict: 包含所有预测结果
        """
        # Backbone
        features = self.extract_features(x)
        
        # FPN
        fpn_features = self.fpn(features)  # [P2, P3, P4]
        
        # RPN
        rpn_cls_preds, rpn_bbox_preds = self.rpn(fpn_features, x.shape[-2:])
        
        return {
            'fpn_features': fpn_features,
            'rpn_cls_preds': rpn_cls_preds,
            'rpn_bbox_preds': rpn_bbox_preds
        }
    
    def load_pretrained_weights(self) -> None:
        """加载预训练权重。"""
        try:
            logger.info("加载ImageNet预训练权重...")
            pretrained = resnet18(weights="IMAGENET1K_V1")
            
            # 加载layer1权重
            self.layer1[0].load_state_dict(pretrained.conv1.state_dict())
            self.layer1[1].load_state_dict(pretrained.bn1.state_dict())
            self.layer1[4].load_state_dict(pretrained.layer1.state_dict())
            
            # 加载layer2, layer3权重
            self.layer2.load_state_dict(pretrained.layer2.state_dict())
            self.layer3.load_state_dict(pretrained.layer3.state_dict())
            
            logger.info("预训练权重加载成功！")
            
            # 冻结前两层
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            
            logger.info("冻结layer1和layer2参数")
            
        except Exception as e:
            logger.error(f"加载预训练权重失败: {e}")
    
    def decode_boxes(self, anchors: torch.Tensor, deltas: torch.Tensor,
                     img_w: int = None, img_h: int = None) -> torch.Tensor:
        """将偏移量解码为实际坐标。
        
        Args:
            anchors: [N, 4] 锚框 (x1, y1, x2, y2)
            deltas: [N, 4] 偏移量 (dx, dy, dw, dh)
            img_w, img_h: 可选，裁剪框到图像范围
        
        Returns:
            boxes: [N, 4] 解码后的框 (x1, y1, x2, y2)
        """
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_cx = anchors[:, 0] + 0.5 * anchor_w
        anchor_cy = anchors[:, 1] + 0.5 * anchor_h
        
        dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
        
        # 恢复原始exp clamp范围（分类准确后不需要过度限制）
        pred_cx = dx * anchor_w + anchor_cx
        pred_cy = dy * anchor_h + anchor_cy
        pred_w = torch.exp(dw.clamp(min=-4.0, max=3.0)) * anchor_w
        pred_h = torch.exp(dh.clamp(min=-4.0, max=3.0)) * anchor_h
        
        x1 = pred_cx - 0.5 * pred_w
        y1 = pred_cy - 0.5 * pred_h
        x2 = pred_cx + 0.5 * pred_w
        y2 = pred_cy + 0.5 * pred_h

        # 裁剪到图像边界（防止框飞出图像外）
        # 注意：先clamp再stack，避免对计算图内tensor做inplace操作
        if img_w is not None and img_h is not None:
            x1 = x1.clamp(min=0, max=img_w)
            y1 = y1.clamp(min=0, max=img_h)
            x2 = x2.clamp(min=0, max=img_w)
            y2 = y2.clamp(min=0, max=img_h)

        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        
        return boxes
    
    def inference(self, x: torch.Tensor) -> Dict:
        """推理流程。
        
        Args:
            x: [B, 3, H, W] 输入图像
        
        Returns:
            dict: {
                'boxes': [B, N, 4] 检测框,
                'scores': [B, N] 置信度,
                'labels': [B, N] 类别标签
            }
        """
        from torchvision.ops import nms, roi_align
        
        B, _, H, W = x.shape
        device = x.device
        
        features = self.extract_features(x)
        fpn_features = self.fpn(features)
        rpn_cls_preds, rpn_bbox_preds = self.rpn(fpn_features, (H, W))
        
        feature_sizes = [(f.size(2), f.size(3)) for f in fpn_features]
        anchors_dict = self.rpn.get_anchors(feature_sizes, device)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for b in range(B):
            proposals = []
            proposal_scores = []
            
            for i, layer_name in enumerate(['P2', 'P3', 'P4']):
                anchors = anchors_dict[layer_name]
                
                cls_pred = rpn_cls_preds[i][b].flatten()
                # RPN输出原始logits，推理时需要sigmoid转为概率
                cls_pred = torch.sigmoid(cls_pred)
                bbox_pred = rpn_bbox_preds[i][b].permute(1, 2, 0).contiguous()
                bbox_pred = bbox_pred.reshape(-1, 15, 4)  # 5尺度×3比例=15个锚框
                bbox_pred = bbox_pred.reshape(-1, 4)
                
                boxes = self.decode_boxes(anchors, bbox_pred, img_w=W, img_h=H)

                # 【关键修复】RPN objectness score 阈值过滤
                # 未训练/随机权重的模型sigmoid输出约0.5-0.7，必须过滤掉低分anchor
                # 否则所有top-K anchor都变成proposal → 网格状假检测
                score_thresh = INFERENCE_CONFIG['rpn_score_thresh']
                valid_mask = cls_pred > score_thresh
                if valid_mask.sum() == 0:
                    continue  # 该层无合格proposal，跳过

                cls_pred = cls_pred[valid_mask]
                boxes = boxes[valid_mask]

                top_n = min(INFERENCE_CONFIG['rpn_pre_nms_top_n'], cls_pred.size(0))
                top_scores, top_inds = cls_pred.topk(top_n)
                top_boxes = boxes[top_inds]
                
                keep = nms(top_boxes, top_scores, INFERENCE_CONFIG['rpn_nms_thresh'])
                
                post_nms_top_n = min(INFERENCE_CONFIG['rpn_post_nms_top_n'], keep.size(0))
                keep = keep[:post_nms_top_n]
                
                proposals.append(top_boxes[keep])
                proposal_scores.append(top_scores[keep])
            
            if len(proposals) > 0:
                proposals = torch.cat(proposals, dim=0)
                proposal_scores = torch.cat(proposal_scores, dim=0)

                # 全局NMS：消除跨层重复proposal（P2/P3/P4可能指向同一区域）
                if proposals.size(0) > 0:
                    keep = nms(proposals, proposal_scores, 0.7)
                    # 保留最多600个跨层去重后的proposal
                    keep = keep[:min(600, keep.size(0))]
                    proposals = proposals[keep]
                    proposal_scores = proposal_scores[keep]
            else:
                all_boxes.append(torch.zeros(0, 4, device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            
            if proposals.size(0) == 0:
                all_boxes.append(torch.zeros(0, 4, device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            
            roi_indices = torch.full((proposals.size(0), 1), b, dtype=torch.float32, device=device)
            roi_boxes = torch.cat([roi_indices, proposals], dim=1)
            
            # 根据proposal大小选择FPN层（修复：之前只使用P3）
            box_sizes = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
            sqrt_sizes = torch.sqrt(box_sizes)
            
            p2_mask = sqrt_sizes < 64
            p3_mask = (sqrt_sizes >= 64) & (sqrt_sizes < 128)
            p4_mask = sqrt_sizes >= 128
            
            all_roi_features = []
            all_indices = []
            all_strides = []
            
            if p2_mask.sum() > 0:
                p2_boxes = roi_boxes[p2_mask]
                p2_feat = fpn_features[0]
                p2_roi = roi_align(p2_feat, p2_boxes, output_size=(7, 7), spatial_scale=1.0/4.0)
                all_roi_features.append(p2_roi)
                all_indices.append(torch.where(p2_mask)[0])
                all_strides.append(torch.full((p2_boxes.size(0),), 4, device=device))
            
            if p3_mask.sum() > 0:
                p3_boxes = roi_boxes[p3_mask]
                p3_feat = fpn_features[1]
                p3_roi = roi_align(p3_feat, p3_boxes, output_size=(7, 7), spatial_scale=1.0/8.0)
                all_roi_features.append(p3_roi)
                all_indices.append(torch.where(p3_mask)[0])
                all_strides.append(torch.full((p3_boxes.size(0),), 8, device=device))
            
            if p4_mask.sum() > 0:
                p4_boxes = roi_boxes[p4_mask]
                p4_feat = fpn_features[2]
                p4_roi = roi_align(p4_feat, p4_boxes, output_size=(7, 7), spatial_scale=1.0/16.0)
                all_roi_features.append(p4_roi)
                all_indices.append(torch.where(p4_mask)[0])
                all_strides.append(torch.full((p4_boxes.size(0),), 16, device=device))
            
            if len(all_roi_features) == 0:
                all_boxes.append(torch.zeros(0, 4, device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            
            roi_features = torch.cat(all_roi_features, dim=0)
            roi_orig_indices = torch.cat(all_indices, dim=0)
            
            cls_logits, bbox_deltas = self.roi_head(roi_features)

            # Softmax多分类（标准Faster R-CNN：含背景类，类间竞争）
            # cls_logits: [N, num_classes+1] = [N, 6]，索引0=BG, 1-5=前景
            cls_probs = torch.softmax(cls_logits, dim=1)
            scores, cls_indices = cls_probs.max(dim=1)              # cls_indices: 0-5

            # 共享bbox回归（所有类别共用前4个delta值）
            selected_deltas = bbox_deltas[:, :4]

            # 应用ROI Head的bbox回归精修
            refined_boxes = self.decode_boxes(proposals[roi_orig_indices], selected_deltas, img_w=W, img_h=H)

            # 过滤掉背景类预测（cls_index=0即BG），只保留前景
            fg_mask = cls_indices > 0  # 前景cls_index: 1-5
            if fg_mask.sum() == 0:
                all_boxes.append(torch.zeros(0, 4, device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                continue

            refined_boxes = refined_boxes[fg_mask]
            scores = scores[fg_mask]
            # cls_index 1→label 1(chongkong), 2→2(shuiban), ..., 5→5(hanfeng)
            labels = cls_indices[fg_mask]  # 直接用cls_index作为label (1-5)

            valid_mask = scores > INFERENCE_CONFIG['score_thresh']
            
            final_boxes = refined_boxes[valid_mask]
            final_scores = scores[valid_mask]
            final_labels = labels[valid_mask]
            
            keep = nms(final_boxes, final_scores, INFERENCE_CONFIG['nms_thresh'])
            keep = keep[:INFERENCE_CONFIG['max_detections']]
            
            all_boxes.append(final_boxes[keep])
            all_scores.append(final_scores[keep])
            all_labels.append(final_labels[keep])
        
        return {
            'boxes': all_boxes,
            'scores': all_scores,
            'labels': all_labels
        }


# ============================================================
# 配置参数
# ============================================================

FPN_CONFIG = {
    'fpn_layers': ['P2', 'P3', 'P4'],
    'fpn_channels': 256,
    
    'anchor_configs': {
        'P2': {'stride': 4, 'scales': [16, 24, 32, 48, 64], 'ratios': [0.5, 1.0, 2.0]},
        'P3': {'stride': 8, 'scales': [32, 48, 64, 96, 128], 'ratios': [0.5, 1.0, 2.0]},
        'P4': {'stride': 16, 'scales': [64, 128, 192, 256, 384], 'ratios': [0.5, 1.0, 2.0]},
    },
    
    'rpn_config': {
        'pos_iou_thresh': 0.5,
        'neg_iou_thresh': 0.3,
        'pos_neg_ratio': 3,
        'nms_thresh': 0.5,
        'pre_nms_top_n': 200,
        'post_nms_top_n': 300,
    },
    
    'loss_config': {
        'rpn_focal_alpha': 0.25,
        'rpn_focal_gamma': 2.0,
        'roi_focal_alpha': 0.5,
        'roi_focal_gamma': 2.0,
    }
}

INFERENCE_CONFIG = {
    'rpn_pre_nms_top_n': 600,     # 每层NMS前取top600（标准Faster R-CNN值）
    'rpn_post_nms_top_n': 300,    # 每层NMS后保留300（3层共~900个proposal，合理范围）
    'rpn_nms_thresh': 0.6,        # RPN NMS阈值
    'rpn_score_thresh': 0.25,     # RPN objectness门槛 (0.3→0.25: 降低以获取更多proposal)
    'score_thresh': 0.25,         # ROI Head分类置信度阈值 (0.30→0.25: 快速验证是否有其他类别)
    'nms_thresh': 0.5,            # 最终检测框NMS阈值 (0.45→0.5: 更严格去重)
    'max_detections': 20,         # 每张图最大检测数 (15→20: 允许更多检测)
}


if __name__ == "__main__":
    print("=" * 60)
    print("两阶段检测器测试")
    print("=" * 60)
    
    model = TwoStageDetector()  # 自动从config获取num_classes
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"类别数: {model.num_classes}")
    
    x = torch.randn(2, 3, 512, 512)
    outputs = model(x)
    
    print(f"\n输入形状: {x.shape}")
    print(f"FPN特征:")
    for i, feat in enumerate(outputs['fpn_features']):
        print(f"  P{i+2}: {feat.shape}")
    
    print(f"\nRPN输出:")
    for i, (cls, bbox) in enumerate(zip(outputs['rpn_cls_preds'], outputs['rpn_bbox_preds'])):
        print(f"  P{i+2} 分类: {cls.shape}, 回归: {bbox.shape}")
    
    print("\n配置参数:")
    for key, value in FPN_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
