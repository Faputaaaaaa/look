"""
两阶段目标检测器（FPN + RPN + Anchor）

架构：
1. Backbone: ResNet18
2. FPN: P2 + P3 + P4 多尺度特征
3. RPN: 区域提议网络
4. ROI Head: 检测头（分类+回归精修）

实际生效的损失函数（在 train.py 中计算）：
- RPN cls: BCEWithLogitsLoss (train.py:639) — [R35] Focal Loss/加权BCE均未解决spatial bias后回退
- RPN bbox: CIoU Loss (train.py:650)
- ROI cls: CrossEntropyLoss (train.py:949) — [Round19] 替代FocalLoss启用类间竞争（项目硬约束）
- ROI bbox: CIoU Loss (train.py:959, 1224)

注意：早期版本的 FocalLoss 类已删除（[R40]清理），实际未参与训练。
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

# [R40] FocalLoss 类已删除：项目实际未使用（RPN cls用BCEWithLogits，ROI cls用CrossEntropy）
# 早期 [Round 5-18] 实验过Focal Loss，[R35] 后回退BCE，[Round19] ROI改CrossEntropy启用类间竞争。
# 历史经验保留在 project_memory 中以防回退。


class CIoULoss(nn.Module):
    """CIoU Loss用于边界框回归 [Round12-B]。
    
    CIoU = IoU - (ρ²(b,b_gt)/c²) - αv
    其中:
    - ρ: 预测框与目标框中心点的欧氏距离
    - c: 最小闭包对角线长度
    - v = (4/π²)(arctan(w_gt/h_gt) - arctan(w/h))²  长宽比一致性项
    - α = v / ((1-IoU) + v)  权重参数
    
    相比GIoU, CIoU额外考虑了中心点距离和长宽比一致性，
    收敛更快，定位精度更高。
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
        
        # 中心点距离 ρ²
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        
        rho_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # 闭包对角线长度平方 c²
        c_sq = (c_x2 - c_x1).clamp(min=1e-6) ** 2 + (c_y2 - c_y1).clamp(min=1e-6) ** 2
        
        # 长宽比一致性 v
        pred_w = pred[:, 2] - pred[:, 0]
        pred_h = pred[:, 3] - pred[:, 1]
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]
        
        v = (4 / (torch.pi ** 2)) * (
            torch.atan(target_w / target_h.clamp(min=1e-6)) -
            torch.atan(pred_w / pred_h.clamp(min=1e-6))
        ) ** 2
        
        # 权重参数 α
        alpha = v / ((1 - iou) + v).clamp(min=1e-6)
        
        # CIoU
        ciou = iou - (rho_sq / c_sq) - alpha * v
        
        # Loss
        loss = 1 - ciou
        
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
    - P2(stride=4): 尺度[8,12,16,24,32], 比率[0.3,0.5,1.0,2.0,3.0] → 25个锚框/位置
    - P3(stride=8): 尺度[8,12,16,24,32], 比率[0.3,0.5,1.0,2.0,3.0] → 25个锚框/位置
    - P4(stride=16): 尺度[8,12,16,24,32], 比率[0.3,0.5,1.0,2.0,3.0] → 25个锚框/位置
    """
    
    def __init__(self):
        # [修正C] 从FPN_CONFIG读取锚框配置，单一事实来源
        # 如果FPN_CONFIG尚未定义（极端情况），使用内置默认值
        try:
            _cfg = FPN_CONFIG.get('anchor_configs', {})
            self.anchor_configs = {
                'P2': {'stride': _cfg.get('P2', {}).get('stride', 4),
                       'scales': _cfg.get('P2', {}).get('scales', [8,12,16,24,32]),
                       'ratios': _cfg.get('P2', {}).get('ratios', [0.3,0.5,1.0,2.0,3.0])},
                'P3': {'stride': _cfg.get('P3', {}).get('stride', 8),
                       'scales': _cfg.get('P3', {}).get('scales', [8,12,16,24,32]),
                       'ratios': _cfg.get('P3', {}).get('ratios', [0.3,0.5,1.0,2.0,3.0])},
                'P4': {'stride': _cfg.get('P4', {}).get('stride', 16),
                       'scales': _cfg.get('P4', {}).get('scales', [8,12,16,24,32]),
                       'ratios': _cfg.get('P4', {}).get('ratios', [0.3,0.5,1.0,2.0,3.0])},
            }
        except NameError:
            # FPN_CONFIG未定义时的回退（仅用于测试等边缘场景）
            self.anchor_configs = {
                'P2': {'stride': 4, 'scales': [8,12,16,24,32], 'ratios': [0.3,0.5,1.0,2.0,3.0]},
                'P3': {'stride': 8, 'scales': [8,12,16,24,32], 'ratios': [0.3,0.5,1.0,2.0,3.0]},
                'P4': {'stride': 16,'scales': [8,12,16,24,32], 'ratios': [0.3,0.5,1.0,2.0,3.0]},
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

        # [Round15] 动态锚框数: 取所有层中最大值
        # P2=25, P3=40, P4=63 → RPN head输出63通道(每层reshape时取前N个)
        self.max_anchors_per_loc = max(
            len(cfg['scales']) * len(cfg['ratios'])
            for cfg in FPN_CONFIG['anchor_configs'].values()
        )  # = 63 (from P4: 9ratios × 7scales)
        self.cls_conv = nn.Conv2d(in_channels, self.max_anchors_per_loc, 1)
        self.bbox_conv = nn.Conv2d(in_channels, self.max_anchors_per_loc * 4, 1)

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
            
            # 分类 [B, 25, H, W] — 输出原始logits（不在此处sigmoid）
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
    """检测头（分类-回归双分支，共享bbox回归）。

    [Round21] 彻底修复ROI Head特征坍缩：
    根因链：Backbone特征相似度高(0.91-0.99) → R20的L2归一化压缩差异 → ReLU归零47% → 坍缩

    修复策略：
    1. 去掉L2归一化（压缩微小差异的元凶）
    2. 用GroupNorm替代（按组归一化，不独立处理每个样本）
    3. 降低Dropout（保留更多特征信息）
    4. 增加残差连接（保护输入信息流）
    5. 恢复隐藏维度到512（增强表达能力）
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 4):
        super().__init__()

        self.num_classes = num_classes
        hidden_dim = 512  # [R21] 恢复到512，增强表达能力

        # 输入维度
        self.in_flat_dim = in_channels * 7 * 7  # 12544

        # ============ 共享底层特征 [R21: GroupNorm + 残差 + 适度Dropout] ============
        self.shared_fc = nn.Sequential(
            nn.Linear(self.in_flat_dim, hidden_dim),
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim),  # [R21关键] GN不是per-sample
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # [R21] 从0.5降到0.3，保留更多信息
        )

        # [R40] 残差投影：input(in_flat_dim) → hidden_dim，让 shared + proj 真正生效
        # 之前注释说"维度不匹配所以跳过"，残差连接形同虚设。
        # 补上 1×1 Linear 投影后：shared = shared_fc(x)；out = shared + proj(x)
        # 这样输入信息能绕过 shared_fc 直接流向 cls/bbox 分支，保护梯度流。
        self.residual_proj = nn.Linear(self.in_flat_dim, hidden_dim)

        # ============ 分类分支 ============
        self.cls_branch = nn.Sequential(
            nn.GroupNorm(num_groups=16, num_channels=hidden_dim),  # 输入归一化
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes + 1),  # 0=BG, 1..N=前景
        )

        # ============ 回归分支 共享4维delta ============
        self.bbox_branch = nn.Sequential(
            nn.GroupNorm(num_groups=16, num_channels=hidden_dim),  # 输入归一化
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 4),
        )

        # ============ [R21] 初始化策略 ============
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                std = np.sqrt(2.0 / fan_in)  # [R21] 使用标准Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # [R40] 残差投影初始化为零（残差分支初始为恒等映射，训练初期不破坏shared_fc的输出）
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

        # 分类头BG偏好初始化
        bg_prior = 0.85
        bias_value = -np.log((1 - bg_prior) / bg_prior)
        nn.init.constant_(self.cls_branch[-1].bias, 0.0)
        with torch.no_grad():
            self.cls_branch[-1].bias[0] = bias_value
            self.cls_branch[-1].bias[1:] = 0.0

        # 回归头初始化为接近零的输出
        nn.init.constant_(self.bbox_branch[-1].bias, 0.0)
        nn.init.normal_(self.bbox_branch[-1].weight, mean=0.0, std=0.01)

    def forward(self, roi_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_pred: [N, num_classes+1] 分类logits
            bbox_pred: [N, 4] 共享边界框精修delta
        """
        x = roi_features.flatten(1)

        # [R21关键改动] 不做L2归一化！保留原始幅度信息
        # 直接让GroupNorm在shared_fc内部处理归一化

        shared = self.shared_fc(x)

        # [R40] 真正生效的残差连接：输入投影到 hidden_dim 后加到 shared 上
        # 残差投影权重初始化为0，训练初期 shared 不变，避免破坏 R21 已验证的收敛行为；
        # 后续训练会让 proj 逐步学习到补充信息。
        residual = self.residual_proj(x)
        shared = shared + residual

        cls_pred = self.cls_branch(shared)
        bbox_pred = self.bbox_branch(shared)

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
    - RPN: Focal Loss (α=0.25, γ=2.0) + CIoU Loss
    - ROI: Focal Loss (α=0.5, γ=2.0) + CIoU Loss
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

        # [R40] 损失函数实例已删除：实际损失计算在 train.py 中
        # RPN cls: BCEWithLogitsLoss (train.py:639)
        # RPN bbox: CIoULoss (train.py:650, RPNTrainer.compute_loss)
        # ROI cls: CrossEntropyLoss (train.py:949)
        # ROI bbox: CIoULoss (train.py:959)
    
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

    def _spatial_uniform_sample(self, proposals: torch.Tensor, scores: torch.Tensor,
                                  grid_size: int = 3, per_cell_max: int = 150,
                                  img_w: int = 768, img_h: int = 768) -> Tuple[torch.Tensor, torch.Tensor]:
        """[R23] 空间均匀采样：将proposal按空间位置分成网格，每格独立取top-K。

        解决问题：RPN的objectness score存在系统性空间偏置（如偏向图像顶部），
        导致纯score排序选出的top-K proposal聚集在一个固定区域。
        本方法通过空间均匀采样打破这种偏置，保证最终proposal覆盖全图。

        Args:
            proposals: [N, 4] (x1,y1,x2,y2) 格式的proposal坐标
            scores: [N] 对应的RPN objectness score
            grid_size: 网格大小（默认3×3）
            per_cell_max: 每个网格格子最多保留的proposal数
            img_w, img_h: 图像宽高（用于计算网格边界）

        Returns:
            采样后的 (proposals, scores)
        """
        if proposals.size(0) == 0:
            return proposals, scores

        # 计算每个proposal的中心点
        cx = (proposals[:, 0] + proposals[:, 2]) / 2
        cy = (proposals[:, 1] + proposals[:, 3]) / 2

        # 确定每个proposal属于哪个网格格子
        cell_w = img_w / grid_size
        cell_h = img_h / grid_size

        # 限制在有效范围内
        cell_x_idx = torch.clamp((cx / cell_w).long(), 0, grid_size - 1)
        cell_y_idx = torch.clamp((cy / cell_h).long(), 0, grid_size - 1)
        cell_idx = cell_y_idx * grid_size + cell_x_idx  # 展平的一维索引

        # 每个格子内按score排序取top-K
        sampled_masks = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                ci = gy * grid_size + gx
                mask = (cell_idx == ci)
                if mask.sum() == 0:
                    continue
                cell_scores = scores[mask]
                n_take = min(per_cell_max, mask.sum().item())
                if n_take <= 0:
                    continue
                # 该格子内取score最高的n_take个
                _, local_top = cell_scores.topk(n_take)
                # 转回全局索引
                global_indices = torch.where(mask)[0][local_top]
                cell_mask = torch.zeros(proposals.size(0), dtype=torch.bool, device=proposals.device)
                cell_mask[global_indices] = True
                sampled_masks.append(cell_mask)

        if not sampled_masks:
            # fallback：如果所有格子都为空（不应该发生），返回原始
            return proposals, scores

        # 合并所有格子的采样结果
        combined_mask = sampled_masks[0]
        for m in sampled_masks[1:]:
            combined_mask = combined_mask | m

        return proposals[combined_mask], scores[combined_mask]

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

                # [Round15] RPN输出统一max_anchors(63)通道，需按实际anchor数切片（与train/validate一致）
                _lc = FPN_CONFIG['anchor_configs'][layer_name]
                num_anchors_actual = len(_lc['scales']) * len(_lc['ratios'])
                max_a = self.rpn.max_anchors_per_loc  # = 63

                # [R26修复] permute对齐anchor(与train一致)，之前flatten导致通道主序错位
                cls_pred = rpn_cls_preds[i][b].permute(1, 2, 0).contiguous()  # [H, W, max_a]
                cls_pred = cls_pred.reshape(-1, max_a)[:, :num_anchors_actual].reshape(-1)
                # RPN输出原始logits，推理时需要sigmoid转为概率
                cls_pred = torch.sigmoid(cls_pred)

                bbox_pred = rpn_bbox_preds[i][b].permute(1, 2, 0).contiguous()
                bbox_pred = bbox_pred.reshape(-1, max_a, 4)
                # 只取前 num_anchors_actual 组有效预测（其余为padding）
                bbox_pred = bbox_pred[:, :num_anchors_actual, :].reshape(-1, 4)
                
                boxes = self.decode_boxes(anchors, bbox_pred, img_w=W, img_h=H)

                # [R25] 推理时与训练保持一致：空间感知采样 + 分网格NMS
                # 解决RPN空间偏置导致proposal分布不均的问题
                feat_h = fpn_features[i].size(2)
                feat_w = fpn_features[i].size(3)
                
                cls_scores_2d = cls_pred.view(feat_h, feat_w, num_anchors_actual)
                boxes_2d = boxes.view(feat_h, feat_w, num_anchors_actual, 4)
                
                # P3主层用更多anchor，P2/P4用较少的
                k_per_loc_map = {'P2': 1, 'P3': 3, 'P4': 1}
                per_cell_max_map = {'P2': 10, 'P3': 30, 'P4': 10}
                k = k_per_loc_map[layer_name]
                pcm = per_cell_max_map[layer_name]
                
                topk_scores, topk_inds = cls_scores_2d.topk(k, dim=2)
                topk_inds_expanded = topk_inds.unsqueeze(-1).expand(-1, -1, -1, 4)
                topk_boxes = boxes_2d.gather(2, topk_inds_expanded)
                sampled_boxes = topk_boxes.reshape(-1, 4)
                sampled_scores = topk_scores.reshape(-1)
                
                # 分网格独立NMS
                grid_size = 8
                cell_w = W / grid_size
                cell_h = H / grid_size
                cx = (sampled_boxes[:, 0] + sampled_boxes[:, 2]) / 2
                cy = (sampled_boxes[:, 1] + sampled_boxes[:, 3]) / 2
                cell_x = torch.clamp((cx / cell_w).long(), 0, grid_size - 1)
                cell_y = torch.clamp((cy / cell_h).long(), 0, grid_size - 1)
                cell_idx = cell_y * grid_size + cell_x
                
                keep_list = []
                for ci in range(grid_size * grid_size):
                    mask = (cell_idx == ci)
                    if mask.sum() == 0:
                        continue
                    cell_boxes = sampled_boxes[mask]
                    cell_scores = sampled_scores[mask]
                    cell_keep = nms(cell_boxes, cell_scores, 0.7)
                    cell_keep = cell_keep[:pcm]
                    global_indices = torch.where(mask)[0][cell_keep]
                    keep_list.append(global_indices)
                
                if len(keep_list) > 0:
                    keep = torch.cat(keep_list)
                    proposals.append(sampled_boxes[keep])
                    proposal_scores.append(sampled_scores[keep])
            
            if len(proposals) > 0:
                proposals = torch.cat(proposals, dim=0)
                proposal_scores = torch.cat(proposal_scores, dim=0)
                # [R25] 不做全局NMS，与训练保持一致
                # 每层分网格NMS已去重，总共约3200个proposal
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
            # [Round19] bbox_deltas现在是共享4维[N,4]，不再per-class

            # Softmax多分类
            cls_probs = torch.softmax(cls_logits, dim=1)  # [N, 6]
            scores, cls_indices = cls_probs.max(dim=1)   # scores: [N], cls_indices: 0-5

            # [Round19] 直接使用共享4维delta，不需要per-class选择
            # bbox_deltas: [N, 4] (dx, dy, dw, dh) — 所有类共用

            # 应用ROI Head的bbox回归精修（只对非BG的proposal才做精修）
            refined_boxes = self.decode_boxes(
                proposals[roi_orig_indices], bbox_deltas, img_w=W, img_h=H
            )

            # ============ 多级过滤策略 ============
            # Step 1: 过滤掉背景类（cls_index=0即BG），只保留前景
            fg_mask = cls_indices > 0  # 前景cls_index: 1-5
            if fg_mask.sum() == 0:
                all_boxes.append(torch.zeros(0, 4, device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                continue

            refined_boxes = refined_boxes[fg_mask]
            scores = scores[fg_mask]
            labels = cls_indices[fg_mask]  # cls_index 1→label 1(chongkong)等

            # Step 2: 过滤异常框（宽或高为0或负数的框
            box_w = refined_boxes[:, 2] - refined_boxes[:, 0]
            box_h = refined_boxes[:, 3] - refined_boxes[:, 1]
            valid_box_mask = (box_w > 2) & (box_h > 2)
            if valid_box_mask.sum() == 0:
                all_boxes.append(torch.zeros(0, 4, device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            refined_boxes = refined_boxes[valid_box_mask]
            scores = scores[valid_box_mask]
            labels = labels[valid_box_mask]

            # Step 3: score阈值过滤
            # [R35] 降低score_thresh从0.3到0.1
            # RPN score最高仅0.49，ROI softmax概率可能也偏低
            # 0.1让足够多候选通过，NMS会过滤重复
            valid_mask = scores > 0.1

            final_boxes = refined_boxes[valid_mask]
            final_scores = scores[valid_mask]
            final_labels = labels[valid_mask]

            # Step 4: NMS去重 + 最多max_detections个
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

    # [R35] anchor配置匹配SAHI切片后的缺陷尺寸
    # 数据分析：1024切片下 missing≈104px, dirty≈304px
    # 旧配置[P2:8-32,P3:16-64,P4:32-128]最大128px，无法覆盖dirty(304px)
    # 新配置覆盖32-512px范围，确保missing和dirty都有匹配anchor
    'anchor_configs': {
        'P2': {'stride': 4, 'scales': [32, 48, 64, 96, 128],    'ratios': [0.5, 1.0, 2.0, 3.0]},   # 20 anchors, 覆盖missing(104px)
        'P3': {'stride': 8, 'scales': [64, 96, 128, 192, 256],  'ratios': [0.5, 1.0, 2.0, 3.0]},   # 20 anchors, 过渡区
        'P4': {'stride': 16,'scales': [128, 192, 256, 384, 512], 'ratios': [0.5, 1.0, 2.0, 3.0]},  # 20 anchors, 覆盖dirty(304px)
    },
    
    'rpn_config': {
        # [R32] 降低pos_thresh从0.5→0.3，增加RPN正样本数
        # 原因：R31诊断显示RPN学到固定spatial bias（10张图Top5 score完全相同），
        #       根因是正样本极度稀疏（1-40个/百万级anchors），梯度信号被淹没。
        #       降低pos_thresh到0.3，正样本数预计增加到几百个，让RPN能学到位置信息。
        'pos_iou_thresh': 0.3,
        'neg_iou_thresh': 0.1,
        'pos_neg_ratio': 3,
        # [问题2] 背景窗口负样本上限：每背景窗口最多补 64 个负 anchor（正负比 1:3 不变）。
        'max_neg_per_bg_window': 64,
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
    # [Round16] 提高RPN质量门槛，避免低质量proposal污染ROI Head
    'rpn_pre_nms_top_n': 2000,     # 适度候选数
    'rpn_post_nms_top_n': 1000,   # 减少但提高质量(避免R15的2000个噪声proposal)
    'rpn_nms_thresh': 0.7,         # 标准NMS
    'rpn_score_thresh': 0.01,      # [关键] 提高阈值过滤低分anchor(R15用0.003太低)
    # [R35] 误检优化：模型过度自信(score全0.99+)，阈值无法区分
    # 改为：MAX_DET=5(原图每张仅1缺陷) + NMS=0.30(更严格去重)
    'score_thresh': 0.10,          # 降低阈值让候选通过，靠NMS+MAX_DET控制数量
    'nms_thresh': 0.30,            # 收紧NMS，合并更多重叠框
    'max_detections': 5,           # 每张切片最多5个检测（原图每张仅1个缺陷）
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
