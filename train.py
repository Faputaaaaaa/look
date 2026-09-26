"""
两阶段目标检测器训练脚本

整合所有商讨结论：
1. cuDNN确定性随机种子
2. 数据增强（翻转+亮度+对比度±10%）
3. ImageNet标准化
4. 两阶段训练（冻结→解冻）
5. 分阶段学习率（余弦衰减）
6. FPN+RPN架构
7. Focal Loss + CIoU Loss
8. 早停机制
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path

from config import config
from model_fpn import (
    TwoStageDetector,
    CIoULoss,
    FPN_CONFIG,  # [Round15] 导入FPN配置(用于动态anchor数)
    INFERENCE_CONFIG,
    extract_rpn_proposals,       # [P0.5] 统一 proposal 提取
    assign_levels_and_roi_align,  # [P0.5] 统一 FPN 层分配 + roi_align
)
from dataset import DefectDataset, create_dataloader
from utils import logger
# from evaluate import draw_loss_curves  # 已移除：evaluate.py不存在，曲线绘制改用tracker.save_curves()


# ============================================================
# 两阶段训练配置
# ============================================================

TRAIN_CONFIG = {
    'stage1_epochs': 8,       # [R22] 从20降到8（R21在E9就已收敛，长训练导致过拟合）
    'stage2_epochs': 50,      # [R39] 30→50，让bbox回归有更多学习机会
    'early_stop_patience': 15,# [R39] 10→15，配合更多epoch
    'stage1_lr': 3e-4,
    'stage2_lr': 3e-5,        # [Round15] 保持微调学习率
    'weight_decay': 5e-4,    # [R27] 从2e-3降到5e-4，减少正则化以加速RPN学习
    'momentum': 0.9,
    'nesterov': True,
    'cls_loss_gamma': 3.0,    # [Round 5] Round4的强γ(更聚焦难样本)
    'cls_loss_alpha': 'auto',
    'label_smoothing': 0.10,  # [Round 5] 中等平滑度
}


# ============================================================
# 工具函数
# ============================================================

def xywh_to_xyxy(boxes: torch.Tensor, img_size: int) -> torch.Tensor:
    """将(cx, cy, w, h)转换为(x1, y1, x2, y2)。"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    
    return torch.stack([x1, y1, x2, y2], dim=1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组框之间的IoU。
    
    Args:
        boxes1: [N, 4] (x1, y1, x2, y2)
        boxes2: [M, 4] (x1, y1, x2, y2)
    
    Returns:
        ious: [N, M] IoU矩阵
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = area1[:, None] + area2[None, :] - inter_area
    
    return inter_area / (union_area + 1e-6)


def spatial_uniform_sample(proposals: torch.Tensor, scores: torch.Tensor,
                           grid_size: int = 4, per_cell_max: int = 80,
                           img_w: int = 768, img_h: int = 768):
    """[R25] 空间均匀采样：保证proposal覆盖全图，避免空间偏置。

    解决问题：RPN的objectness score存在系统性空间偏置（如偏向图像顶部），
    导致纯score排序选出的top-K proposal聚集在一个固定区域，
    ROI Head无法获得GT所在区域的正样本。

    Args:
        proposals: [N, 4] (x1,y1,x2,y2)
        scores: [N] 对应的score
        grid_size: 网格大小
        per_cell_max: 每个格子最多保留的proposal数
        img_w, img_h: 图像宽高

    Returns:
        采样后的 (proposals, scores)
    """
    if proposals.size(0) == 0:
        return proposals, scores

    cx = (proposals[:, 0] + proposals[:, 2]) / 2
    cy = (proposals[:, 1] + proposals[:, 3]) / 2

    cell_w = img_w / grid_size
    cell_h = img_h / grid_size

    cell_x_idx = torch.clamp((cx / cell_w).long(), 0, grid_size - 1)
    cell_y_idx = torch.clamp((cy / cell_h).long(), 0, grid_size - 1)
    cell_idx = cell_y_idx * grid_size + cell_x_idx

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
            _, local_top = cell_scores.topk(n_take)
            global_indices = torch.where(mask)[0][local_top]
            cell_mask = torch.zeros(proposals.size(0), dtype=torch.bool, device=proposals.device)
            cell_mask[global_indices] = True
            sampled_masks.append(cell_mask)

    if not sampled_masks:
        return proposals, scores

    final_mask = torch.stack(sampled_masks, dim=0).any(dim=0)
    return proposals[final_mask], scores[final_mask]


def encode_boxes(gt_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    """将GT框编码为相对于proposals的偏移量。

    Args:
        gt_boxes: [N, 4] (x1, y1, x2, y2)
        proposals: [N, 4] (x1, y1, x2, y2)

    Returns:
        deltas: [N, 4] (dx, dy, dw, dh)
    """
    proposal_w = proposals[:, 2] - proposals[:, 0]
    proposal_h = proposals[:, 3] - proposals[:, 1]
    proposal_cx = (proposals[:, 0] + proposals[:, 2]) / 2
    proposal_cy = (proposals[:, 1] + proposals[:, 3]) / 2

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

    dx = (gt_cx - proposal_cx) / proposal_w.clamp(min=1e-4)
    dy = (gt_cy - proposal_cy) / proposal_h.clamp(min=1e-4)
    dw = torch.log(gt_w / proposal_w.clamp(min=1e-4))
    dh = torch.log(gt_h / proposal_h.clamp(min=1e-4))

    return torch.stack([dx, dy, dw, dh], dim=1)


def clip_grad_norm_grouped(model: TwoStageDetector,
                           rpn_max_norm: float = 5.0,
                           roi_max_norm: float = 1.0):
    """[R40] 分组梯度裁剪：RPN 用较大阈值，ROI Head 用严格阈值。

    设计理由：
    - RPN 正样本极度稀疏（百万级 anchors 中 1-40 个正样本），梯度信号本就微弱，
      max_norm=1.0 会过度压缩有效梯度，导致 RPN 学不动（[R28] 已验证）。
      → 用 5.0 给 RPN 足够的梯度空间。
    - ROI Head 的 bbox 回归需要高精度（IoU 目标 0.7），梯度抖动会直接导致 box 不稳。
      → 用 1.0 严格限制，符合 project_memory 硬约束。
    - 二者解耦后，RPN 学习不被压制，ROI 精修不被抖动干扰。

    Args:
        model: TwoStageDetector
        rpn_max_norm: RPN 参数的梯度范数上限
        roi_max_norm: ROI Head 参数的梯度范数上限

    Returns:
        (rpn_grad_norm, roi_grad_norm): 裁剪前各自的梯度范数，用于诊断
    """
    rpn_params = []
    roi_params = []
    other_params = []  # backbone + FPN

    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if 'rpn.' in name:
            rpn_params.append(p)
        elif 'roi_head.' in name:
            roi_params.append(p)
        else:
            # backbone / FPN 归入 "其他"，用 ROI 的严格阈值（保护预训练权重）
            other_params.append(p)

    rpn_grad_norm = torch.nn.utils.clip_grad_norm_(rpn_params, max_norm=rpn_max_norm).item() \
        if rpn_params else 0.0
    roi_grad_norm = torch.nn.utils.clip_grad_norm_(roi_params, max_norm=roi_max_norm).item() \
        if roi_params else 0.0
    # backbone+FPN 用 ROI 阈值（与 ROI 一致，保护预训练权重不被破坏）
    if other_params:
        torch.nn.utils.clip_grad_norm_(other_params, max_norm=roi_max_norm)

    return rpn_grad_norm, roi_grad_norm


def multi_class_focal_loss(cls_logits: torch.Tensor, targets: torch.Tensor,
                           alpha=None, gamma: float = 2.0) -> torch.Tensor:
    """[R40已弃用] Sigmoid-per-class Focal Loss。

    历史用途：早期 [Round14-18] 用于 ROI 分类，[Round19] 起改为 CrossEntropyLoss 启用类间竞争。
    函数体已删除（清理死代码），如需恢复请查阅 git 历史 R39 版本。
    """
    raise NotImplementedError(
        "multi_class_focal_loss 已弃用 (R40)。"
        "ROI 分类实际使用 F.cross_entropy (train.py:949)，"
        "符合 project_memory 中 'CrossEntropy Loss 必须用于分类' 的硬约束。"
    )


# ============================================================
# 冻结/解冻函数
# ============================================================

def freeze_backbone(model: TwoStageDetector) -> None:
    """冻结backbone的layer1和layer2（保留layer3可训练）。

    阶段1只冻浅层，让layer3+FPN一起适应目标数据集。
    """
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    # 注意：不冻结layer3，让FPN的C4输入能适应数据集

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[阶段1] Layer1,Layer2已冻结 (Layer3保持可训练)")
    print(f"  可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")


def unfreeze_backbone(model: TwoStageDetector) -> None:
    """解冻backbone（layer2, layer3）。"""
    for param in model.layer2.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[阶段2] Layer2, Layer3已解冻")
    print(f"  可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")


# ============================================================
# 学习率调度器
# ============================================================

def build_param_groups(model, base_lr, rpn_lr_mult=5.0, wd=5e-4):
    """[R34] 构建参数组：RPN使用更高学习率，无weight_decay。

    [R28] 5x LR让RPN开始学习（logits 0.049→1.52）
    [R33] 10x LR导致RPN cls饱和崩溃到固定空间偏置
    [R34] 回到5x LR + Focal Loss，避免饱和崩溃
    原因：RPN正样本极度稀疏（1-40个/百万级anchors），梯度信号微弱。
    """
    rpn_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'rpn.' in name:
            rpn_params.append(param)
        else:
            other_params.append(param)

    n_rpn = sum(p.numel() for p in rpn_params)
    n_other = sum(p.numel() for p in other_params)
    print(f"  [R28参数组] RPN: {n_rpn:,} params (lr={base_lr*rpn_lr_mult:.6f}, wd=0)")
    print(f"  [R28参数组] 其他: {n_other:,} params (lr={base_lr:.6f}, wd={wd})")

    return [
        {'params': rpn_params, 'lr': base_lr * rpn_lr_mult, 'weight_decay': 0.0},
        {'params': other_params, 'lr': base_lr, 'weight_decay': wd},
    ]


def get_lr_scheduler(optimizer, stage: int, epochs: int):
    """获取学习率调度器（余弦退火+重启）。

    使用CosineAnnealingWarmRestarts实现周期性恢复学习率，
    让模型在长时间训练中保持学习能力。

    Args:
        optimizer: 优化器
        stage: 训练阶段 (1或2)
        epochs: 该阶段的总轮数

    Returns:
        scheduler: 学习率调度器
    """
    if stage == 1:
        # 阶段1: 每10轮重启一次
        T_0 = 10  # 首次重启周期
        T_mult = 2  # 后续周期倍增（10, 20, 40...）
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=1e-6  # 最小学习率，不为0
        )
    else:
        # 阶段2: 每15轮重启一次（更长周期用于精细调优）
        T_0 = 15
        T_mult = 2
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=1e-7
        )


class WarmupCosineScheduler:
    """带预热的余弦退火调度器。

    前 warmup_epochs 轮：线性从 ~0 增长到各参数组的 target_lr，并**真正写入 optimizer**。
    之后：交给 base_scheduler（CosineAnnealingWarmRestarts）。

    用法：
        base_scheduler = CosineAnnealingWarmRestarts(optimizer, ...)
        scheduler = WarmupCosineScheduler(base_scheduler, warmup_epochs=5)
        for epoch in range(epochs):
            train(...)
            scheduler.step()
    """

    def __init__(self, base_scheduler, warmup_epochs: int = 5):
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        # 各参数组预热结束时的目标 lr（阶段1: RPN 5x / 其他 1x；阶段2同理）。
        # base_scheduler.base_lrs 与 optimizer.param_groups 的初始 lr 一致。
        if hasattr(self.base_scheduler, 'base_lrs'):
            self.target_lrs = list(self.base_scheduler.base_lrs)
        else:
            self.target_lrs = [g['lr'] for g in self.base_scheduler.optimizer.param_groups]
        self.last_lrs = [0.0] * len(self.target_lrs)
        # 预热从 ~0 起步：立即把各参数组 lr 置为极小值，使第 0 个 epoch 即低 lr 开始
        self._apply_lrs([t * 1e-8 for t in self.target_lrs])

    def _apply_lrs(self, lrs):
        """把 lrs 写入 optimizer 各参数组，并记录为 last_lrs（真实 lr）。"""
        self.last_lrs = list(lrs)
        for group, lr in zip(self.base_scheduler.optimizer.param_groups, lrs):
            group['lr'] = lr

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # 线性预热：lr = target * (epoch+1) / warmup_epochs，写回 optimizer
            self.current_epoch += 1
            progress = self.current_epoch / self.warmup_epochs
            self._apply_lrs([t * progress for t in self.target_lrs])
        else:
            self.current_epoch += 1
            self.base_scheduler.step()
            self.last_lrs = list(self.base_scheduler.get_last_lr())

    def get_last_lr(self):
        """返回真实当前 lr（各参数组，已写入 optimizer）。"""
        return self.last_lrs

    def state_dict(self):
        """[C3] 序列化调度器状态（供 checkpoint 保存）。"""
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
        }

    def load_state_dict(self, sd):
        """[C3] 恢复调度器状态（供 --resume）。需在 optimizer.load_state_dict 之后调用。"""
        self.base_scheduler.load_state_dict(sd['base_scheduler'])
        self.current_epoch = sd.get('current_epoch', 0)
        self.warmup_epochs = sd.get('warmup_epochs', self.warmup_epochs)
        # 从 optimizer 读取真实当前 lr（optimizer.load_state_dict 已恢复 lr）
        self.last_lrs = [g['lr'] for g in self.base_scheduler.optimizer.param_groups]


# ============================================================
# 早停机制
# ============================================================

class EarlyStopping:
    """早停机制。"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


# ============================================================
# 训练跟踪器
# ============================================================

class TrainingTracker:
    """训练跟踪器。"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.rpn_cls_losses = []
        self.rpn_bbox_losses = []
        self.learning_rates = []
        self.stages = []
        self.val_maps = []  # [C3] (epoch, mAP) 每 N epoch 的轻量验证结果

    def add(self, train_loss: float, val_loss: float,
            rpn_cls_loss: float, rpn_bbox_loss: float,
            lr: float, stage: int):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.rpn_cls_losses.append(rpn_cls_loss)
        self.rpn_bbox_losses.append(rpn_bbox_loss)
        self.learning_rates.append(lr)
        self.stages.append(stage)

    def add_val_map(self, epoch: int, mAP: float):
        self.val_maps.append((epoch, mAP))
    
    def save_curves(self, save_path: str):
        """保存训练曲线。"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # RPN损失
        axes[0, 1].plot(epochs, self.rpn_cls_losses, 'g-', label='RPN Cls Loss')
        axes[0, 1].plot(epochs, self.rpn_bbox_losses, 'm-', label='RPN Bbox Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('RPN Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率
        axes[1, 0].plot(epochs, self.learning_rates, 'c-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].grid(True)
        
        # 阶段标记
        stage1_epochs = sum(1 for s in self.stages if s == 1)
        if stage1_epochs > 0:
            axes[1, 1].axvline(x=stage1_epochs, color='r', linestyle='--', label='Stage 1→2')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Stage')
        axes[1, 1].set_title('Training Stages')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def save_log(self, save_path: str):
        """保存训练日志。"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Epoch,Stage,Train_Loss,Val_Loss,RPN_Cls_Loss,RPN_Bbox_Loss,LR\n")
            for i in range(len(self.train_losses)):
                f.write(f"{i+1},{self.stages[i]},"
                        f"{self.train_losses[i]:.6f},"
                        f"{self.val_losses[i]:.6f},"
                        f"{self.rpn_cls_losses[i]:.6f},"
                        f"{self.rpn_bbox_losses[i]:.6f},"
                        f"{self.learning_rates[i]:.6f}\n")


# ============================================================
# RPN训练器
# ============================================================

class RPNTrainer:
    """RPN训练器。
    
    正负样本匹配:
    - 正样本: IoU >= 0.5
    - 负样本: IoU < 0.3
    - 正负比例: 1:3
    """
    
    def __init__(self):
        cfg = FPN_CONFIG['rpn_config']
        self.pos_iou_thresh = cfg['pos_iou_thresh']
        self.neg_iou_thresh = cfg['neg_iou_thresh']
        self.pos_neg_ratio = cfg['pos_neg_ratio']
        self.max_neg_per_bg_window = cfg.get('max_neg_per_bg_window', 64)  # [问题2] 背景窗口负样本上限

        # [R35] RPN分类回退到标准BCEWithLogitsLoss
        # [R28-R34] Focal Loss/加权BCE均未解决spatial bias
        # [R35] SAHI切片让缺陷清晰可学，先用最简单的BCE建立基线
        # 切片后正样本anchor数量提升(缺陷占比12-27%)，不再需要加权
        self.bbox_loss_fn = CIoULoss()

        # [A13] anchor 采样统计（仅监控，不改采样策略）
        self.reset_stats()

    def reset_stats(self) -> None:
        """[A13] 重置每 epoch 的 RPN anchor 采样统计。

        只做监控，不改变 pos:neg=1:3 的采样策略。统计口径：
        - pos_anchors: 所有正 anchor 总数（各 FPN 层累计，正样本全保留）。
        - neg_anchors: 采样进入损失的负 anchor 总数。
        - pos_win_neg_anchors: 来自含缺陷窗口的负 anchor 数。
        - bg_neg_anchors: 来自背景窗口（无 GT）的负 anchor 数。
        """
        self.stats = {
            'pos_anchors': 0,
            'neg_anchors': 0,
            'pos_win_neg_anchors': 0,
            'bg_neg_anchors': 0,
        }

    def assign_targets(self, anchors: torch.Tensor, gt_boxes: torch.Tensor):
        """为锚框分配目标标签。"""
        device = anchors.device
        num_anchors = anchors.size(0)
        
        gt_boxes = gt_boxes.to(device)
        
        if gt_boxes.size(0) == 0:
            labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
            matched_gt = torch.zeros(num_anchors, dtype=torch.long, device=device)
            return labels, matched_gt, torch.tensor([], dtype=torch.long, device=device)
        
        iou = box_iou(anchors, gt_boxes)
        max_iou, matched_gt = iou.max(dim=1)
        
        labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        labels[max_iou >= self.pos_iou_thresh] = 1
        labels[max_iou < self.neg_iou_thresh] = 0
        labels[(max_iou >= self.neg_iou_thresh) & (max_iou < self.pos_iou_thresh)] = -1
        
        for i in range(gt_boxes.size(0)):
            max_iou_for_gt = iou[:, i].argmax()
            labels[max_iou_for_gt] = 1
        
        pos_inds = (labels == 1).nonzero(as_tuple=True)[0]
        return labels, matched_gt, pos_inds
    
    def sample_anchors(self, labels: torch.Tensor, cls_pred: torch.Tensor = None,
                        batch_size: int = 512):
        """[R32] 随机/截断采样锚框用于训练。

        [R32修改] batch_size从256→512，配合pos_thresh=0.3增加正样本数。
        原因：R31诊断显示RPN正样本极度稀疏（1-40个/百万级anchors），
              导致RPN学到固定spatial bias。增加batch_size让RPN看到更多正样本。

        策略:
        - 正样本: 全部保留（正样本稀缺，尤其是稀有类别）
        - 负样本: 随机截取（R12-B已验证有效，OHEM已回退）
        """
        pos_inds = (labels == 1).nonzero(as_tuple=True)[0]
        neg_inds = (labels == 0).nonzero(as_tuple=True)[0]

        if pos_inds.size(0) > 0:
            # 含缺陷窗口：正样本限量，负样本按 pos:neg=1:3 截取（负样本主要来源）
            num_pos = min(pos_inds.size(0), batch_size // (self.pos_neg_ratio + 1))
            num_neg = min(neg_inds.size(0), num_pos * self.pos_neg_ratio)
        else:
            # [问题2] 背景窗口：无正样本，负样本限量 max_neg_per_bg_window（默认 64）。
            # 之前 num_pos=0 -> num_neg=0 -> 最终返回全量负 anchor（~672k/窗口），
            # 导致背景窗口贡献 99.7% 负 anchor 淹没正样本。现在每背景窗口最多补 64 个。
            num_pos = 0
            num_neg = min(neg_inds.size(0), self.max_neg_per_bg_window)

        # 正样本: 随机截取(正样本内部难度差异小)
        if pos_inds.size(0) > num_pos and num_pos > 0:
            perm = torch.randperm(pos_inds.size(0), device=pos_inds.device)[:num_pos]
            pos_inds = pos_inds[perm]

        # 负样本: 随机截取（R12-B基线策略，不使用OHEM）
        if neg_inds.size(0) > num_neg and num_neg > 0:
            perm = torch.randperm(neg_inds.size(0), device=neg_inds.device)[:num_neg]
            neg_inds = neg_inds[perm]

        if pos_inds.size(0) > 0 or neg_inds.size(0) > 0:
            return torch.cat([pos_inds, neg_inds])
        return pos_inds
    
    def compute_loss(self, cls_pred: torch.Tensor, bbox_pred: torch.Tensor,
                     anchors: torch.Tensor, gt_boxes: torch.Tensor,
                     decode_fn=None):
        """计算RPN损失。

        Args:
            decode_fn: 可选的解码函数，用于将delta转为绝对坐标(CIoU需要)
        """
        labels, matched_gt, pos_inds = self.assign_targets(anchors, gt_boxes)
        sampled_inds = self.sample_anchors(labels)  # [R12-B基线] 不传cls_pred，使用随机采样

        # [A13] 统计正/负 anchor 采样情况（分别记录含缺陷窗口 / 背景窗口两个负样本来源）
        if hasattr(self, 'stats'):
            n_pos = int((labels == 1).sum().item())
            n_neg_sampled = int((labels[sampled_inds] == 0).sum().item()) if sampled_inds.size(0) > 0 else 0
            self.stats['pos_anchors'] += n_pos
            self.stats['neg_anchors'] += n_neg_sampled
            if gt_boxes.size(0) == 0:
                self.stats['bg_neg_anchors'] += n_neg_sampled
            else:
                self.stats['pos_win_neg_anchors'] += n_neg_sampled

        device = anchors.device
        gt_boxes = gt_boxes.to(device)

        if sampled_inds.size(0) == 0:
            return (torch.tensor(0.0, device=cls_pred.device),
                    torch.tensor(0.0, device=cls_pred.device),
                    torch.tensor(0.0, device=cls_pred.device))

        sampled_labels = labels[sampled_inds]
        # [R35] 标准BCEWithLogitsLoss，简化建立SAHI基线
        targets = sampled_labels.float()
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_pred[sampled_inds], targets
        )

        if pos_inds.size(0) > 0:
            pos_anchors = anchors[pos_inds]
            pos_gt = gt_boxes[matched_gt[pos_inds]]

            if decode_fn is not None:
                # CIoU Loss: 在绝对坐标空间计算（与模型设计一致）
                pred_boxes = decode_fn(pos_anchors, bbox_pred[pos_inds])
                bbox_loss = self.bbox_loss_fn(pred_boxes, pos_gt)
            else:
                # 兜底: Smooth L1（在delta空间计算）
                target_deltas = encode_boxes(pos_gt, pos_anchors)
                bbox_loss = torch.nn.functional.smooth_l1_loss(
                    bbox_pred[pos_inds], target_deltas, reduction='mean'
                )
        else:
            bbox_loss = torch.tensor(0.0, device=cls_pred.device)

        total_loss = cls_loss + bbox_loss
        return cls_loss, bbox_loss, total_loss


# ============================================================
# 训练函数
# ============================================================

def train_one_epoch(model, train_loader, optimizer, rpn_trainer, device, img_size,
                     scaler: GradScaler = None):
    """训练一个epoch（完整两阶段）。

    Args:
        scaler: AMP GradScaler，为None时不使用混合精度
    """
    model.train()
    rpn_trainer.reset_stats()  # [A13] 每 epoch 重置采样统计

    total_loss = 0.0
    total_rpn_cls = 0.0
    total_rpn_bbox = 0.0
    total_roi_cls = 0.0
    total_roi_bbox = 0.0
    num_batches = 0
    
    for imgs, num_objects, bbox_labels, meta in train_loader:
        imgs = imgs.to(device)
        batch_size = imgs.size(0)

        # AMP: 前向传播在混合精度下运行
        if scaler is not None:
            with autocast('cuda'):
                outputs = model(imgs)
        else:
            outputs = model(imgs)
        fpn_features = outputs['fpn_features']
        rpn_cls_preds = outputs['rpn_cls_preds']
        rpn_bbox_preds = outputs['rpn_bbox_preds']
        
        feature_sizes = [(f.size(2), f.size(3)) for f in fpn_features]
        anchors_dict = model.rpn.get_anchors(feature_sizes, device)
        
        batch_rpn_cls_loss = 0.0
        batch_rpn_bbox_loss = 0.0
        batch_roi_cls_loss = 0.0
        batch_roi_bbox_loss = 0.0
        num_valid_samples = 0  # 有效样本数（背景窗口也计入：参与 RPN 负样本损失）

        for b in range(batch_size):
            n_obj = num_objects[b].item()

            if n_obj == 0:
                # [P0.4] 背景窗口：空 GT，参与 RPN 分类损失（负样本），ROI 正样本跳过
                gt_boxes = torch.zeros((0, 4), device=device)
                gt_class_ids = torch.zeros((0,), dtype=torch.long, device=device)
            else:
                gt_boxes_xywh = bbox_labels[b, :n_obj, :4]
                gt_boxes = xywh_to_xyxy(gt_boxes_xywh, img_size).to(device)
                gt_class_ids = bbox_labels[b, :n_obj, 4].long().to(device)

            num_valid_samples += 1
            
            # RPN loss：对每层切片计算正负样本损失（[R26] permute 对齐 anchor）
            for i, layer_name in enumerate(['P2', 'P3', 'P4']):
                anchors = anchors_dict[layer_name]
                _lc = FPN_CONFIG['anchor_configs'][layer_name]
                num_anchors_actual = len(_lc['scales']) * len(_lc['ratios'])
                max_a = model.rpn.max_anchors_per_loc

                cls_pred = rpn_cls_preds[i][b].permute(1, 2, 0).contiguous()  # [H, W, max_a]
                cls_pred = cls_pred.reshape(-1, max_a)[:, :num_anchors_actual].reshape(-1)
                bbox_pred = rpn_bbox_preds[i][b].permute(1, 2, 0).contiguous()
                bbox_pred = bbox_pred.reshape(-1, max_a, 4)[:, :num_anchors_actual, :].reshape(-1, 4)

                cls_loss, bbox_loss, _ = rpn_trainer.compute_loss(
                    cls_pred, bbox_pred, anchors, gt_boxes,
                    decode_fn=lambda a, d: model.decode_boxes(a, d, img_w=img_size, img_h=img_size)
                )
                batch_rpn_cls_loss = batch_rpn_cls_loss + cls_loss
                batch_rpn_bbox_loss = batch_rpn_bbox_loss + bbox_loss

            if n_obj == 0:
                # [P0.4] 背景窗口：跳过 ROI Head（无 GT 正样本，只贡献 RPN 负样本）
                continue

            # [P0.5] 统一 proposal 提取 + 按尺寸分配 FPN 层 + roi_align
            proposals, _ = extract_rpn_proposals(
                model, rpn_cls_preds, rpn_bbox_preds, fpn_features, anchors_dict,
                img_size=img_size, batch_idx=b)
            if proposals.size(0) == 0:
                continue

            roi_features, roi_orig_indices = assign_levels_and_roi_align(
                proposals, fpn_features, batch_idx=b, device=device,
                target_dtype=model.roi_head.shared_fc[0].weight.dtype)
            if roi_features is None:
                continue
            
            cls_logits, bbox_deltas = model.roi_head(roi_features)
            
            # 计算原始proposal的IoU
            ious = box_iou(proposals, gt_boxes)
            max_ious, matched_gt_idx = ious.max(dim=1)

            # [R27] 降低pos_thresh从0.5→0.4，增加ROI正样本数
            # 原因：RPN proposal质量不高时，0.5阈值过于严格
            # [R30] 降低pos_thresh从0.4→0.3，neg_thresh从0.3→0.2
            # 原因：inference时99.7% proposal IoU=0，0.3% IoU>0.3。
            #       R29的pos_thresh=0.4太严格，让ROI Head见过更多低IoU正样本。
            pos_thresh = 0.3
            neg_thresh = 0.2

            pos_mask = max_ious >= pos_thresh
            neg_mask = (max_ious < neg_thresh) & (~pos_mask)
            
            # 映射到ROI特征的索引
            roi_pos_mask = pos_mask[roi_orig_indices]
            roi_neg_mask = neg_mask[roi_orig_indices]
            
            # [Round 9] 恢复Round 6验证通过的FocalLoss配置
            num_pos = min(roi_pos_mask.sum().item(), 32)
            num_neg = min(roi_neg_mask.sum().item(), num_pos * 3)

            # [R31修复] 移除GT注入后，如果num_pos==0，选择最高IoU的proposal作为正样本保底
            # 原因：R28的RPN已学习（logits=1.52），每张图有0-9个IoU>0.3的proposal。
            #       但偶尔num_pos==0时不能跳过（会导致ROI Head不训练），
            #       而是用最高IoU的proposal作为正样本（即使IoU<0.3）。
            #       这样ROI Head始终用真实的RPN proposal训练，分布与推理一致。
            if num_pos == 0:
                # 选择IoU最高的5个proposal作为正样本
                roi_max_ious = max_ious[roi_orig_indices]
                top_k = min(5, roi_max_ious.size(0))
                _, top_iou_inds = roi_max_ious.topk(top_k)
                pos_inds = top_iou_inds
                # 重新计算num_neg
                num_pos = top_k
                num_neg = min(roi_neg_mask.sum().item(), num_pos * 3)
                if num_neg == 0:
                    # 如果也没有负样本，随机选一些
                    all_inds = torch.arange(roi_features.size(0), device=device)
                    perm = torch.randperm(all_inds.size(0), device=device)[:min(num_pos * 3, all_inds.size(0))]
                    neg_inds = all_inds[perm]
                else:
                    raw_neg_inds = torch.where(roi_neg_mask)[0]
                    if raw_neg_inds.size(0) > num_neg:
                        perm = torch.randperm(raw_neg_inds.size(0), device=device)[:num_neg]
                        neg_inds = raw_neg_inds[perm]
                    else:
                        neg_inds = raw_neg_inds
            else:
                pos_inds = torch.where(roi_pos_mask)[0][:num_pos]

                # [R12-B基线] 负样本: 随机截取（不使用OHEM）
                raw_neg_inds = torch.where(roi_neg_mask)[0]
                if raw_neg_inds.size(0) > num_neg:
                    perm = torch.randperm(raw_neg_inds.size(0), device=device)[:num_neg]
                    neg_inds = raw_neg_inds[perm]
                else:
                    neg_inds = raw_neg_inds[:num_neg] if raw_neg_inds.size(0) > 0 else raw_neg_inds
            
            sampled_inds = torch.cat([pos_inds, neg_inds])
            
            sampled_cls_logits = cls_logits[sampled_inds]
            # cls_logits: [N, 6] — 索引0=BG, 索引1-5=前景类(chongkong~hanfeng)
            # 正样本标签 = gt_class_id + 1（将0-4映射到1-5）
            # 负样本标签 = 0（背景类）
            sampled_gt_labels = torch.zeros(len(sampled_inds), dtype=torch.long, device=device)
            orig_pos_indices = roi_orig_indices[pos_inds]
            sampled_gt_labels[:len(pos_inds)] = gt_class_ids[matched_gt_idx[orig_pos_indices]] + 1

            # [Round19] 使用CrossEntropyLoss替代FocalLoss（启用类间竞争）
            # [R24] xinpaide数据集2类权重（missing, dirty各100张，balanced）
            # 权重: BG=0.5, missing=1.0, dirty=1.0
            if not hasattr(model, '_cls_weight'):
                # [A7] 动态生成类别权重：BG=0.5，各前景类=1.0（长度 = num_classes+1）
                cls_w = torch.tensor([0.5] + [1.0] * len(config.class_names), device=device)
                model._cls_weight = cls_w
            roi_cls_loss = F.cross_entropy(sampled_cls_logits, sampled_gt_labels,
                                           weight=model._cls_weight)

            if len(pos_inds) > 0:
                pos_proposals = proposals[roi_orig_indices[pos_inds]]
                pos_gt_boxes = gt_boxes[matched_gt_idx[roi_orig_indices[pos_inds]]]
                # [Round19] bbox_deltas现在是共享4维[N,4]，直接使用
                pos_bbox_deltas = bbox_deltas[pos_inds]  # [num_pos, 4]

                pred_boxes = model.decode_boxes(pos_proposals, pos_bbox_deltas, img_w=img_size, img_h=img_size)
                roi_bbox_loss = CIoULoss()(pred_boxes, pos_gt_boxes)
            else:
                roi_bbox_loss = torch.tensor(0.0, device=device)
            
            batch_roi_cls_loss = batch_roi_cls_loss + roi_cls_loss
            batch_roi_bbox_loss = batch_roi_bbox_loss + roi_bbox_loss
        
        # 用有效样本数做平均（背景窗口也计入，因其参与 RPN 负样本损失）
        effective_batch_size = max(num_valid_samples, 1)
        # [Round18] 回退到等权loss（R17的2x bbox权重无益）
        # [R39] 验证loss也用相同权重，保持训练/验证一致
        batch_loss = (batch_rpn_cls_loss + batch_rpn_bbox_loss +
                     batch_roi_cls_loss + 2.0 * batch_roi_bbox_loss) / effective_batch_size

        # [Round14修复] 确保batch_loss始终为Tensor（防止全空批次时scaler报错）
        if not isinstance(batch_loss, torch.Tensor):
            # 全空批次：跳过本次优化步骤
            continue

        optimizer.zero_grad()
        if scaler is not None:
            # AMP: 缩放loss防止FP16下溢，自动处理混合精度反向传播
            scaler.scale(batch_loss).backward()
            # [R40] 分组梯度裁剪（在unscale之后执行）
            scaler.unscale_(optimizer)
            rpn_gn, roi_gn = clip_grad_norm_grouped(model, rpn_max_norm=5.0, roi_max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            # [R40] 分组梯度裁剪：RPN 用 5.0（梯度信号弱需空间），ROI Head 用 1.0（bbox精修需稳定）
            # 取代旧的统一 max_norm=5.0：既满足 project_memory 硬约束（1.0 for bbox精修），
            # 又不让 RPN 学不动（[R28] 经验）。
            rpn_gn, roi_gn = clip_grad_norm_grouped(model, rpn_max_norm=5.0, roi_max_norm=1.0)
            optimizer.step()
        
        total_loss += batch_loss.item() if hasattr(batch_loss, 'item') else batch_loss
        total_rpn_cls += (batch_rpn_cls_loss.item() if hasattr(batch_rpn_cls_loss, 'item') else batch_rpn_cls_loss) / effective_batch_size
        total_rpn_bbox += (batch_rpn_bbox_loss.item() if hasattr(batch_rpn_bbox_loss, 'item') else batch_rpn_bbox_loss) / effective_batch_size
        total_roi_cls += (batch_roi_cls_loss.item() if hasattr(batch_roi_cls_loss, 'item') else batch_roi_cls_loss) / effective_batch_size
        total_roi_bbox += (batch_roi_bbox_loss.item() if hasattr(batch_roi_bbox_loss, 'item') else batch_roi_bbox_loss) / effective_batch_size
        num_batches += 1
    
    # [A13] 每 epoch 打印 RPN anchor 采样统计（两个负样本来源分别打印）
    s = rpn_trainer.stats
    print(f"[A13] RPN采样: 正anchor={s['pos_anchors']:,}  负anchor={s['neg_anchors']:,} "
          f"(含缺陷窗口={s['pos_win_neg_anchors']:,}, 背景窗口={s['bg_neg_anchors']:,})")

    return (total_loss / max(num_batches, 1),
            total_rpn_cls / max(num_batches, 1),
            total_rpn_bbox / max(num_batches, 1),
            total_roi_cls / max(num_batches, 1),
            total_roi_bbox / max(num_batches, 1))


def validate(model, val_loader, rpn_trainer, device, img_size):
    """验证（完整两阶段）。"""
    model.eval()
    
    total_loss = 0.0
    total_rpn_cls = 0.0
    total_rpn_bbox = 0.0
    total_roi_cls = 0.0
    total_roi_bbox = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, num_objects, bbox_labels, meta in val_loader:
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            
            outputs = model(imgs)
            fpn_features = outputs['fpn_features']
            rpn_cls_preds = outputs['rpn_cls_preds']
            rpn_bbox_preds = outputs['rpn_bbox_preds']
            
            feature_sizes = [(f.size(2), f.size(3)) for f in fpn_features]
            anchors_dict = model.rpn.get_anchors(feature_sizes, device)
            
            batch_rpn_cls_loss = 0.0
            batch_rpn_bbox_loss = 0.0
            batch_roi_cls_loss = 0.0
            batch_roi_bbox_loss = 0.0
            num_valid_samples = 0
            
            for b in range(batch_size):
                n_obj = num_objects[b].item()
                if n_obj == 0:
                    # [P0.4] 背景窗口：空 GT，参与 RPN 分类损失（负样本），ROI 正样本跳过（与 train 对齐）
                    gt_boxes = torch.zeros((0, 4), device=device)
                    gt_class_ids_val = torch.zeros((0,), dtype=torch.long, device=device)
                else:
                    gt_boxes_xywh = bbox_labels[b, :n_obj, :4]
                    gt_boxes = xywh_to_xyxy(gt_boxes_xywh, img_size).to(device)
                    gt_class_ids_val = bbox_labels[b, :n_obj, 4].long().to(device)

                num_valid_samples += 1

                # RPN loss：对每层切片计算（背景窗口也参与，与 train 一致）
                for i, layer_name in enumerate(['P2', 'P3', 'P4']):
                    anchors = anchors_dict[layer_name]
                    _lc2 = FPN_CONFIG['anchor_configs'][layer_name]
                    num_anchors_actual = len(_lc2['scales']) * len(_lc2['ratios'])
                    max_a = model.rpn.max_anchors_per_loc

                    cls_pred = rpn_cls_preds[i][b].permute(1, 2, 0).contiguous()  # [H, W, max_a]
                    cls_pred = cls_pred.reshape(-1, max_a)[:, :num_anchors_actual].reshape(-1)
                    bbox_pred = rpn_bbox_preds[i][b].permute(1, 2, 0).contiguous()
                    bbox_pred = bbox_pred.reshape(-1, max_a, 4)[:, :num_anchors_actual, :].reshape(-1, 4)

                    cls_loss, bbox_loss, _ = rpn_trainer.compute_loss(
                        cls_pred, bbox_pred, anchors, gt_boxes,
                        decode_fn=lambda a, d: model.decode_boxes(a, d, img_w=img_size, img_h=img_size)
                    )
                    batch_rpn_cls_loss = batch_rpn_cls_loss + cls_loss
                    batch_rpn_bbox_loss = batch_rpn_bbox_loss + bbox_loss

                if n_obj == 0:
                    # [P0.4] 背景窗口：跳过 ROI Head
                    continue

                # [P0.5] 统一 proposal 提取 + 按尺寸分配 FPN 层 + roi_align
                proposals, _ = extract_rpn_proposals(
                    model, rpn_cls_preds, rpn_bbox_preds, fpn_features, anchors_dict,
                    img_size=img_size, batch_idx=b)
                if proposals.size(0) == 0:
                    continue

                roi_features, roi_orig_indices = assign_levels_and_roi_align(
                    proposals, fpn_features, batch_idx=b, device=device,
                    target_dtype=model.roi_head.shared_fc[0].weight.dtype)
                if roi_features is None:
                    continue
                
                cls_logits, bbox_deltas = model.roi_head(roi_features)
                
                ious = box_iou(proposals, gt_boxes)
                max_ious, matched_gt_idx = ious.max(dim=1)

                # [R27] 与训练一致：pos_thresh=0.4
                # [R30] 与训练一致：pos_thresh=0.3, neg_thresh=0.2
                pos_mask = max_ious >= 0.3
                neg_mask = (max_ious < 0.2) & (~pos_mask)
                
                roi_pos_mask = pos_mask[roi_orig_indices]
                roi_neg_mask = neg_mask[roi_orig_indices]
                
                num_pos = min(roi_pos_mask.sum().item(), 32)
                num_neg = min(roi_neg_mask.sum().item(), num_pos * 3)

                # [R40] num_pos==0 保底：与训练阶段 (train.py:901-921) 完全一致
                # 之前验证阶段没有保底，空批次被直接跳过，导致验证loss系统性偏低（不计入分母）。
                # 现在取IoU最高的5个proposal作为正样本，让验证loss反映真实分布。
                if num_pos == 0:
                    roi_max_ious = max_ious[roi_orig_indices]
                    top_k = min(5, roi_max_ious.size(0))
                    if top_k == 0:
                        continue
                    _, top_iou_inds = roi_max_ious.topk(top_k)
                    pos_inds = top_iou_inds
                    num_pos = top_k
                    num_neg = min(roi_neg_mask.sum().item(), num_pos * 3)
                    if num_neg == 0:
                        # 负样本也没有，随机抽
                        all_inds = torch.arange(roi_features.size(0), device=device)
                        perm = torch.randperm(all_inds.size(0), device=device)[:min(num_pos * 3, all_inds.size(0))]
                        neg_inds = all_inds[perm]
                    else:
                        raw_neg_inds_val = torch.where(roi_neg_mask)[0]
                        if raw_neg_inds_val.size(0) > num_neg:
                            perm = torch.randperm(raw_neg_inds_val.size(0), device=device)[:num_neg]
                            neg_inds = raw_neg_inds_val[perm]
                        else:
                            neg_inds = raw_neg_inds_val
                else:
                    pos_inds = torch.where(roi_pos_mask)[0][:num_pos]
                    raw_neg_inds_val = torch.where(roi_neg_mask)[0]
                    if raw_neg_inds_val.size(0) > num_neg:
                        perm = torch.randperm(raw_neg_inds_val.size(0), device=device)[:num_neg]
                        neg_inds = raw_neg_inds_val[perm]
                    else:
                        neg_inds = raw_neg_inds_val

                sampled_inds = torch.cat([pos_inds, neg_inds])

                sampled_cls_logits = cls_logits[sampled_inds]
                sampled_gt_labels = torch.zeros(len(sampled_inds), dtype=torch.long, device=device)
                orig_pos_indices = roi_orig_indices[pos_inds]
                sampled_gt_labels[:len(pos_inds)] = gt_class_ids_val[matched_gt_idx[orig_pos_indices]] + 1

                # [Round19] CrossEntropyLoss（与训练一致，带类别权重）
                if hasattr(model, '_cls_weight'):
                    roi_cls_loss = F.cross_entropy(sampled_cls_logits, sampled_gt_labels,
                                                   weight=model._cls_weight)
                else:
                    roi_cls_loss = F.cross_entropy(sampled_cls_logits, sampled_gt_labels)

                if len(pos_inds) > 0:
                    pos_proposals = proposals[roi_orig_indices[pos_inds]]
                    pos_gt_boxes = gt_boxes[matched_gt_idx[roi_orig_indices[pos_inds]]]
                    # [Round19] 共享4维delta，直接使用
                    pos_bbox_deltas = bbox_deltas[pos_inds]

                    pred_boxes = model.decode_boxes(pos_proposals, pos_bbox_deltas, img_w=img_size, img_h=img_size)
                    roi_bbox_loss = CIoULoss()(pred_boxes, pos_gt_boxes)
                else:
                    roi_bbox_loss = torch.tensor(0.0, device=device)

                batch_roi_cls_loss = batch_roi_cls_loss + roi_cls_loss
                batch_roi_bbox_loss = batch_roi_bbox_loss + roi_bbox_loss
            
            effective_batch_size = max(num_valid_samples, 1)
            # [Round18] 回退到等权
            # [R39] 增加roi_bbox权重到2.0，强化bbox回归精度
            # 诊断发现missing漏检IoU=0.47，差0.03过阈值，是bbox回归不足
            batch_loss = (batch_rpn_cls_loss + batch_rpn_bbox_loss +
                         batch_roi_cls_loss + 2.0 * batch_roi_bbox_loss) / effective_batch_size

            total_loss += batch_loss.item() if hasattr(batch_loss, 'item') else batch_loss
            total_rpn_cls += (batch_rpn_cls_loss.item() if hasattr(batch_rpn_cls_loss, 'item') else batch_rpn_cls_loss) / effective_batch_size
            total_rpn_bbox += (batch_rpn_bbox_loss.item() if hasattr(batch_rpn_bbox_loss, 'item') else batch_rpn_bbox_loss) / effective_batch_size
            total_roi_cls += (batch_roi_cls_loss.item() if hasattr(batch_roi_cls_loss, 'item') else batch_roi_cls_loss) / effective_batch_size
            total_roi_bbox += (batch_roi_bbox_loss.item() if hasattr(batch_roi_bbox_loss, 'item') else batch_roi_bbox_loss) / effective_batch_size
            num_batches += 1
    
    return (total_loss / max(num_batches, 1),
            total_rpn_cls / max(num_batches, 1),
            total_rpn_bbox / max(num_batches, 1),
            total_roi_cls / max(num_batches, 1),
            total_roi_bbox / max(num_batches, 1))


def _get_rng_state():
    """[C3] 采集 Python / NumPy / PyTorch 的随机数状态。"""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    """[C3] 恢复随机数状态。"""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def save_checkpoint(path, model, optimizer, scheduler, epoch, stage, best_loss):
    """[C3] 保存完整 checkpoint（模型 + 优化器 + 调度器 + epoch + RNG + config）。"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,            # 全局 epoch（0-based，跨两阶段）
        'stage': stage,            # 1 或 2
        'best_loss': best_loss,
        'rng_state': _get_rng_state(),
        'config': dict(config.__dict__),
        'train_config': dict(TRAIN_CONFIG),
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """[C3] 加载 checkpoint，恢复 model / optimizer / scheduler / RNG 全部状态。

    Returns:
        dict: checkpoint 内容（含 epoch / stage / best_loss / config）。
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'rng_state' in ckpt:
        _set_rng_state(ckpt['rng_state'])
    return ckpt


def _build_optimizer_scheduler(model, stage):
    """[C3] 按阶段构建 optimizer + scheduler（并冻结/解冻 backbone）。"""
    if stage == 1:
        freeze_backbone(model)
        lr = TRAIN_CONFIG['stage1_lr']
    else:
        unfreeze_backbone(model)
        lr = TRAIN_CONFIG['stage2_lr']
    param_groups = build_param_groups(
        model, lr, rpn_lr_mult=5.0, wd=TRAIN_CONFIG['weight_decay']
    )
    optimizer = optim.SGD(
        param_groups, momentum=TRAIN_CONFIG['momentum'], nesterov=True
    )
    base_scheduler = get_lr_scheduler(optimizer, stage=stage, epochs=0)
    scheduler = WarmupCosineScheduler(base_scheduler, warmup_epochs=config.warmup_epochs)
    return optimizer, scheduler


def run_val_metrics(model, max_imgs=20):
    """[C3] 轻量 mAP 评估（每 N epoch 调用），失败不阻塞训练。"""
    try:
        from sahi_inference import evaluate_map
        was_training = model.training
        model.eval()
        mAP = evaluate_map(model, split='val', max_imgs=max_imgs)
        if was_training:
            model.train()
        return mAP
    except Exception as e:
        print(f"[val-metrics] 评估失败（继续训练）: {e}")
        model.train()
        return None


def train_model(train_loader, val_loader, model, epochs=None, resume_ckpt=None,
                total_epochs=None, val_metrics_every=0):
    """训练两阶段检测器（支持 AMP + 预热 + --resume 续训 + 每 N epoch 轻量验证）。

    Args:
        resume_ckpt: load_checkpoint 返回的 dict；None 表示从头训练。
        total_epochs: 覆盖总 epoch 数（None = stage1+stage2）。
        val_metrics_every: 每 N epoch 跑一次轻量 mAP 验证（0=关）。
    """
    rpn_trainer = RPNTrainer()
    tracker = TrainingTracker()

    use_amp = config.use_amp and (config.device == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print(f"\n[AMP] 混合精度训练已启用")
    else:
        print(f"\n[AMP] 混合精度未启用 (device={config.device}, use_amp={config.use_amp})")

    warmup_epochs = config.warmup_epochs
    print(f"[Warmup] 预热轮数: {warmup_epochs}")

    stage1_epochs = TRAIN_CONFIG['stage1_epochs']
    stage2_epochs = TRAIN_CONFIG['stage2_epochs']
    if total_epochs is None:
        total_epochs = stage1_epochs + stage2_epochs
    print(f"[Epochs] 总 epoch 数: {total_epochs} (stage1={stage1_epochs}, stage2={stage2_epochs})")

    # [C3] 恢复状态
    start_epoch = 0
    best_loss = float('inf')
    if resume_ckpt is not None:
        start_epoch = int(resume_ckpt.get('epoch', 0)) + 1  # 从下一 epoch 继续
        best_loss = resume_ckpt.get('best_loss', float('inf'))
        print(f"[Resume] 从全局 epoch {start_epoch} 续训，best_loss={best_loss:.4f}")

    early_stopping = EarlyStopping(patience=TRAIN_CONFIG['early_stop_patience'])
    optimizer = None
    scheduler = None
    current_stage = None

    pbar = tqdm(range(start_epoch, total_epochs), desc='训练')
    for epoch in pbar:
        stage = 1 if epoch < stage1_epochs else 2

        # 进入新阶段时（或 resume 后首次）重建 optimizer + scheduler
        if stage != current_stage:
            if stage == 1:
                print("\n" + "=" * 60)
                print("阶段1: 冻结Layer1/Layer2，训练RPN+FPN+检测头")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("阶段2: 解冻Layer3，微调整个模型")
                print("=" * 60)
            optimizer, scheduler = _build_optimizer_scheduler(model, stage)
            # [C3] resume：仅当 checkpoint 记录的 stage == 当前 stage 时恢复 optimizer + scheduler
            if resume_ckpt is not None and resume_ckpt.get('stage') == stage:
                optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
                if resume_ckpt.get('scheduler_state_dict'):
                    scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
                print(f"[Resume] 已恢复阶段{stage} optimizer + scheduler 状态")
            current_stage = stage

        train_loss, rpn_cls, rpn_bbox, roi_cls, roi_bbox = train_one_epoch(
            model, train_loader, optimizer, rpn_trainer,
            config.device, config.img_size, scaler=scaler
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        val_loss, val_rpn_cls, val_rpn_bbox, val_roi_cls, val_roi_bbox = validate(
            model, val_loader, rpn_trainer, config.device, config.img_size
        )

        tracker.add(train_loss, val_loss, rpn_cls, rpn_bbox, current_lr, stage=stage)
        pbar.set_postfix({'stage': stage, 'loss': f'{val_loss:.4f}', 'lr': f'{current_lr:.6f}'})

        # [C3] 每 epoch 保存 latest checkpoint（供自动恢复）
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        latest_path = os.path.join(os.path.dirname(config.model_save_path), "latest.pth")
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, stage, best_loss)

        # best checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(config.model_save_path, model, optimizer, scheduler, epoch, stage, best_loss)

        # [C3] 每 N epoch 轻量 mAP 验证
        if val_metrics_every > 0 and (epoch + 1) % val_metrics_every == 0:
            mAP = run_val_metrics(model, max_imgs=20)
            if mAP is not None:
                tracker.add_val_map(epoch, mAP)
                print(f"[val-metrics] epoch {epoch+1}: mAP@0.5={mAP:.4f}")

        if stage == 2 and early_stopping(val_loss):
            print(f"\n早停！验证损失连续{TRAIN_CONFIG['early_stop_patience']}轮没有改善")
            break

    logger.info(f"最佳验证损失: {best_loss:.4f}")
    return model, tracker


# ============================================================
# 主函数
# ============================================================

def parse_args():
    """[C3] 解析命令行参数（--resume / --exp / --total-epochs / --val-metrics-every）。"""
    import argparse
    parser = argparse.ArgumentParser(description='两阶段缺陷检测训练')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 续训（保存完整状态的 .pth 路径）')
    parser.add_argument('--exp', type=str, default=None,
                        help='实验名称（覆盖默认时间戳）')
    parser.add_argument('--total-epochs', type=int, default=None,
                        help='总 epoch 数（覆盖默认 stage1+stage2）')
    parser.add_argument('--val-metrics-every', type=int, default=0,
                        help='每 N epoch 跑一次轻量 mAP 验证（0=关）')
    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()
    if args.exp:
        # 覆盖实验名，并重新推导派生路径（__post_init__ 已在 import 时跑过）
        config.experiment_name = args.exp
        base_dir = Path("experiments") / config.experiment_name
        config.model_save_path = str(base_dir / "models" / "best_model.pth")
        config.log_dir = str(base_dir / "logs")

    # 设置随机种子
    config.set_seed()
    exp_dir = config.setup_experiment()
    config.print_config()
    config.save_config(exp_dir)  # [C2] 落盘配置快照（可复现）
    
    # 打印训练配置
    print("\n" + "=" * 60)
    print("两阶段检测器训练配置")
    print("=" * 60)
    print(f"  阶段1: Epoch 1~{TRAIN_CONFIG['stage1_epochs']}, 冻结Layer1/2, lr={TRAIN_CONFIG['stage1_lr']}")
    print(f"  阶段2: Epoch {TRAIN_CONFIG['stage1_epochs']+1}~{TRAIN_CONFIG['stage1_epochs']+TRAIN_CONFIG['stage2_epochs']}, 解冻Layer3, lr={TRAIN_CONFIG['stage2_lr']}")
    print(f"  早停容忍: {TRAIN_CONFIG['early_stop_patience']} 轮")
    print(f"  AMP混合精度: {'启用' if config.use_amp else '关闭'}")
    print(f"  学习率预热: {config.warmup_epochs} 轮")
    print("\n[FPN配置]")
    for layer, cfg in FPN_CONFIG['anchor_configs'].items():
        num_anchors = len(cfg['scales']) * len(cfg['ratios'])
        print(f"  {layer}: stride={cfg['stride']}, anchors/位置={num_anchors}")
    print("\n[损失函数（已统一）]")
    print(f"  RPN 分类: BCEWithLogitsLoss (train.py:639)")
    print(f"  RPN 回归: CIoU Loss (绝对坐标空间)")
    print(f"  ROI 分类: CrossEntropyLoss (类间竞争, train.py:949)")
    print(f"  ROI 回归: CIoU Loss (绝对坐标空间)")
    print(f"  梯度裁剪: [R40] 分组裁剪 RPN=5.0 / ROI+Backbone=1.0")
    print(f"  数据增强: [R40] 水平/垂直翻转 + ColorJitter (滑动窗口路径)")
    print(f"  ROI残差:  [R40] residual_proj 投影层已启用")
    print("=" * 60)
    
    # 加载数据
    logger.info("正在加载数据集...")
    # [R24] xinpaide：合并dirty+missing，按8:2划分
    # 训练集：每子集前80张，验证集：每子集后20张（无重叠）
    train_loader = create_dataloader(
        config.train_img_dirs, config.train_label_dirs,
        mode='train', max_samples_per_dir=config.train_samples_per_subset,
        from_end=False
    )
    val_loader = create_dataloader(
        config.val_img_dirs, config.val_label_dirs,
        mode='val', max_samples_per_dir=config.val_samples_per_subset,
        from_end=True
    )
    
    logger.info(f"训练集: {len(train_loader.dataset)}张, 验证集: {len(val_loader.dataset)}张")
    
    # 创建模型
    model = TwoStageDetector(num_classes=len(config.class_names)).to(config.device)
    resume_ckpt = None
    if args.resume:
        # [C3] resume：直接从 checkpoint 恢复权重，无需 ImageNet 预训练
        resume_ckpt = load_checkpoint(args.resume, model)
        logger.info(f"已从 {args.resume} 恢复模型权重")
    else:
        model.load_pretrained_weights()

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    logger.info("开始训练...")
    model, tracker = train_model(train_loader, val_loader, model, None, resume_ckpt=resume_ckpt,
                                 total_epochs=args.total_epochs,
                                 val_metrics_every=args.val_metrics_every)
    
    # 保存结果
    curve_path = str(exp_dir / "training_curves.png")
    tracker.save_curves(curve_path)
    
    log_path = str(exp_dir / "training_log.csv")
    tracker.save_log(log_path)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("训练完成总结")
    print("=" * 70)
    print(f"模型保存路径: {config.model_save_path}")
    print(f"训练曲线: {curve_path}")
    print(f"训练日志: {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
