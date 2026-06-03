"""
两阶段目标检测器训练脚本

整合所有商讨结论：
1. cuDNN确定性随机种子
2. 数据增强（翻转+亮度+对比度±10%）
3. ImageNet标准化
4. 两阶段训练（冻结→解冻）
5. 分阶段学习率（余弦衰减）
6. FPN+RPN架构
7. Focal Loss + GIoU Loss
8. 早停机制
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.ops import nms, roi_align
from tqdm import tqdm
from pathlib import Path

from config import config
from model_fpn import (
    TwoStageDetector, 
    FocalLoss, 
    GIoULoss, 
    FPN_CONFIG,
    INFERENCE_CONFIG
)
from dataset import DefectDataset, create_dataloader
from utils import logger
from evaluate import draw_loss_curves


# ============================================================
# 两阶段训练配置
# ============================================================

TRAIN_CONFIG = {
    'stage1_epochs': 8,       # 阶段1: 冻结backbone (从10改为8，减少过拟合)
    'stage2_epochs': 22,      # 阶段2: 解冻backbone (从40改为22，总30轮)
    'stage1_lr': 1e-3,        # 阶段1学习率（较高）
    'stage2_lr': 1e-4,        # 阶段2学习率（较低）
    'weight_decay': 5e-4,     # 权重衰减 (从1e-4增到5e-4，加强正则化)
    'early_stop_patience': 5, # 早停容忍次数 (从10改为5，更快停止)
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


# ============================================================
# 冻结/解冻函数
# ============================================================

def freeze_backbone(model: TwoStageDetector) -> None:
    """冻结backbone（layer1, layer2, layer3）。"""
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[阶段1] Backbone已冻结")
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
    
    def add(self, train_loss: float, val_loss: float, 
            rpn_cls_loss: float, rpn_bbox_loss: float,
            lr: float, stage: int):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.rpn_cls_losses.append(rpn_cls_loss)
        self.rpn_bbox_losses.append(rpn_bbox_loss)
        self.learning_rates.append(lr)
        self.stages.append(stage)
    
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
        
        self.cls_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.bbox_loss_fn = GIoULoss()
    
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
    
    def sample_anchors(self, labels: torch.Tensor, batch_size: int = 256):
        """采样锚框用于训练。"""
        pos_inds = (labels == 1).nonzero(as_tuple=True)[0]
        neg_inds = (labels == 0).nonzero(as_tuple=True)[0]
        
        num_pos = min(pos_inds.size(0), batch_size // (self.pos_neg_ratio + 1))
        num_neg = min(neg_inds.size(0), num_pos * self.pos_neg_ratio)
        
        if pos_inds.size(0) > num_pos and num_pos > 0:
            perm = torch.randperm(pos_inds.size(0), device=pos_inds.device)[:num_pos]
            pos_inds = pos_inds[perm]
        if neg_inds.size(0) > num_neg and num_neg > 0:
            perm = torch.randperm(neg_inds.size(0), device=neg_inds.device)[:num_neg]
            neg_inds = neg_inds[perm]
        
        if pos_inds.size(0) > 0 or neg_inds.size(0) > 0:
            return torch.cat([pos_inds, neg_inds])
        return pos_inds
    
    def compute_loss(self, cls_pred: torch.Tensor, bbox_pred: torch.Tensor,
                     anchors: torch.Tensor, gt_boxes: torch.Tensor):
        """计算RPN损失。"""
        labels, matched_gt, pos_inds = self.assign_targets(anchors, gt_boxes)
        sampled_inds = self.sample_anchors(labels)
        
        device = anchors.device
        gt_boxes = gt_boxes.to(device)
        
        if sampled_inds.size(0) == 0:
            return (torch.tensor(0.0, device=cls_pred.device),
                    torch.tensor(0.0, device=cls_pred.device),
                    torch.tensor(0.0, device=cls_pred.device))
        
        sampled_labels = labels[sampled_inds]
        cls_loss = self.cls_loss_fn(cls_pred[sampled_inds], sampled_labels.float())
        
        if pos_inds.size(0) > 0:
            pos_anchors = anchors[pos_inds]
            pos_gt = gt_boxes[matched_gt[pos_inds]]
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

def train_one_epoch(model, train_loader, optimizer, rpn_trainer, device, img_size):
    """训练一个epoch（完整两阶段）。"""
    model.train()
    
    total_loss = 0.0
    total_rpn_cls = 0.0
    total_rpn_bbox = 0.0
    total_roi_cls = 0.0
    total_roi_bbox = 0.0
    num_batches = 0
    
    for imgs, num_objects, bbox_labels, meta in train_loader:
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
        
        for b in range(batch_size):
            n_obj = num_objects[b].item()
            if n_obj == 0:
                continue
            
            gt_boxes_xywh = bbox_labels[b, :n_obj, :4]
            gt_boxes = xywh_to_xyxy(gt_boxes_xywh, img_size).to(device)
            gt_class_ids = bbox_labels[b, :n_obj, 4].long().to(device)
            
            proposals = []
            proposal_scores = []
            
            for i, layer_name in enumerate(['P2', 'P3', 'P4']):
                anchors = anchors_dict[layer_name]
                
                cls_pred = rpn_cls_preds[i][b].flatten()
                bbox_pred = rpn_bbox_preds[i][b].permute(1, 2, 0).contiguous()
                bbox_pred = bbox_pred.reshape(-1, 15, 4)
                bbox_pred = bbox_pred.reshape(-1, 4)
                
                boxes = model.decode_boxes(anchors, bbox_pred)
                
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
            else:
                continue
            
            for i, layer_name in enumerate(['P2', 'P3', 'P4']):
                anchors = anchors_dict[layer_name]
                
                cls_pred = rpn_cls_preds[i][b].flatten()
                bbox_pred = rpn_bbox_preds[i][b].permute(1, 2, 0).contiguous()
                bbox_pred = bbox_pred.reshape(-1, 15, 4)
                bbox_pred = bbox_pred.reshape(-1, 4)
                
                cls_loss, bbox_loss, _ = rpn_trainer.compute_loss(
                    cls_pred, bbox_pred, anchors, gt_boxes
                )
                
                batch_rpn_cls_loss += cls_loss
                batch_rpn_bbox_loss += bbox_loss
            
            roi_indices = torch.full((proposals.size(0), 1), b, dtype=torch.float32, device=device)
            roi_boxes = torch.cat([roi_indices, proposals], dim=1)
            
            # 根据proposal大小选择FPN层（修复：之前只使用P3）
            # 小目标用P2(高分辨率)，中目标用P3，大目标用P4
            box_sizes = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
            sqrt_sizes = torch.sqrt(box_sizes)
            
            # 按大小分配到不同FPN层
            p2_mask = sqrt_sizes < 64  # 小目标 -> P2
            p3_mask = (sqrt_sizes >= 64) & (sqrt_sizes < 128)  # 中目标 -> P3
            p4_mask = sqrt_sizes >= 128  # 大目标 -> P4
            
            all_roi_features = []
            all_indices = []
            
            # P2层处理
            if p2_mask.sum() > 0:
                p2_boxes = roi_boxes[p2_mask]
                p2_feat = fpn_features[0]
                p2_roi = roi_align(p2_feat, p2_boxes, output_size=(7, 7), spatial_scale=1.0/4.0)
                all_roi_features.append(p2_roi)
                all_indices.append(torch.where(p2_mask)[0])
            
            # P3层处理
            if p3_mask.sum() > 0:
                p3_boxes = roi_boxes[p3_mask]
                p3_feat = fpn_features[1]
                p3_roi = roi_align(p3_feat, p3_boxes, output_size=(7, 7), spatial_scale=1.0/8.0)
                all_roi_features.append(p3_roi)
                all_indices.append(torch.where(p3_mask)[0])
            
            # P4层处理
            if p4_mask.sum() > 0:
                p4_boxes = roi_boxes[p4_mask]
                p4_feat = fpn_features[2]
                p4_roi = roi_align(p4_feat, p4_boxes, output_size=(7, 7), spatial_scale=1.0/16.0)
                all_roi_features.append(p4_roi)
                all_indices.append(torch.where(p4_mask)[0])
            
            if len(all_roi_features) == 0:
                continue
            
            # 合并来自不同层的ROI特征
            roi_features = torch.cat(all_roi_features, dim=0)
            roi_orig_indices = torch.cat(all_indices, dim=0)
            
            cls_logits, bbox_deltas = model.roi_head(roi_features)
            
            # 计算原始proposal的IoU
            ious = box_iou(proposals, gt_boxes)
            max_ious, matched_gt_idx = ious.max(dim=1)
            
            pos_thresh = 0.5
            neg_thresh = 0.3
            
            pos_mask = max_ious >= pos_thresh
            neg_mask = (max_ious < neg_thresh) & (~pos_mask)
            
            # 映射到ROI特征的索引
            roi_pos_mask = pos_mask[roi_orig_indices]
            roi_neg_mask = neg_mask[roi_orig_indices]
            
            num_pos = min(roi_pos_mask.sum().item(), 32)
            num_neg = min(roi_neg_mask.sum().item(), num_pos * 3)
            
            if num_pos == 0 or num_neg == 0:
                continue
            
            pos_inds = torch.where(roi_pos_mask)[0][:num_pos]
            neg_inds = torch.where(roi_neg_mask)[0][:num_neg]
            
            sampled_inds = torch.cat([pos_inds, neg_inds])
            
            sampled_cls_logits = cls_logits[sampled_inds]
            sampled_gt_labels = torch.zeros(len(sampled_inds), dtype=torch.long, device=device)
            orig_pos_indices = roi_orig_indices[pos_inds]
            sampled_gt_labels[:len(pos_inds)] = gt_class_ids[matched_gt_idx[orig_pos_indices]]
            
            roi_cls_loss = nn.functional.cross_entropy(sampled_cls_logits, sampled_gt_labels)
            
            if len(pos_inds) > 0:
                pos_proposals = proposals[roi_orig_indices[pos_inds]]
                pos_gt_boxes = gt_boxes[matched_gt_idx[roi_orig_indices[pos_inds]]]
                pos_bbox_deltas = bbox_deltas[pos_inds]
                pos_gt_labels_for_bbox = gt_class_ids[matched_gt_idx[roi_orig_indices[pos_inds]]]
                
                target_deltas = encode_boxes(pos_gt_boxes, pos_proposals)
                
                selected_bbox_deltas = torch.zeros_like(target_deltas)
                for cls_idx in range(model.num_classes):
                    mask = (pos_gt_labels_for_bbox == cls_idx)
                    if mask.sum() > 0:
                        start = cls_idx * 4
                        end = start + 4
                        selected_bbox_deltas[mask] = pos_bbox_deltas[mask, start:end]
                
                roi_bbox_loss = nn.functional.smooth_l1_loss(selected_bbox_deltas, target_deltas, reduction='mean')
            else:
                roi_bbox_loss = torch.tensor(0.0, device=device)
            
            batch_roi_cls_loss += roi_cls_loss
            batch_roi_bbox_loss += roi_bbox_loss
        
        batch_loss = (batch_rpn_cls_loss + batch_rpn_bbox_loss + 
                     batch_roi_cls_loss + batch_roi_bbox_loss) / batch_size
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item() if hasattr(batch_loss, 'item') else batch_loss
        total_rpn_cls += (batch_rpn_cls_loss.item() if hasattr(batch_rpn_cls_loss, 'item') else batch_rpn_cls_loss) / batch_size
        total_rpn_bbox += (batch_rpn_bbox_loss.item() if hasattr(batch_rpn_bbox_loss, 'item') else batch_rpn_bbox_loss) / batch_size
        total_roi_cls += (batch_roi_cls_loss.item() if hasattr(batch_roi_cls_loss, 'item') else batch_roi_cls_loss) / batch_size
        total_roi_bbox += (batch_roi_bbox_loss.item() if hasattr(batch_roi_bbox_loss, 'item') else batch_roi_bbox_loss) / batch_size
        num_batches += 1
    
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
            
            for b in range(batch_size):
                n_obj = num_objects[b].item()
                if n_obj == 0:
                    continue
                
                gt_boxes_xywh = bbox_labels[b, :n_obj, :4]
                gt_boxes = xywh_to_xyxy(gt_boxes_xywh, img_size).to(device)
                gt_class_ids_val = bbox_labels[b, :n_obj, 4].long().to(device)
                
                proposals = []
                
                for i, layer_name in enumerate(['P2', 'P3', 'P4']):
                    anchors = anchors_dict[layer_name]
                    
                    cls_pred = rpn_cls_preds[i][b].flatten()
                    bbox_pred = rpn_bbox_preds[i][b].permute(1, 2, 0).contiguous()
                    bbox_pred = bbox_pred.reshape(-1, 15, 4)
                    bbox_pred = bbox_pred.reshape(-1, 4)
                    
                    cls_loss, bbox_loss, _ = rpn_trainer.compute_loss(
                        cls_pred, bbox_pred, anchors, gt_boxes
                    )
                    
                    batch_rpn_cls_loss += cls_loss
                    batch_rpn_bbox_loss += bbox_loss
                    
                    boxes = model.decode_boxes(anchors, bbox_pred)
                    
                    top_n = min(INFERENCE_CONFIG['rpn_pre_nms_top_n'], cls_pred.size(0))
                    top_scores, top_inds = cls_pred.topk(top_n)
                    top_boxes = boxes[top_inds]
                    
                    keep = nms(top_boxes, top_scores, INFERENCE_CONFIG['rpn_nms_thresh'])
                    post_nms_top_n = min(INFERENCE_CONFIG['rpn_post_nms_top_n'], keep.size(0))
                    keep = keep[:post_nms_top_n]
                    
                    proposals.append(top_boxes[keep])
                
                if len(proposals) > 0:
                    proposals = torch.cat(proposals, dim=0)
                else:
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
                
                if p2_mask.sum() > 0:
                    p2_boxes = roi_boxes[p2_mask]
                    p2_feat = fpn_features[0]
                    p2_roi = roi_align(p2_feat, p2_boxes, output_size=(7, 7), spatial_scale=1.0/4.0)
                    all_roi_features.append(p2_roi)
                    all_indices.append(torch.where(p2_mask)[0])
                
                if p3_mask.sum() > 0:
                    p3_boxes = roi_boxes[p3_mask]
                    p3_feat = fpn_features[1]
                    p3_roi = roi_align(p3_feat, p3_boxes, output_size=(7, 7), spatial_scale=1.0/8.0)
                    all_roi_features.append(p3_roi)
                    all_indices.append(torch.where(p3_mask)[0])
                
                if p4_mask.sum() > 0:
                    p4_boxes = roi_boxes[p4_mask]
                    p4_feat = fpn_features[2]
                    p4_roi = roi_align(p4_feat, p4_boxes, output_size=(7, 7), spatial_scale=1.0/16.0)
                    all_roi_features.append(p4_roi)
                    all_indices.append(torch.where(p4_mask)[0])
                
                if len(all_roi_features) == 0:
                    continue
                
                roi_features = torch.cat(all_roi_features, dim=0)
                roi_orig_indices = torch.cat(all_indices, dim=0)
                
                cls_logits, bbox_deltas = model.roi_head(roi_features)
                
                ious = box_iou(proposals, gt_boxes)
                max_ious, matched_gt_idx = ious.max(dim=1)
                
                pos_mask = max_ious >= 0.5
                neg_mask = (max_ious < 0.3) & (~pos_mask)
                
                roi_pos_mask = pos_mask[roi_orig_indices]
                roi_neg_mask = neg_mask[roi_orig_indices]
                
                num_pos = min(roi_pos_mask.sum().item(), 32)
                num_neg = min(roi_neg_mask.sum().item(), num_pos * 3)
                
                if num_pos > 0 and num_neg > 0:
                    pos_inds = torch.where(roi_pos_mask)[0][:num_pos]
                    neg_inds = torch.where(roi_neg_mask)[0][:num_neg]
                    sampled_inds = torch.cat([pos_inds, neg_inds])
                    
                    sampled_cls_logits = cls_logits[sampled_inds]
                    sampled_gt_labels = torch.zeros(len(sampled_inds), dtype=torch.long, device=device)
                    orig_pos_indices = roi_orig_indices[pos_inds]
                    sampled_gt_labels[:len(pos_inds)] = gt_class_ids_val[matched_gt_idx[orig_pos_indices]]
                    
                    roi_cls_loss = nn.functional.cross_entropy(sampled_cls_logits, sampled_gt_labels)
                    
                    if len(pos_inds) > 0:
                        pos_proposals = proposals[roi_orig_indices[pos_inds]]
                        pos_gt_boxes = gt_boxes[matched_gt_idx[roi_orig_indices[pos_inds]]]
                        pos_bbox_deltas = bbox_deltas[pos_inds]
                        pos_gt_labels_for_bbox = gt_class_ids_val[matched_gt_idx[roi_orig_indices[pos_inds]]]
                        
                        target_deltas = encode_boxes(pos_gt_boxes, pos_proposals)
                        
                        selected_bbox_deltas = torch.zeros_like(target_deltas)
                        for cls_idx in range(model.num_classes):
                            mask = (pos_gt_labels_for_bbox == cls_idx)
                            if mask.sum() > 0:
                                start = cls_idx * 4
                                end = start + 4
                                selected_bbox_deltas[mask] = pos_bbox_deltas[mask, start:end]
                        
                        roi_bbox_loss = nn.functional.smooth_l1_loss(selected_bbox_deltas, target_deltas, reduction='mean')
                    else:
                        roi_bbox_loss = torch.tensor(0.0, device=device)
                    
                    batch_roi_cls_loss += roi_cls_loss
                    batch_roi_bbox_loss += roi_bbox_loss
            
            batch_loss = (batch_rpn_cls_loss + batch_rpn_bbox_loss + 
                         batch_roi_cls_loss + batch_roi_bbox_loss) / batch_size
            
            total_loss += batch_loss.item() if hasattr(batch_loss, 'item') else batch_loss
            total_rpn_cls += (batch_rpn_cls_loss.item() if hasattr(batch_rpn_cls_loss, 'item') else batch_rpn_cls_loss) / batch_size
            total_rpn_bbox += (batch_rpn_bbox_loss.item() if hasattr(batch_rpn_bbox_loss, 'item') else batch_rpn_bbox_loss) / batch_size
            total_roi_cls += (batch_roi_cls_loss.item() if hasattr(batch_roi_cls_loss, 'item') else batch_roi_cls_loss) / batch_size
            total_roi_bbox += (batch_roi_bbox_loss.item() if hasattr(batch_roi_bbox_loss, 'item') else batch_roi_bbox_loss) / batch_size
            num_batches += 1
    
    return (total_loss / max(num_batches, 1),
            total_rpn_cls / max(num_batches, 1),
            total_rpn_bbox / max(num_batches, 1),
            total_roi_cls / max(num_batches, 1),
            total_roi_bbox / max(num_batches, 1))


def train_model(train_loader, val_loader, model, epochs):
    """训练两阶段检测器。"""
    rpn_trainer = RPNTrainer()
    tracker = TrainingTracker()
    best_loss = float('inf')
    
    # ============================================================
    # 阶段1: 冻结backbone
    # ============================================================
    print("\n" + "=" * 60)
    print("阶段1: 冻结backbone，训练RPN和检测头")
    print("=" * 60)
    
    freeze_backbone(model)
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TRAIN_CONFIG['stage1_lr'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    scheduler = get_lr_scheduler(optimizer, stage=1, epochs=TRAIN_CONFIG['stage1_epochs'])
    
    pbar = tqdm(range(TRAIN_CONFIG['stage1_epochs']), desc='阶段1')
    
    for epoch in pbar:
        train_loss, rpn_cls, rpn_bbox, roi_cls, roi_bbox = train_one_epoch(
            model, train_loader, optimizer, rpn_trainer, 
            config.device, config.img_size
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        val_loss, val_rpn_cls, val_rpn_bbox, val_roi_cls, val_roi_bbox = validate(
            model, val_loader, rpn_trainer, config.device, config.img_size
        )
        
        tracker.add(train_loss, val_loss, rpn_cls, rpn_bbox, current_lr, stage=1)
        
        pbar.set_postfix({
            'loss': f'{val_loss:.4f}',
            'rpn_cls': f'{rpn_cls:.4f}',
            'rpn_bbox': f'{rpn_bbox:.4f}',
            'roi_cls': f'{roi_cls:.4f}',
            'roi_bbox': f'{roi_bbox:.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), config.model_save_path)
    
    # ============================================================
    # 阶段2: 解冻backbone
    # ============================================================
    print("\n" + "=" * 60)
    print("阶段2: 解冻backbone，微调整个模型")
    print("=" * 60)
    
    unfreeze_backbone(model)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['stage2_lr'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    scheduler = get_lr_scheduler(optimizer, stage=2, epochs=TRAIN_CONFIG['stage2_epochs'])
    early_stopping = EarlyStopping(patience=TRAIN_CONFIG['early_stop_patience'])
    
    pbar = tqdm(range(TRAIN_CONFIG['stage2_epochs']), desc='阶段2')
    
    for epoch in pbar:
        train_loss, rpn_cls, rpn_bbox, roi_cls, roi_bbox = train_one_epoch(
            model, train_loader, optimizer, rpn_trainer,
            config.device, config.img_size
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        val_loss, val_rpn_cls, val_rpn_bbox, val_roi_cls, val_roi_bbox = validate(
            model, val_loader, rpn_trainer, config.device, config.img_size
        )
        
        tracker.add(train_loss, val_loss, rpn_cls, rpn_bbox, current_lr, stage=2)
        
        pbar.set_postfix({
            'loss': f'{val_loss:.4f}',
            'rpn_cls': f'{rpn_cls:.4f}',
            'rpn_bbox': f'{rpn_bbox:.4f}',
            'roi_cls': f'{roi_cls:.4f}',
            'roi_bbox': f'{roi_bbox:.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), config.model_save_path)
        
        if early_stopping(val_loss):
            print(f"\n早停！验证损失连续{TRAIN_CONFIG['early_stop_patience']}轮没有改善")
            break
    
    logger.info(f"最佳验证损失: {best_loss:.4f}")
    return model, tracker


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数。"""
    # 设置随机种子
    config.set_seed()
    exp_dir = config.setup_experiment()
    config.print_config()
    
    # 打印训练配置
    print("\n" + "=" * 60)
    print("两阶段检测器训练配置")
    print("=" * 60)
    print(f"  阶段1: Epoch 1~{TRAIN_CONFIG['stage1_epochs']}, 冻结backbone, lr={TRAIN_CONFIG['stage1_lr']}")
    print(f"  阶段2: Epoch {TRAIN_CONFIG['stage1_epochs']+1}~{TRAIN_CONFIG['stage1_epochs']+TRAIN_CONFIG['stage2_epochs']}, 解冻backbone, lr={TRAIN_CONFIG['stage2_lr']}")
    print(f"  早停容忍: {TRAIN_CONFIG['early_stop_patience']} 轮")
    print("\n[FPN配置]")
    for layer, cfg in FPN_CONFIG['anchor_configs'].items():
        num_anchors = len(cfg['scales']) * len(cfg['ratios'])
        print(f"  {layer}: stride={cfg['stride']}, anchors/位置={num_anchors}")
    print("\n[损失函数]")
    print(f"  RPN Focal Loss: α=0.25, γ=2.0")
    print(f"  RPN GIoU Loss")
    print("=" * 60)
    
    # 加载数据
    logger.info("正在加载数据集...")
    train_loader = create_dataloader(config.train_img_dir, config.train_label_dir, mode='train')
    val_loader = create_dataloader(config.val_img_dir, config.val_label_dir, mode='val')
    
    logger.info(f"训练集: {len(train_loader.dataset)}张, 验证集: {len(val_loader.dataset)}张")
    
    # 创建模型
    model = TwoStageDetector(num_classes=len(config.class_names)).to(config.device)
    model.load_pretrained_weights()
    
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    logger.info("开始训练...")
    model, tracker = train_model(train_loader, val_loader, model, None)
    
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
