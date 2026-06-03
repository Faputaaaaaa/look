import numpy as np
from typing import List, Tuple, Optional
import torch
from torch import Tensor
import logging


# 配置日志
def setup_logger(name: str = "defect_detection") -> logging.Logger:
    """设置日志记录器。
    
    Args:
        name: 日志器名称
    
    Returns:
        logging.Logger: 配置好的日志器
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)


logger = setup_logger()


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> List[int]:
    """非极大值抑制。
    
    用于去除重叠的预测框，保留最优结果。
    
    算法流程：
        1. 按置信度降序排列所有框
        2. 选择置信度最高的框作为当前框
        3. 计算当前框与其余框的IoU
        4. 移除IoU超过阈值的框（认为是重复检测）
        5. 重复步骤2-4直到所有框处理完毕
    
    Args:
        boxes: 边界框数组，形状 [N, 4]，格式 [cx, cy, w, h]，归一化坐标
        scores: 置信度数组，形状 [N,]
        threshold: IoU阈值，超过此值的框将被抑制
    
    Returns:
        List[int]: 保留的边界框索引列表
    """
    if len(boxes) == 0:
        return []

    # 转换为 [x1, y1, x2, y2] 格式
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    areas = (x2 - x1) * (y2 - y1)
    indices = np.argsort(scores)[::-1]  # 按置信度降序排列

    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # 计算当前框与其余框的IoU
        current_box = [x1[current], y1[current], x2[current], y2[current]]
        other_indices = indices[1:]

        xx1 = np.maximum(current_box[0], x1[other_indices])
        yy1 = np.maximum(current_box[1], y1[other_indices])
        xx2 = np.minimum(current_box[2], x2[other_indices])
        yy2 = np.minimum(current_box[3], y2[other_indices])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = areas[current] + areas[other_indices] - inter_area
        ious = inter_area / union_area

        # 保留IoU小的
        mask = ious < threshold
        indices = other_indices[mask]

    return keep


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray, img_size: int) -> float:
    """计算两个边界框的交并比。
    
    Args:
        bbox1: 第一个边界框，格式为 [cx, cy, w, h]，归一化坐标
        bbox2: 第二个边界框，格式同上
        img_size: 图像尺寸（像素），用于坐标转换
    
    Returns:
        float: IoU值，范围 [0, 1]
    
    Example:
        >>> bbox1 = np.array([0.5, 0.5, 0.2, 0.2])
        >>> bbox2 = np.array([0.5, 0.5, 0.3, 0.3])
        >>> iou = calculate_iou(bbox1, bbox2, 224)
        >>> print(f"IoU: {iou:.3f}")
    """
    def norm2pixel(bbox: np.ndarray, size: int) -> np.ndarray:
        """将归一化坐标转换为像素坐标"""
        cx, cy, w, h = bbox
        xmin = (cx - w / 2) * size
        ymin = (cy - h / 2) * size
        xmax = (cx + w / 2) * size
        ymax = (cy + h / 2) * size
        return np.clip([xmin, ymin, xmax, ymax], 0, size)

    x1, y1, x2, y2 = norm2pixel(bbox1, img_size)
    x1t, y1t, x2t, y2t = norm2pixel(bbox2, img_size)

    inter_x1 = max(x1, x1t)
    inter_y1 = max(y1, y1t)
    inter_x2 = min(x2, x2t)
    inter_y2 = min(y2, y2t)
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)

    pred_area = (x2 - x1) * (y2 - y1)
    true_area = (x2t - x1t) * (y2t - y1t)
    union_area = pred_area + true_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def decode_boxes(anchors: Tensor, pred_deltas: Tensor) -> Tensor:
    """将anchor和预测偏移量解码为最终边界框。
    
    Args:
        anchors: Anchor框，形状 [N, 4]，格式 [x1, y1, x2, y2]
        pred_deltas: 预测偏移量，形状 [1, 4*N, H, W] 或 [N, 4]
    
    Returns:
        Tensor: 解码后的边界框，形状 [N, 4]，格式 [x1, y1, x2, y2]
    """
    # 处理输入形状
    if pred_deltas.dim() == 4:
        pred_deltas = pred_deltas.view(-1, 4)

    # 中心点 + 宽高格式转换
    anchor_centers_x = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_centers_y = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]

    # 应用偏移
    pred_centers_x = anchor_centers_x + pred_deltas[:, 0] * anchor_w
    pred_centers_y = anchor_centers_y + pred_deltas[:, 1] * anchor_h
    pred_w = anchor_w * torch.exp(pred_deltas[:, 2])
    pred_h = anchor_h * torch.exp(pred_deltas[:, 3])

    # 转回 x1, y1, x2, y2
    x1 = pred_centers_x - pred_w / 2
    y1 = pred_centers_y - pred_h / 2
    x2 = pred_centers_x + pred_w / 2
    y2 = pred_centers_y + pred_h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def clip_boxes(boxes: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """裁剪边界框到图像范围内。
    
    Args:
        boxes: 边界框数组，形状 [N, 4]，格式 [x1, y1, x2, y2]
        img_size: 图像尺寸 (width, height)
    
    Returns:
        np.ndarray: 裁剪后的边界框
    """
    width, height = img_size
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height)
    return boxes


def normalize_boxes(boxes: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """将边界框坐标归一化到 [0, 1] 范围。
    
    Args:
        boxes: 边界框数组，形状 [N, 4]，格式 [x1, y1, x2, y2]
        img_size: 图像尺寸 (width, height)
    
    Returns:
        np.ndarray: 归一化后的边界框
    """
    width, height = img_size
    boxes[:, 0] /= width
    boxes[:, 1] /= height
    boxes[:, 2] /= width
    boxes[:, 3] /= height
    return boxes


def denormalize_boxes(boxes: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """将归一化边界框转换回像素坐标。
    
    Args:
        boxes: 边界框数组，形状 [N, 4]，格式 [x1, y1, x2, y2]
        img_size: 图像尺寸 (width, height)
    
    Returns:
        np.ndarray: 像素坐标的边界框
    """
    width, height = img_size
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height
    return boxes
