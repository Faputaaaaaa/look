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


# [R40] 已删除以下冗余函数（项目实际未使用，统一改用 model_fpn / torchvision 实现）：
#   - nms           → 实际使用 torchvision.ops.nms (train.py, sahi_inference.py)
#   - decode_boxes  → 实际使用 TwoStageDetector.decode_boxes (model_fpn.py:668，带 clamp)
#                    那个版本对 exp(deltas) 做 clamp(-4, 3) 防止数值爆炸，是项目实际生效的版本。


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray, img_size: int) -> float:
    """计算两个边界框的交并比。

    Args:
        bbox1: 第一个边界框，格式为 [cx, cy, w, h]，归一化坐标
        bbox2: 第二个边界框，格式同上
        img_size: 图像尺寸（像素），用于坐标转换

    Returns:
        float: IoU值，范围 [0, 1]
    """
    def norm2pixel(bbox: np.ndarray, size: int) -> np.ndarray:
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
    """将边界框坐标归一化到 [0, 1] 范围。"""
    width, height = img_size
    boxes[:, 0] /= width
    boxes[:, 1] /= height
    boxes[:, 2] /= width
    boxes[:, 3] /= height
    return boxes


def denormalize_boxes(boxes: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """将归一化边界框转换回像素坐标。"""
    width, height = img_size
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height
    return boxes
