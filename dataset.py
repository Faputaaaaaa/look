"""
数据处理模块（支持VOC格式和YOLO格式）

功能：
1. 支持VOC格式(XML)和YOLO格式(TXT)标签
2. 多目标支持
3. 智能裁剪、滑动窗口、直接缩放三种预处理方法
4. 类别过滤

使用方法：
    from dataset import DefectDataset, create_dataloader
    
    train_loader = create_dataloader(config.train_img_dir, config.train_label_dir, mode='train')
"""

import os
import random
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import config


class DefectDataset(Dataset):
    """缺陷检测数据集类。
    
    支持：
    1. VOC格式(XML)和YOLO格式(TXT)标签
    2. 多目标检测
    3. 三种预处理方法: smart_crop / sliding_window / resize
    
    Attributes:
        img_dir: 图像目录
        label_dir: 标签目录
        mode: 模式 ('train' / 'val' / 'test')
        transform: 图像变换
        preprocess_method: 预处理方法
        label_format: 标签格式 ('voc' / 'yolo')
    """
    
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        mode: str = 'train',
        transform: Optional[transforms.Compose] = None,
        preprocess_method: str = None,
        label_format: str = None
    ) -> None:
        """初始化数据集。
        
        Args:
            img_dir: 图像目录
            label_dir: 标签目录
            mode: 模式
            transform: 图像变换
            preprocess_method: 预处理方法
            label_format: 标签格式
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.mode = mode
        self.transform = transform  # 保持兼容性
        # 新增：分离颜色增强和归一化
        self.color_transform = getattr(transform, 'color_transform', None) if transform else None
        self.normalize_transform = getattr(transform, 'normalize_transform', None) if transform else None
        self.preprocess_method = preprocess_method or config.preprocess_method
        self.label_format = label_format or config.label_format
        
        # 从config获取参数
        self.img_size = config.img_size
        self.window_size = config.window_size
        self.window_stride = config.window_stride
        self.crop_padding = config.crop_padding
        self.max_objects = config.max_objects
        self.class_name_map = config.class_name_map
        
        # 加载数据
        self.samples = self._load_samples()
        self.filtered_count = 0
        
        print(f"[Dataset] 加载了 {len(self.samples)} 个样本，模式: {self.mode}，方法: {self.preprocess_method}，格式: {self.label_format}")
    
    def _load_samples(self) -> List[Dict]:
        """加载样本。
        
        Returns:
            List[Dict]: 样本列表
        """
        samples = []
        
        # 获取所有图片
        img_files = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        
        for img_file in img_files:
            # 查找对应的标签文件
            if self.label_format == "voc":
                label_file = self.label_dir / f"{img_file.stem}.xml"
            else:
                label_file = self.label_dir / f"{img_file.stem}.txt"
            
            # 读取标签
            bboxes = []
            if label_file.exists():
                if self.label_format == "voc":
                    bboxes = self._read_voc_labels(label_file)
                else:
                    bboxes = self._read_yolo_labels(label_file)
            
            if self.preprocess_method == "smart_crop":
                # 智能裁剪：每个缺陷生成一个样本
                if len(bboxes) > 0:
                    for i, bbox in enumerate(bboxes):
                        samples.append({
                            'img_path': str(img_file),
                            'label_path': str(label_file),
                            'bbox_idx': i,
                            'all_bboxes': bboxes,
                            'target_bbox': bbox
                        })
                else:
                    # 无缺陷图片，随机裁剪
                    samples.append({
                        'img_path': str(img_file),
                        'label_path': str(label_file),
                        'bbox_idx': -1,
                        'all_bboxes': [],
                        'target_bbox': None
                    })
            
            elif self.preprocess_method == "sliding_window":
                # 滑动窗口：生成所有窗口位置
                windows = self._generate_windows(img_file)
                for window in windows:
                    window_bboxes = self._get_bboxes_in_window(bboxes, window)
                    samples.append({
                        'img_path': str(img_file),
                        'label_path': str(label_file),
                        'window': window,
                        'all_bboxes': window_bboxes
                    })
            
            else:  # resize
                samples.append({
                    'img_path': str(img_file),
                    'label_path': str(label_file),
                    'all_bboxes': bboxes
                })
        
        return samples
    
    def _read_voc_labels(self, label_path: Path) -> List[Dict]:
        """读取VOC格式XML标签。
        
        Args:
            label_path: 标签文件路径
        
        Returns:
            List[Dict]: 标签列表
        """
        bboxes = []
        try:
            tree = ET.parse(label_path)
            root = tree.getroot()
            
            # 获取图片尺寸
            size = root.find('size')
            if size is not None:
                img_w = int(size.find('width').text)
                img_h = int(size.find('height').text)
            else:
                img_w = config.original_img_width
                img_h = config.original_img_height
            
            # 解析每个目标
            for obj in root.findall('object'):
                name = obj.find('name').text
                
                # 类别映射
                if name in self.class_name_map:
                    class_id = self.class_name_map[name]
                else:
                    continue  # 跳过不在目标类别中的缺陷
                
                # 边界框
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # 转换为YOLO格式（归一化的中心坐标和宽高）
                cx = (xmin + xmax) / 2 / img_w
                cy = (ymin + ymax) / 2 / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h
                
                # 验证边界框
                if w > 0 and h > 0 and 0 <= cx <= 1 and 0 <= cy <= 1:
                    bboxes.append({
                        'class_id': class_id,
                        'cx': cx,
                        'cy': cy,
                        'w': w,
                        'h': h,
                        'name': name
                    })
        
        except Exception as e:
            print(f"[Dataset] 读取VOC标签失败: {label_path}, 错误: {e}")
        
        return bboxes
    
    def _read_yolo_labels(self, label_path: Path) -> List[Dict]:
        """读取YOLO格式TXT标签。
        
        Args:
            label_path: 标签文件路径
        
        Returns:
            List[Dict]: 标签列表
        """
        bboxes = []
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        if w > 0 and h > 0 and 0 <= cx <= 1 and 0 <= cy <= 1:
                            bboxes.append({
                                'class_id': class_id,
                                'cx': cx,
                                'cy': cy,
                                'w': w,
                                'h': h
                            })
        except Exception as e:
            print(f"[Dataset] 读取YOLO标签失败: {label_path}, 错误: {e}")
        
        return bboxes
    
    def _generate_windows(self, img_path: Path) -> List[Dict]:
        """生成滑动窗口位置。"""
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except:
            img_w, img_h = config.original_img_width, config.original_img_height
        
        windows = []
        
        for y in range(0, img_h - self.window_size + 1, self.window_stride):
            for x in range(0, img_w - self.window_size + 1, self.window_stride):
                windows.append({
                    'x': x,
                    'y': y,
                    'w': self.window_size,
                    'h': self.window_size
                })
        
        # 处理边界
        if img_w > self.window_size:
            x = img_w - self.window_size
            for y in range(0, img_h - self.window_size + 1, self.window_stride):
                windows.append({'x': x, 'y': y, 'w': self.window_size, 'h': self.window_size})
        
        if img_h > self.window_size:
            y = img_h - self.window_size
            for x in range(0, img_w - self.window_size + 1, self.window_stride):
                windows.append({'x': x, 'y': y, 'w': self.window_size, 'h': self.window_size})
        
        return windows
    
    def _get_bboxes_in_window(self, bboxes: List[Dict], window: Dict) -> List[Dict]:
        """获取窗口内的边界框。"""
        img_w = config.original_img_width
        img_h = config.original_img_height
        
        window_bboxes = []
        
        for bbox in bboxes:
            cx_px = bbox['cx'] * img_w
            cy_px = bbox['cy'] * img_h
            w_px = bbox['w'] * img_w
            h_px = bbox['h'] * img_h
            
            if (window['x'] <= cx_px <= window['x'] + window['w'] and
                window['y'] <= cy_px <= window['y'] + window['h']):
                
                new_cx = (cx_px - window['x']) / window['w']
                new_cy = (cy_px - window['y']) / window['h']
                new_w = w_px / window['w']
                new_h = h_px / window['h']
                
                window_bboxes.append({
                    'class_id': bbox['class_id'],
                    'cx': new_cx,
                    'cy': new_cy,
                    'w': new_w,
                    'h': new_h,
                    'original_bbox': bbox
                })
        
        return window_bboxes
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """获取样本。
        
        Returns:
            Tuple: (image, num_objects, bboxes, meta)
            - image: [3, H, W] 图像张量
            - num_objects: 目标数量
            - bboxes: [max_objects, 5] 边界框 (cx, cy, w, h, class_id)
            - meta: 元信息字典
        """
        sample = self.samples[idx]
        
        # 加载图片（保持RGB三通道）
        img = Image.open(sample['img_path']).convert('RGB')
        img_w, img_h = img.size
        
        if self.preprocess_method == "smart_crop":
            return self._process_smart_crop(img, sample, img_w, img_h)
        elif self.preprocess_method == "sliding_window":
            return self._process_sliding_window(img, sample)
        else:
            return self._process_resize(img, sample, img_w, img_h)
    
    def _process_smart_crop(
        self, 
        img: Image.Image, 
        sample: Dict, 
        img_w: int, 
        img_h: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """智能裁剪处理。"""
        if sample['target_bbox'] is not None:
            bbox = sample['target_bbox']
            cx_px = bbox['cx'] * img_w
            cy_px = bbox['cy'] * img_h
            
            crop_size = max(self.img_size, int(max(bbox['w'] * img_w, bbox['h'] * img_h) * 2))
            crop_size = min(crop_size, min(img_w, img_h))
            
            left = max(0, int(cx_px - crop_size // 2))
            top = max(0, int(cy_px - crop_size // 2))
            right = min(img_w, left + crop_size)
            bottom = min(img_h, top + crop_size)
            
            if right - left < crop_size:
                left = max(0, right - crop_size)
            if bottom - top < crop_size:
                top = max(0, bottom - crop_size)
            
            img_cropped = img.crop((left, top, right, bottom))
            actual_crop_w = right - left
            actual_crop_h = bottom - top
            
            new_cx = (cx_px - left) / actual_crop_w
            new_cy = (cy_px - top) / actual_crop_h
            new_w = bbox['w'] * img_w / actual_crop_w
            new_h = bbox['h'] * img_h / actual_crop_h
            
            bboxes = [[new_cx, new_cy, new_w, new_h, bbox['class_id']]]
            num_objects = 1
        else:
            crop_size = self.img_size
            left = random.randint(0, max(0, img_w - crop_size))
            top = random.randint(0, max(0, img_h - crop_size))
            right = min(img_w, left + crop_size)
            bottom = min(img_h, top + crop_size)
            
            img_cropped = img.crop((left, top, right, bottom))
            bboxes = []
            num_objects = 0
        
        img_resized = img_cropped.resize((self.img_size, self.img_size))
        
        if len(bboxes) > 0:
            actual_w = right - left
            actual_h = bottom - top
            bboxes[0][2] = bboxes[0][2] * actual_w / self.img_size
            bboxes[0][3] = bboxes[0][3] * actual_h / self.img_size
        
        # 使用新的transform结构
        if hasattr(self, 'normalize_transform') and self.normalize_transform:
            img_tensor = self.normalize_transform(img_resized)
        elif self.transform:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)
        
        bboxes_padded = self._pad_bboxes(bboxes)
        
        meta = {
            'img_path': sample['img_path'],
            'crop_region': (left, top, right, bottom),
            'original_size': (img_w, img_h)
        }
        
        return img_tensor, torch.tensor(num_objects, dtype=torch.long), bboxes_padded, meta
    
    def _process_sliding_window(
        self, 
        img: Image.Image, 
        sample: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """滑动窗口处理。"""
        window = sample['window']
        
        img_cropped = img.crop((
            window['x'], 
            window['y'], 
            window['x'] + window['w'], 
            window['y'] + window['h']
        ))
        
        img_resized = img_cropped.resize((self.img_size, self.img_size))
        
        # 使用新的transform结构
        if hasattr(self, 'normalize_transform') and self.normalize_transform:
            img_tensor = self.normalize_transform(img_resized)
        elif self.transform:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)
        
        bboxes = []
        for bbox in sample['all_bboxes']:
            bboxes.append([bbox['cx'], bbox['cy'], bbox['w'], bbox['h'], bbox['class_id']])
        
        num_objects = len(bboxes)
        bboxes_padded = self._pad_bboxes(bboxes)
        
        meta = {
            'img_path': sample['img_path'],
            'window': window,
            'original_size': img.size
        }
        
        return img_tensor, torch.tensor(num_objects, dtype=torch.long), bboxes_padded, meta
    
    def _process_resize(
        self, 
        img: Image.Image, 
        sample: Dict, 
        img_w: int, 
        img_h: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """直接缩放处理（包含数据增强和标注框同步）。
        
        关键改进：几何变换（如水平翻转）需要同步更新标注框坐标！
        """
        # 1. 准备原始标注框
        bboxes = []
        for bbox in sample['all_bboxes']:
            bboxes.append([bbox['cx'], bbox['cy'], bbox['w'], bbox['h'], bbox['class_id']])
        num_objects = len(bboxes)
        
        # 2. 缩放图像到目标尺寸
        img_resized = img.resize((self.img_size, self.img_size))
        
        # 3. 【关键】手动处理几何增强 + 标注框同步
        if self.mode == 'train' and config.use_augmentation:
            # 3.1 水平翻转（需要同步更新标注框的cx）
            if random.random() < 0.5:  # p=0.5
                img_resized = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
                # 同步翻转所有标注框: cx → 1 - cx
                bboxes = [[1.0 - b[0], b[1], b[2], b[3], b[4]] for b in bboxes]
            
            # 3.2 颜色增强（不影响位置，不需要同步标注框）
            if self.color_transform:
                img_resized = self.color_transform(img_resized)
        
        # 4. 转换为Tensor并归一化
        if self.normalize_transform:
            img_tensor = self.normalize_transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)
        
        # 5. 填充标注框到固定大小
        bboxes_padded = self._pad_bboxes(bboxes)
        
        meta = {
            'img_path': sample['img_path'],
            'original_size': (img_w, img_h)
        }
        
        return img_tensor, torch.tensor(num_objects, dtype=torch.long), bboxes_padded, meta
    
    def _pad_bboxes(self, bboxes: List[List]) -> torch.Tensor:
        """填充边界框到固定大小。
        
        Returns:
            torch.Tensor: [max_objects, 5] (cx, cy, w, h, class_id)
        """
        padded = np.zeros((self.max_objects, 5), dtype=np.float32)
        
        num_bboxes = min(len(bboxes), self.max_objects)
        if num_bboxes > 0:
            padded[:num_bboxes] = bboxes[:num_bboxes]
        
        return torch.tensor(padded, dtype=torch.float32)


def create_dataloader(
    img_dir: str,
    label_dir: str,
    mode: str = 'train',
    batch_size: int = None,
    num_workers: int = None,
    preprocess_method: str = None,
    label_format: str = None
) -> torch.utils.data.DataLoader:
    """创建数据加载器。
    
    Args:
        img_dir: 图像目录
        label_dir: 标签目录
        mode: 模式
        batch_size: 批量大小
        num_workers: 工作进程数
        preprocess_method: 预处理方法
        label_format: 标签格式
    
    Returns:
        DataLoader: 数据加载器
    """
    batch_size = batch_size or config.batch_size
    num_workers = num_workers or config.num_workers
    
    # 数据增强（针对灰度图优化 + 标注框同步）
    if mode == 'train' and config.use_augmentation:
        # 颜色增强（不影响位置，不需要同步标注框）
        color_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.15,    # 亮度±15%（灰度图有效）
                contrast=0.15,      # 对比度±15%（灰度图最有效！）
                saturation=0,       # 灰度图无饱和度
                hue=0               # 灰度图无色调
            ),
        ])
        
        # 归一化（ToTensor + Normalize）
        normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建一个包装对象传递给Dataset
        transform = type('Transform', (), {
            'color_transform': color_transform,
            'normalize_transform': normalize_transform
        })()
    else:
        # 验证/测试模式：只做归一化
        normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform = type('Transform', (), {
            'color_transform': None,
            'normalize_transform': normalize_transform
        })()
    
    dataset = DefectDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        mode=mode,
        transform=transform,
        preprocess_method=preprocess_method,
        label_format=label_format
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=config.pin_memory
    )
    
    return dataloader


if __name__ == "__main__":
    print("测试数据集加载:")
    print(f"训练集路径: {config.train_img_dir}")
    print(f"标签路径: {config.train_label_dir}")
    
    train_loader = create_dataloader(
        config.train_img_dir, 
        config.train_label_dir, 
        mode='train'
    )
    
    for img, num_obj, bboxes, meta in train_loader:
        print(f"\n图像形状: {img.shape}")
        print(f"缺陷数量: {num_obj}")
        print(f"边界框形状: {bboxes.shape}")
        print(f"边界框内容:\n{bboxes[0]}")
        print(f"元信息: {meta['img_path'][0]}")
        break
