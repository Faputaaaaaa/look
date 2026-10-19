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
        img_dir,
        label_dir,
        mode: str = 'train',
        transform: Optional[transforms.Compose] = None,
        preprocess_method: str = None,
        label_format: str = None,
        max_samples_per_dir: int = None,
        from_end: bool = False,
    ) -> None:
        """初始化数据集。
        
        Args:
            img_dir: 图像目录（str或list[str]，支持多目录合并）
            label_dir: 标签目录（str或list[str]，与img_dir对应）
            mode: 模式
            transform: 图像变换
            preprocess_method: 预处理方法
            label_format: 标签格式
            max_samples_per_dir: 每个目录最多加载的样本数（None表示全部）
            from_end: 是否从末尾开始取样本（用于验证集，避免与训练集重叠）
        """
        # 支持单目录或多目录
        if isinstance(img_dir, (list, tuple)):
            self.img_dirs = [Path(d) for d in img_dir]
            self.label_dirs = [Path(d) for d in label_dir]
        else:
            self.img_dirs = [Path(img_dir)]
            self.label_dirs = [Path(label_dir)]
        self.mode = mode
        self.transform = transform  # 保持兼容性
        # 新增：分离颜色增强和归一化
        self.color_transform = getattr(transform, 'color_transform', None) if transform else None
        self.normalize_transform = getattr(transform, 'normalize_transform', None) if transform else None
        self.preprocess_method = preprocess_method or config.preprocess_method
        self.label_format = label_format or config.label_format
        self.max_samples_per_dir = max_samples_per_dir
        self.from_end = from_end
        
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
        
        for dir_idx, (img_dir, label_dir) in enumerate(zip(self.img_dirs, self.label_dirs)):
            # 获取所有图片并排序（确保可复现）
            img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            
            # 限制每个目录的样本数
            if self.max_samples_per_dir is not None:
                if self.from_end:
                    img_files = img_files[-self.max_samples_per_dir:]
                else:
                    img_files = img_files[:self.max_samples_per_dir]
            
            for img_file in img_files:
                # 查找对应的标签文件
                if self.label_format == "voc":
                    label_file = label_dir / f"{img_file.stem}.xml"
                else:
                    label_file = label_dir / f"{img_file.stem}.txt"
                
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
                    # [R35] SAHI滑动窗口：含缺陷窗口全保留 + 背景窗口限量
                    # 原图4096×2736，1024切片+stride512产生~35窗口/图
                    # 但每图仅1个缺陷，含缺陷窗口约2-4个，其余为纯背景
                    # 背景>缺陷会淹没RPN正样本，故每图限量2个背景窗口
                    windows = self._generate_windows(img_file)
                    pos_window_samples = []
                    neg_window_samples = []
                    for window in windows:
                        window_bboxes = self._get_bboxes_in_window(bboxes, window)
                        sample_item = {
                            'img_path': str(img_file),
                            'label_path': str(label_file),
                            'window': window,
                            'all_bboxes': window_bboxes
                        }
                        if len(window_bboxes) > 0:
                            pos_window_samples.append(sample_item)
                        else:
                            neg_window_samples.append(sample_item)
                    # 含缺陷窗口全保留
                    samples.extend(pos_window_samples)
                    # 背景窗口限量：每图最多2个（随机采样）
                    max_neg_per_img = 2
                    if len(neg_window_samples) > max_neg_per_img:
                        neg_window_samples = random.sample(neg_window_samples, max_neg_per_img)
                    samples.extend(neg_window_samples)
                
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

        # [A9] YOLO 数字 ID → 类名 → 模型 class_id 的映射（单一事实来源在 config.yolo_id_to_name）
        yolo_id_to_name = config.yolo_id_to_name

        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        yolo_class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])

                        if w > 0 and h > 0 and 0 <= cx <= 1 and 0 <= cy <= 1:
                            # 转换为类名，再通过class_name_map转为模型class_id
                            class_name = yolo_id_to_name.get(yolo_class_id)
                            if class_name and class_name in self.class_name_map:
                                model_class_id = self.class_name_map[class_name]
                            else:
                                continue  # 跳过不在目标类别中的缺陷

                            bboxes.append({
                                'class_id': model_class_id,
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
        """获取窗口内的边界框，并把 GT clip 到窗口边界 [0, window_size]。

        [A10] 修复：之前只判断缺陷中心在窗口内，但记录完整宽高不 clip，
        导致贴边缺陷的 GT 越界（x1<0 / w>1）。现在：
        1. 把 GT 在窗口像素坐标下 clip 到 [0, window_w]×[0, window_h]。
        2. clip 后宽或高 < 4px 的 GT 丢弃（模型输入 640 空间下 4px 已不可学）。
        """
        img_w = config.original_img_width
        img_h = config.original_img_height
        win_w = window['w']
        win_h = window['h']
        MIN_PX = 4  # clip 后最小尺寸阈值（窗口像素 = 模型输入像素）

        window_bboxes = []

        for bbox in bboxes:
            cx_px = bbox['cx'] * img_w
            cy_px = bbox['cy'] * img_h
            w_px = bbox['w'] * img_w
            h_px = bbox['h'] * img_h

            if (window['x'] <= cx_px <= window['x'] + win_w and
                window['y'] <= cy_px <= window['y'] + win_h):

                # 窗口相对像素坐标（未 clip）
                x1 = cx_px - w_px / 2 - window['x']
                y1 = cy_px - h_px / 2 - window['y']
                x2 = cx_px + w_px / 2 - window['x']
                y2 = cy_px + h_px / 2 - window['y']

                # [A10] clip 到窗口边界
                x1 = max(0.0, x1)
                y1 = max(0.0, y1)
                x2 = min(float(win_w), x2)
                y2 = min(float(win_h), y2)

                clip_w = x2 - x1
                clip_h = y2 - y1
                if clip_w < MIN_PX or clip_h < MIN_PX:
                    continue  # clip 后太小，丢弃

                new_cx = (x1 + x2) / 2 / win_w
                new_cy = (y1 + y2) / 2 / win_h
                new_w = clip_w / win_w
                new_h = clip_h / win_h

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
        """滑动窗口处理。

        [R40] 启用数据增强（仅bbox-safe操作：水平翻转 + ColorJitter）。
        - 不启用RandomCrop：dirty占切片43%，crop后可能切掉缺陷
        - 不启用马赛克：4图拼接会让dirty占满整图，破坏缺陷尺度分布
        - 翻转和颜色增强不破坏bbox坐标（仅翻转需对称翻转cx/cy）
        """
        window = sample['window']

        img_cropped = img.crop((
            window['x'],
            window['y'],
            window['x'] + window['w'],
            window['y'] + window['h']
        ))

        img_resized = img_cropped.resize((self.img_size, self.img_size))

        # 收集bbox（归一化坐标，0~1相对切片）
        bboxes = []
        for bbox in sample['all_bboxes']:
            bboxes.append([bbox['cx'], bbox['cy'], bbox['w'], bbox['h'], bbox['class_id']])
        num_objects = len(bboxes)

        # [R40] 训练模式启用bbox-safe数据增强
        # 注意：滑动窗口路径下bbox已经是0~1归一化坐标，翻转操作直接镜像cx/cy即可
        if self.mode == 'train' and config.use_augmentation:
            # 水平翻转 p=0.5
            if random.random() < 0.5:
                img_resized = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
                bboxes = [[1.0 - b[0], b[1], b[2], b[3], b[4]] for b in bboxes]

            # 颜色增强（不影响bbox）
            if self.color_transform:
                img_resized = self.color_transform(img_resized)

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
        """Letterbox预处理（保持宽高比，消除形状畸变）。
        
        将原始图像按比例缩放到目标尺寸内，短边贴合，
        长边空白区域用灰色(128)填充。bbox坐标同步变换。
        """
        # 1. 准备原始标注框（归一化坐标 cx,cy,w,h,class_id）
        bboxes = []
        for bbox in sample['all_bboxes']:
            bboxes.append([bbox['cx'], bbox['cy'], bbox['w'], bbox['h'], bbox['class_id']])
        num_objects = len(bboxes)
        
        # 2. Letterbox resize：保持宽高比
        iw, ih = img.size  # 原始尺寸
        w, h = self.img_size, self.img_size  # 目标尺寸(正方形)
        scale = min(w / iw, h / ih)  # 缩放比例（取较小值保持比例）
        nw, nh = int(iw * scale), int(ih * scale)  # 缩放后的实际尺寸
        
        img_resized = img.resize((nw, nh), Image.BILINEAR)
        
        # 创建灰色画布并粘贴到中心
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        offset_x = (w - nw) // 2
        offset_y = (h - nh) // 2
        new_img.paste(img_resized, (offset_x, offset_y))
        img_resized = new_img
        
        # 3. 同步变换bbox坐标（从原图归一化坐标 → letterbox后像素坐标）
        # 注意：后续数据增强（翻转、RandomCrop）都使用像素坐标系
        if num_objects > 0:
            for i in range(len(bboxes)):
                cx, cy, bw, bh, cls_id = bboxes[i]
                # 归一化坐标 → 原图像素坐标
                cx_px = cx * iw
                cy_px = cy * ih
                bw_px = bw * iw
                bh_px = bh * ih
                # 应用scale和offset → 新像素坐标
                new_cx_px = cx_px * scale + offset_x
                new_cy_px = cy_px * scale + offset_y
                new_bw_px = bw_px * scale
                new_bh_px = bh_px * scale
                bboxes[i] = [
                    new_cx_px,
                    new_cy_px,
                    new_bw_px,
                    new_bh_px,
                    cls_id
                ]
        
        # 4. 数据增强（在letterbox后的正方形图上操作）
        if self.mode == 'train' and config.use_augmentation:
            # 4.0 马赛克增强 [Round14] — 以p=0.5概率将4张图拼接为1张
            if random.random() < 0.5:
                img_resized, bboxes = self._mosaic_augment(img, img_w, img_h)
                num_objects = len(bboxes)

            # 4.0.5 [R22] RandomCrop — 打破空间偏置（特别是Y轴坍缩）
            # 以p=0.35概率随机裁剪70%-100%区域，再缩放回原尺寸
            # 这迫使模型学习位置不变的特征，而非记忆固定空间模式
            if random.random() < 0.35 and len(bboxes) > 0:
                crop_ratio = random.uniform(0.7, 1.0)
                crop_w = int(self.img_size * crop_ratio)
                crop_h = int(self.img_size * crop_ratio)
                max_left = max(0, self.img_size - crop_w)
                max_top = max(0, self.img_size - crop_h)
                left = random.randint(0, max_left) if max_left > 0 else 0
                top = random.randint(0, max_top) if max_top > 0 else 0

                img_resized = img_resized.crop((left, top, left + crop_w, top + crop_h))
                img_resized = img_resized.resize((self.img_size, self.img_size), Image.BILINEAR)

                # 同步调整bbox坐标 (cx, cy, w, h → 新坐标系)
                new_bboxes = []
                for b in bboxes:
                    cx, cy, bw, bh, cls_id = b
                    new_cx = (cx - left) / crop_ratio if crop_ratio > 0 else cx
                    new_cy = (cy - top) / crop_ratio if crop_ratio > 0 else cy
                    new_bw = bw / crop_ratio if crop_ratio > 0 else bw
                    new_bh = bh / crop_ratio if crop_ratio > 0 else bh
                    # 只保留中心仍在图像内的框（允许部分越界）
                    if 0 < new_cx < self.img_size and 0 < new_cy < self.img_size:
                        new_bboxes.append([new_cx, new_cy, new_bw, new_bh, cls_id])
                # 如果所有框都被裁掉了，保留原始空列表（后面会处理）
                if new_bboxes:
                    bboxes = new_bboxes
                    num_objects = len(bboxes)

            # 4.1 水平翻转
            if random.random() < 0.5:
                img_resized = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
                bboxes = [[self.img_size - b[0], b[1], b[2], b[3], b[4]] for b in bboxes]

            # 4.2 颜色增强
            if self.color_transform:
                img_resized = self.color_transform(img_resized)
        
        # 5. 转换为Tensor并归一化
        if self.normalize_transform:
            img_tensor = self.normalize_transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)
        
        # 5.5 bbox从像素坐标归一化到0-1范围（模型训练使用归一化坐标）
        if num_objects > 0:
            for i in range(len(bboxes)):
                cx, cy, bw, bh, cls_id = bboxes[i]
                bboxes[i] = [
                    cx / self.img_size,
                    cy / self.img_size,
                    bw / self.img_size,
                    bh / self.img_size,
                    cls_id
                ]
        
        # 6. 填充标注框到固定大小
        bboxes_padded = self._pad_bboxes(bboxes)
        
        meta = {
            'img_path': sample['img_path'],
            'original_size': (img_w, img_h),
            'letterbox_scale': scale,
            'letterbox_offset': (offset_x, offset_y)
        }
        
        return img_tensor, torch.tensor(num_objects, dtype=torch.long), bboxes_padded, meta
    
    def _mosaic_augment(
        self,
        img: Image.Image,
        orig_w: int,
        orig_h: int
    ) -> Tuple[Image.Image, List[List]]:
        """[Round14] 马赛克增强：将4张随机图片拼接为1张img_size×img_size的图。

        拼接布局:
            ┌───────┬───────┐
            │ img0  │ img1  │
            ├───────┼───────┤
            │ img2  │ img3  │
            └───────┴───────┘

        Returns:
            Tuple[Image.Image, List[List]]: (拼接后的图像, 归一化bbox列表[cx,cy,w,h,cls_id])
        """
        target_size = self.img_size
        half = target_size // 2

        # 随机选3个其他样本索引（加上当前共4张）
        indices = [idx for idx in range(len(self.samples)) if self.samples[idx].get('all_bboxes')]
        if len(indices) < 4:
            # 样本不足4张时，回退到原图letterbox处理
            return self._letterbox_single(img, orig_w, orig_h)

        chosen = random.sample(indices, min(4, len(indices)))
        # 确保当前图片在候选中（如果当前图有标注框）
        mosaic_images = []
        mosaic_bboxes = []

        for i, sample_idx in enumerate(chosen):
            sample = self.samples[sample_idx]
            try:
                src_img = Image.open(sample['img_path']).convert('RGB')
                sw, sh = src_img.size
            except Exception:
                continue

            bboxes_raw = sample.get('all_bboxes', [])

            # 计算裁剪区域（基于中心偏移，增加多样性）
            cx_offset = random.uniform(-0.125, 0.125)  # ±12.5%中心偏移
            cy_offset = random.uniform(-0.125, 0.125)

            if i == 0:      # 左上
                crop_x1, crop_y1 = 0, 0
                crop_x2, crop_y2 = sw // 2 + int(sw * cx_offset), sh // 2 + int(sh * cy_offset)
                paste_x, paste_y = 0, 0
            elif i == 1:    # 右上
                crop_x1 = sw // 2 + int(sw * cx_offset)
                crop_y1 = 0
                crop_x2, crop_y2 = sw, sh // 2 + int(sh * cy_offset)
                paste_x, paste_y = half, 0
            elif i == 2:    # 左下
                crop_x1, crop_y1 = 0, sh // 2 + int(sh * cy_offset)
                crop_x2 = sw // 2 + int(sw * cx_offset)
                crop_y2 = sh
                paste_x, paste_y = 0, half
            else:           # 右下
                crop_x1 = sw // 2 + int(sw * cx_offset)
                crop_y1 = sh // 2 + int(sh * cy_offset)
                crop_x2, crop_y2 = sw, sh
                paste_x, paste_y = half, half

            # 裁剪并resize到half×half
            crop_x1 = max(0, min(crop_x1, sw - 1))
            crop_y1 = max(0, min(crop_y1, sh - 1))
            crop_x2 = max(crop_x1 + 1, min(crop_x2, sw))
            crop_y2 = max(crop_y1 + 1, min(crop_y2, sh))

            cropped = src_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            resized = cropped.resize((half, half), Image.BILINEAR)

            # 转换bbox坐标：原图归一化 → 裁剪区域像素 → half尺寸像素 → target_size归一化
            region_w = crop_x2 - crop_x1
            region_h = crop_y2 - crop_y1
            for bbox in bboxes_raw:
                # 原图归一化 → 原图像素
                px_cx = bbox['cx'] * sw
                px_cy = bbox['cy'] * sh
                px_w = bbox['w'] * sw
                px_h = bbox['h'] * sh

                # 检查是否在裁剪区域内
                if (crop_x1 <= px_cx <= crop_x2 and crop_y1 <= px_cy <= crop_y2):
                    # → 裁剪区域相对坐标
                    rel_cx = (px_cx - crop_x1) / region_w
                    rel_cy = (px_cy - crop_y1) / region_h
                    rel_w = px_w / region_w
                    rel_h = px_h / region_h

                    # → target_size归一化坐标
                    new_cx = (paste_x + rel_cx * half) / target_size
                    new_cy = (paste_y + rel_cy * half) / target_size
                    new_w = rel_w * half / target_size
                    new_h = rel_h * half / target_size

                    # 边界裁剪
                    new_cx = max(0.01, min(new_cx, 0.99))
                    new_cy = max(0.01, min(new_cy, 0.99))

                    mosaic_bboxes.append([new_cx, new_cy, new_w, new_h, bbox['class_id']])

            mosaic_images.append((paste_x, paste_y, resized))

        # 创建拼接画布并粘贴4个子图
        canvas = Image.new('RGB', (target_size, target_size), (128, 128, 128))
        for px, py, sub_img in mosaic_images:
            canvas.paste(sub_img, (px, py))

        return canvas, mosaic_bboxes

    def _letterbox_single(
        self,
        img: Image.Image,
        orig_w: int,
        orig_h: int
    ) -> Tuple[Image.Image, List[List]]:
        """对单张图做标准letterbox处理（马赛克回退路径）。"""
        w, h = self.img_size, self.img_size
        iw, ih = img.size
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        img_resized = img.resize((nw, nh), Image.BILINEAR)
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        offset_x = (w - nw) // 2
        offset_y = (h - nh) // 2
        new_img.paste(img_resized, (offset_x, offset_y))

        bboxes = []
        # 从self.samples中获取当前图的bboxes（需要通过__getitem__上下文）
        return new_img, bboxes

    def _pad_bboxes(self, bboxes: List[List]) -> torch.Tensor:
        """填充边界框到固定大小。
        
        Returns:
            torch.Tensor: [max_objects, 5] (cx, cy, w, h, class_id)
        """
        padded = np.zeros((self.max_objects, 5), dtype=np.float32)
        
        # 多缺陷不能默默丢掉尾部 GT：num_objects 必须与实际标签一致。
        if len(bboxes) > self.max_objects:
            raise ValueError(
                f"一张切片包含 {len(bboxes)} 个 GT，超过 config.max_objects={self.max_objects}；"
                "请增加 max_objects 或排查标注。"
            )
        num_bboxes = len(bboxes)
        if num_bboxes > 0:
            padded[:num_bboxes] = bboxes[:num_bboxes]
        
        return torch.tensor(padded, dtype=torch.float32)


def create_dataloader(
    img_dir,
    label_dir,
    mode: str = 'train',
    batch_size: int = None,
    num_workers: int = None,
    preprocess_method: str = None,
    label_format: str = None,
    max_samples_per_dir: int = None,
    from_end: bool = False,
) -> torch.utils.data.DataLoader:
    """创建数据加载器。
    
    Args:
        img_dir: 图像目录（str或list[str]）
        label_dir: 标签目录（str或list[str]）
        mode: 模式
        batch_size: 批量大小
        num_workers: 工作进程数
        preprocess_method: 预处理方法
        label_format: 标签格式
        max_samples_per_dir: 每个目录最多加载样本数
        from_end: 是否从末尾取样本
    
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
        label_format=label_format,
        max_samples_per_dir=max_samples_per_dir,
        from_end=from_end,
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
    print(f"训练集路径: {config.train_img_dirs}")
    print(f"每子集训练样本数: {config.train_samples_per_subset}")
    
    train_loader = create_dataloader(
        config.train_img_dirs, 
        config.train_label_dirs, 
        mode='train',
        max_samples_per_dir=config.train_samples_per_subset,
    )
    
    for img, num_obj, bboxes, meta in train_loader:
        print(f"\n图像形状: {img.shape}")
        print(f"缺陷数量: {num_obj}")
        print(f"边界框形状: {bboxes.shape}")
        print(f"边界框内容:\n{bboxes[0]}")
        print(f"元信息: {meta['img_path'][0]}")
        break
