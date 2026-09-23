"""生成预处理后的图片，保存到本地供查看

输出内容：
1. 原始大图（带缺陷框标注）
2. 滑动窗口切片（含缺陷的窗口）
3. 数据增强后的切片（翻转+颜色抖动）
4. 归一化后反归一化的可视化图
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import torchvision.transforms as transforms

from config import config
from dataset import DefectDataset

random.seed(42)

OUTPUT_DIR = Path(r"d:\deeplearning\learn\项目\preprocessed_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取YOLO标签
yolo_id_to_name = {0:'ghosting',1:'faint',2:'fuzziness',3:'missing',4:'dirty',5:'scratch',6:'body_scratch'}

def read_yolo_bboxes(label_path, img_w, img_h):
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                yolo_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                cls_name = yolo_id_to_name.get(yolo_id, str(yolo_id))
                if cls_name in config.class_name_map:
                    x1 = (cx - w/2) * img_w
                    y1 = (cy - h/2) * img_h
                    x2 = (cx + w/2) * img_w
                    y2 = (cy + h/2) * img_h
                    bboxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'name':cls_name})
    return bboxes

def draw_boxes(img, bboxes, color=(0,255,0), width=6):
    draw = ImageDraw.Draw(img)
    for b in bboxes:
        draw.rectangle([b['x1'], b['y1'], b['x2'], b['y2']], outline=color, width=width)
        draw.text((b['x1'], b['y1']-30), b['name'], fill=color)
    return img

# 颜色增强（与训练一致）
color_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0, hue=0),
])
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def denormalize(tensor):
    """反归一化转PIL图"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = tensor * std + mean
    img = img.clamp(0, 1)
    return transforms.ToPILImage()(img)

import torch

# 取1张dirty和1张missing
for subset in ['dirty', 'missing']:
    img_dir = Path(config.data_dir) / subset / "images"
    lbl_dir = Path(config.data_dir) / subset / "labels"
    img_files = sorted(img_dir.glob("*.jpg"))
    # 取第5张（中间位置，避免边界）
    img_file = img_files[4]
    lbl_file = lbl_dir / (img_file.stem + ".txt")

    img = Image.open(img_file).convert('RGB')
    img_w, img_h = img.size
    bboxes = read_yolo_bboxes(lbl_file, img_w, img_h)

    print(f"\n处理 {subset}: {img_file.name}")
    print(f"  原图尺寸: {img_w}x{img_h}, 缺陷数: {len(bboxes)}")

    # 1. 保存带标注的原图（缩小到1024宽方便查看）
    img_annotated = draw_boxes(img.copy(), bboxes)
    scale = 1024 / img_w
    img_annotated_small = img_annotated.resize((1024, int(img_h*scale)))
    p1 = OUTPUT_DIR / f"01_original_{subset}_annotated.jpg"
    img_annotated_small.save(p1)
    print(f"  保存: {p1}")

    # 2. 找包含缺陷的滑动窗口（与dataset.py._generate_windows一致，含边界窗口）
    ws = config.window_size  # 640
    stride = config.window_stride  # 320
    windows = []
    for y in range(0, img_h - ws + 1, stride):
        for x in range(0, img_w - ws + 1, stride):
            windows.append((x, y))
    # 边界窗口
    if img_w > ws:
        x = img_w - ws
        for y in range(0, img_h - ws + 1, stride):
            windows.append((x, y))
    if img_h > ws:
        y = img_h - ws
        for x in range(0, img_w - ws + 1, stride):
            windows.append((x, y))

    best_window = None
    best_bboxes_in = []
    for (x, y) in windows:
        win_bboxes = []
        for b in bboxes:
            cx = (b['x1']+b['x2'])/2
            cy = (b['y1']+b['y2'])/2
            if x <= cx <= x+ws and y <= cy <= y+ws:
                win_bboxes.append({
                    'x1': b['x1']-x, 'y1': b['y1']-y,
                    'x2': b['x2']-x, 'y2': b['y2']-y,
                    'name': b['name']
                })
        if len(win_bboxes) > len(best_bboxes_in):
            best_window = (x, y)
            best_bboxes_in = win_bboxes

    x, y = best_window
    img_crop = img.crop((x, y, x+ws, y+ws))
    print(f"  选中窗口: ({x},{y}), 窗口内缺陷数: {len(best_bboxes_in)}")

    # 保存原始切片（带标注）
    img_crop_annotated = draw_boxes(img_crop.copy(), best_bboxes_in, color=(0,255,0), width=4)
    p2 = OUTPUT_DIR / f"02_window_crop_{subset}_annotated.jpg"
    img_crop_annotated.save(p2)
    print(f"  保存: {p2}")

    # 3. 数据增强后的切片
    img_aug = img_crop.copy()
    aug_info = []
    # 水平翻转
    if random.random() < 0.5:
        img_aug = img_aug.transpose(Image.FLIP_LEFT_RIGHT)
        aug_info.append("水平翻转")
        best_bboxes_in = [{**b, 'x1':ws-b['x2'], 'x2':ws-b['x1']} for b in best_bboxes_in]
    # 垂直翻转
    if random.random() < 0.3:
        img_aug = img_aug.transpose(Image.FLIP_TOP_BOTTOM)
        aug_info.append("垂直翻转")
        best_bboxes_in = [{**b, 'y1':ws-b['y2'], 'y2':ws-b['y1']} for b in best_bboxes_in]
    # 颜色增强
    img_aug = color_transform(img_aug)
    aug_info.append("颜色抖动")
    print(f"  增强操作: {', '.join(aug_info)}")

    img_aug_annotated = draw_boxes(img_aug.copy(), best_bboxes_in, color=(255,0,0), width=4)
    p3 = OUTPUT_DIR / f"03_augmented_{subset}_annotated.jpg"
    img_aug_annotated.save(p3)
    print(f"  保存: {p3}")

    # 4. 归一化后反归一化（模型实际输入的样子）
    tensor = normalize(img_aug)
    img_denorm = denormalize(tensor)
    p4 = OUTPUT_DIR / f"04_normalized_denorm_{subset}.jpg"
    img_denorm.save(p4)
    print(f"  保存: {p4}")

print(f"\n全部完成！输出目录: {OUTPUT_DIR}")
print(f"共生成 {len(list(OUTPUT_DIR.glob('*.jpg')))} 张图片")
