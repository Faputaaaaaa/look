"""
旧模型推理对比脚本 - 兼容5类无背景的旧架构
用于对比exp_20260528_150629/203754与当前模型的效果差异
"""
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import config

config.setup_experiment()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {device}")

# ============================================================
# 定义旧版模型架构（无背景类的5分类）
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple

class ROIHead_Old(nn.Module):
    """旧版ROI Head - 无背景类（5类直接输出）"""

    def __init__(self, in_channels: int = 256, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.3)

        # 旧版：没有+1 for background
        self.cls_head = nn.Linear(1024, num_classes)  # 直接输出num_classes维
        # 旧版bbox_head可能是固定16维（4类×4或特殊设计）
        self.bbox_head = nn.Linear(1024, 16)  # 固定16维以匹配旧模型

    def forward(self, roi_features):
        x = roi_features.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        cls_pred = self.cls_head(x)
        bbox_pred = self.bbox_head(x)

        return cls_pred, bbox_pred


# ============================================================
# 加载完整的新模型架构，但替换ROI Head
# ============================================================
print("\n" + "="*70)
print("加载旧模型 (兼容模式)")
print("="*70)

# 先导入新模型获取其他组件
from model_fpn import TwoStageDetector, FPN_CONFIG, INFERENCE_CONFIG

# 创建新模型实例
model = TwoStageDetector().to(device)

# 替换为旧版ROI Head (5类无背景, bbox固定16维)
old_roi_head = ROIHead_Old(in_channels=256, num_classes=5).to(device)
model.roi_head = old_roi_head

# 加载旧模型权重
OLD_MODEL_PATH = "experiments/exp_20260528_150629/models/best_model.pth"
checkpoint = torch.load(OLD_MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"[OK] 旧模型加载成功: {OLD_MODEL_PATH}")
    print(f"   训练Epoch: {checkpoint.get('epoch', 'N/A')}")
else:
    model.load_state_dict(checkpoint, strict=False)
    print("[OK] 旧模型加载成功 (直接state_dict格式)")

model.eval()

# 打印模型结构信息
print(f"\n[INFO] 当前模型类别数: {model.num_classes}")
print(f"[INFO] ROI Head cls_head输出维度: {model.roi_head.cls_head.out_features}")
print(f"[INFO] ROI Head bbox_head输出维度: {model.roi_head.bbox_head.out_features}")

# ============================================================
# 加载测试集
# ============================================================
print(f"\n{'='*70}")
print("加载测试集...")
print("="*70)

from dataset import DefectDataset

test_dataset = DefectDataset(
    img_dir=config.test_img_dir,
    label_dir=config.test_label_dir,
    mode='test'
)
print(f"[OK] 测试集: {len(test_dataset)} 张图片")

# ============================================================
# 推理 + 可视化
# ============================================================
OUTPUT_DIR = "experiments/old_vs_new_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 10
CLASS_NAMES = {1: 'chongkong', 2: 'shuiban', 3: 'youban', 4: 'siban', 5: 'hanfeng'}
CLASS_COLORS = {
    1: (255, 0, 0),      # chongkong - 红色
    2: (255, 165, 0),    # shuiban - 橙色
    3: (0, 255, 0),      # youban - 绿色
    4: (128, 0, 128),    # siban - 紫色
    5: (0, 0, 255),      # hanfeng - 蓝色
}

try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

print(f"\n{'='*70}")
print(f"开始推理 (共{min(NUM_SAMPLES, len(test_dataset))}张)...")
print("="*70)

total_detections = 0
total_images_with_det = 0
class_counts = {i: 0 for i in range(1, 6)}
all_scores = []
all_box_areas = []

for idx in range(min(NUM_SAMPLES, len(test_dataset))):
    sample = test_dataset[idx]
    image_tensor = sample[0].unsqueeze(0).to(device)
    meta = sample[3] if len(sample) > 3 else {}

    with torch.no_grad():
        results = model.inference(image_tensor)

    boxes = results['boxes'][0]
    scores = results['scores'][0]
    labels = results['labels'][0]

    # 还原到原图尺寸
    orig_w = int(meta.get('orig_width', config.img_size))
    orig_h = int(meta.get('orig_height', config.img_size))
    scale_x = orig_w / config.img_size
    scale_y = orig_h / config.img_size

    # 读取原始图像绘制可视化
    img_path = test_dataset.samples[idx]['img_path']
    orig_img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(orig_img)

    num_dets = boxes.size(0)
    total_detections += num_dets

    if num_dets > 0:
        total_images_with_det += 1

    det_info = []
    for i in range(num_dets):
        x1, y1, x2, y2 = boxes[i].tolist()
        # 缩放到原图坐标
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y

        score = scores[i].item()
        label_id = labels[i].item()
        class_name = CLASS_NAMES.get(label_id, f'cls{label_id}')
        color = CLASS_COLORS.get(label_id, (255, 0, 0))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{class_name}:{score:.2f}"
        draw.text((x1, y1 - 20), text, fill=color, font=font)

        class_counts[label_id] += 1
        all_scores.append(score)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        all_box_areas.append(w * h)

        det_info.append(f"{class_name}:{score:.2f}")

    save_path = os.path.join(OUTPUT_DIR, f"old_model_pred_{idx:03d}.png")
    orig_img.save(save_path)
    print(f"[{idx+1:3d}] 检测数={num_dets}, 类别={det_info}")

# ============================================================
# 统计报告
# ============================================================
print(f"\n{'='*70}")
print("【旧模型推理统计】")
print("="*70)
print(f"总检测框数: {total_detections}")
print(f"有检测的图片: {total_images_with_det}/{NUM_SAMPLES}")

if all_scores:
    scores_arr = np.array(all_scores)
    areas_arr = np.array(all_box_areas)
    print(f"\n【置信度分布】")
    print(f"  范围: [{scores_arr.min():.3f}, {scores_arr.max():.3f}]")
    print(f"  均值: {scores_arr.mean():.3f}")
    print(f"  标准差: {scores_arr.std():.3f}")

    print(f"\n【检测框尺寸分布 (像素)】")
    print(f"  面积范围: [{areas_arr.min():.0f}, {areas_arr.max():.0f}] px²")
    print(f"  平均面积: {areas_arr.mean():.0f} px²")

    # 与图像面积的比例
    avg_area_pct = areas_arr.mean() / (orig_w * orig_h) * 100
    print(f"  占图像比例: {avg_area_pct:.1f}%")

print(f"\n【类别分布】")
for cls_id, count in class_counts.items():
    pct = count / max(total_detections, 1) * 100
    name = CLASS_NAMES.get(cls_id, f'cls{cls_id}')
    print(f"  {name}: {count} ({pct:.1f}%)")

print(f"\n[OK] 可视化已保存到: {OUTPUT_DIR}")
