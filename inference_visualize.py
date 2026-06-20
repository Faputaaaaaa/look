"""
推理可视化脚本 - 使用训练好的模型在测试集上运行，生成带红框的检测图
红框 = 模型预测（我们的检测结果）
"""
import torch
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from config import config
from model_fpn import TwoStageDetector

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "experiments/exp_20260609_201408/models/best_model.pth"
OUTPUT_DIR = "experiments/exp_20260609_201408/visualizations"
NUM_SAMPLES = 20
SCORE_THRESH = 0.05  # 与INFERENCE_CONFIG一致

# 类别名称和颜色（红框）
CLASS_NAMES = {1: 'chongkong', 2: 'shuiban', 3: 'youban', 4: 'siban', 5: 'hanfeng'}
CLASS_COLORS = {
    1: (255, 0, 0),     # 红色 - 冲孔
    2: (255, 100, 0),   # 橙红 - 水斑
    3: (255, 200, 0),   # 黄色 - 油斑
    4: (0, 150, 255),   # 蓝色 - 丝斑
    5: (180, 0, 255),   # 紫色 - 焊缝
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {device}")

# ============================================================
# 1. 加载模型
# ============================================================
print(f"\n{'='*60}")
print("加载模型...")
print(f"{'='*60}")
model = TwoStageDetector().to(device)

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] 模型文件不存在: {MODEL_PATH}")
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH, map_location=device)
# 兼容两种保存格式
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   训练Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   最佳验证损失: {checkpoint.get('best_val_loss', 'N/A')}")
else:
    model.load_state_dict(checkpoint)
    print("   (直接state_dict格式)")
print(f"[OK] 模型加载成功")
model.eval()

# ============================================================
# 2. 加载测试集
# ============================================================
print(f"\n{'='*60}")
print("加载测试集...")
print(f"{'='*60}")

from dataset import DefectDataset

# 先初始化config以生成路径
config.setup_experiment()

test_dataset = DefectDataset(
    img_dir=config.test_img_dir,
    label_dir=config.test_label_dir,
    mode='test'
)

print(f"[OK] 测试集加载完成: {len(test_dataset)} 张图片")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 3. 推理 + 可视化
# ============================================================
print(f"\n{'='*60}")
print(f"开始推理可视化 (共{min(NUM_SAMPLES, len(test_dataset))}张)...")
print(f"{'='*60}")

total_detections = 0
total_images_with_det = 0

for idx in range(min(NUM_SAMPLES, len(test_dataset))):
    sample = test_dataset[idx]
    # dataset返回tuple: (image_tensor, num_objects, bboxes, meta)
    image_tensor = sample[0].unsqueeze(0).to(device)  # [1, 3, H, W]

    # 获取原始图像信息（用于还原坐标）
    meta = sample[3] if len(sample) > 3 else {}
    original_img_path = meta.get('original_path', f'unknown_{idx}')

    with torch.no_grad():
        results = model.inference(image_tensor)

    boxes = results['boxes'][0]      # [N, 4]
    scores = results['scores'][0]    # [N]
    labels = results['labels'][0]    # [N]

    num_dets = boxes.size(0)
    total_detections += num_dets
    if num_dets > 0:
        total_images_with_det += 1

    # ===== 绘制可视化图 =====
    # 从tensor转回PIL图像用于绘图
    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)

    # 尝试加载字体（Windows中文支持）
    try:
        font = ImageFont.truetype("msyh.ttc", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    # 绘制每个检测框（红色）
    for i in range(num_dets):
        x1, y1, x2, y2 = boxes[i].tolist()
        score = scores[i].item()
        label_id = labels[i].item()
        class_name = CLASS_NAMES.get(label_id, f'cls{label_id}')
        color = CLASS_COLORS.get(label_id, (255, 0, 0))

        # 绘制红框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制标签文字
        text = f"{class_name}:{score:.2f}"
        draw.text((x1, y1 - 20), text, fill=color, font=font)

    # 保存图片
    save_name = f"pred_{idx:03d}_dets{num_dets}.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    pil_img.save(save_path)

    # 打印进度
    det_info = ""
    if num_dets > 0:
        top_scores = scores[:min(3, num_dets)].tolist()
        top_labels = labels[:min(3, num_dets)].tolist()
        det_info = " | ".join([f"{CLASS_NAMES.get(l, '?')}:{s:.2f}" for l, s in zip(top_labels, top_scores)])
        if num_dets > 3:
            det_info += f"... 共{num_dets}个"

    print(f"  [{idx+1:3d}/{NUM_SAMPLES}] {save_name}: {num_dets}个检测 | {det_info}")

# ============================================================
# 4. 总结报告
# ============================================================
print(f"\n{'='*60}")
print("推理可视化完成!")
print(f"{'='*60}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"总检测框数: {total_detections}")
print(f"有检测的图片: {total_images_with_det}/{min(NUM_SAMPLES, len(test_dataset))}")

if total_images_with_det == 0:
    print("\n[WARN] 警告: 所有图片均无检测结果! 可能原因:")
    print("   1. score_thresh过高 → 尝试降低到0.01")
    print("   2. 模型未充分训练 → 增加epoch或检查loss是否正常下降")
    print("   3. RPN proposal不足 → 检查rpn_score_thresh")
elif total_images_with_det < NUM_SAMPLES * 0.5:
    print(f"\n[WARN] 仅{total_images_with_det}/{NUM_SAMPLES}张有检测,覆盖率偏低")

print("\n请查看可视化结果图片!")
