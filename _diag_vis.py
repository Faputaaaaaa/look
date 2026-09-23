"""诊断 R40 可视化图：分析红绿框关系"""
import os
from pathlib import Path

# PIL 读取图片大小，确认是否生成成功
from PIL import Image

files = [
    'r40_vis_dirty_frame_0098.jpg',
    'r40_vis_dirty_frame_0099.jpg',
    'r40_vis_missing_frame_0098.jpg',
    'r40_vis_missing_frame_0099.jpg',
]

print("=" * 60)
print("R40 可视化图诊断")
print("=" * 60)

for f in files:
    p = Path(f)
    if p.exists():
        img = Image.open(p)
        print(f"\n{f}")
        print(f"  尺寸: {img.size}")
        print(f"  大小: {os.path.getsize(p) / 1024:.1f} KB")
    else:
        print(f"\n{f} - 不存在!")

print("\n" + "=" * 60)
print("说明：由于无法直接看图，需要从推理日志中提取 IoU")
print("=" * 60)
