"""快速生成R40可视化图（跳过50图mAP评估，只画4张）"""
import sys
from pathlib import Path

# 复用 sahi_inference 的模型加载和可视化函数
from sahi_inference import load_model, visualize_one, DATA_DIR, SUBSETS

if __name__ == "__main__":
    print("加载R40模型...")
    model = load_model()
    print("模型加载完成，开始生成可视化（验证集后2张）...")
    for subset in SUBSETS:
        img_dir = DATA_DIR / subset / "images"
        imgs = sorted(img_dir.glob("*.jpg"))[-2:]
        for img_file in imgs:
            out = f"r40_vis_{subset}_{img_file.name}"
            visualize_one(model, str(img_file), out)
    print("全部完成。")
