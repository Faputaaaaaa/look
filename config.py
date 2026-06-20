"""
融合版配置模块

特点：
1. 简洁：直接在代码里修改参数
2. 完整：包含随机种子、实验目录、配置保存
3. 滑动窗口：专门配置窗口参数

使用方法：
    from config import config
    
    # 直接修改参数
    config.img_size = 512
    config.window_size = 1024
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class Config:
    """统一配置类。
    
    所有参数集中管理，直接修改即可。
    包含随机种子设置，确保结果可复现。
    
    使用方法：
        # 修改参数
        config.img_size = 512
        config.batch_size = 4
        
        # 设置随机种子
        config.set_seed()
        
        # 创建实验目录
        config.setup_experiment()
    """
    
    # ============================================================
    # 基础参数 - 直接修改这里
    # ============================================================
    
    # 模型参数
    img_size: int = 768              # 模型输入尺寸（回退到768以匹配现有训练权重）
    batch_size: int = 1              # 批量大小（改为1以适应8GB显存限制）
    epochs: int = 50                 # 训练轮数
    learning_rate: float = 1e-4      # 学习率
    weight_decay: float = 1e-4       # 权重衰减
    
    # 随机种子（确保结果可复现）
    seed: int = 42
    
    # ============================================================
    # 滑动窗口参数 - 根据你的数据集分析结果设置
    # ============================================================
    
    # 原始图片尺寸（从数据集分析得到）
    original_img_width: int = 2048   # 原图宽度
    original_img_height: int = 1000  # 原图高度
    
    # 滑动窗口参数
    window_size: int = 1024          # 窗口大小（建议：最大缺陷的2-3倍）
    window_stride: int = 512         # 窗口步长（建议：窗口大小的1/2，重叠率50%）
    
    # 智能裁剪参数（训练时使用）
    crop_padding: int = 100          # 裁剪时的padding（确保缺陷完整）
    min_crop_size: int = 512         # 最小裁剪尺寸
    
    # ============================================================
    # 数据集参数
    # ============================================================
    
    # 数据集路径（GC10DET数据集）
    data_dir: str = r"d:\deeplearning\learn\项目\公开数据集GC10DET"
    
    # 标签格式: "voc" (XML) 或 "yolo" (TXT)
    label_format: str = "voc"
    
    # 类别名称（GC10DET: 冲孔、水斑、油斑、丝斑、焊缝）
    class_names: List[str] = field(default_factory=lambda: [
        'chongkong', 'shuiban', 'youban', 'siban', 'hanfeng'
    ])
    
    # 类别映射（XML标签名称 -> 类别ID）
    # 例如: "1_chongkong" -> 0, "4_shuiban" -> 1, "2_hanfeng" -> 4
    class_name_map: Dict = field(default_factory=lambda: {
        '1_chongkong': 0, 'chongkong': 0,
        '4_shuiban': 1, 'shuiban': 1,
        '5_youban': 2, 'youban': 2,
        '6_siban': 3, 'siban': 3,
        '2_hanfeng': 4, 'hanfeng': 4,
    })
    
    # 最大目标数量（用于padding）
    max_objects: int = 20  # 从10提升到20（覆盖youban类单张图13个目标的情况）
    
    # ============================================================
    # 训练参数
    # ============================================================
    
    # 学习率调度
    lr_scheduler: str = "cosine"     # cosine / step / constant
    warmup_epochs: int = 2           # 预热轮数（从5降到2，避免占阶段1过大比例）
    
    # 数据增强
    use_augmentation: bool = True
    color_jitter: float = 0.3
    random_flip: bool = True
    
    # 损失权重
    cls_weight: float = 1.0
    bbox_weight: float = 1.0
    
    # ============================================================
    # 推理参数
    # ============================================================
    
    nms_threshold: float = 0.5       # NMS阈值
    score_threshold: float = 0.5     # 置信度阈值
    
    # ============================================================
    # 路径参数
    # ============================================================
    
    # 实验名称（为空则自动生成）
    experiment_name: str = ""
    
    # 模型保存路径（为空则自动设置）
    model_save_path: str = ""
    
    # 日志保存路径（为空则自动设置）
    log_dir: str = ""
    
    # ============================================================
    # 设备参数
    # ============================================================
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0  # Windows兼容性：设为0避免multiprocessing问题
    pin_memory: bool = True
    use_amp: bool = True             # 混合精度训练
    
    # ============================================================
    # 方法参数
    # ============================================================
    
    # 预处理方法: "smart_crop" / "sliding_window" / "resize"
    # - smart_crop: 智能裁剪（训练推荐，利用标签信息）
    # - sliding_window: 滑动窗口（推理推荐）
    # - resize: 直接缩放（推荐，简单可靠）
    preprocess_method: str = "resize"
    
    def __post_init__(self):
        """初始化后处理。"""
        # 自动生成实验名称
        if not self.experiment_name:
            self.experiment_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        
        # 计算滑动窗口重叠率
        self.window_overlap = 1 - self.window_stride / self.window_size
        
        # 设置路径
        base_dir = Path("experiments") / self.experiment_name
        
        if not self.model_save_path:
            self.model_save_path = str(base_dir / "models" / "best_model.pth")
        
        if not self.log_dir:
            self.log_dir = str(base_dir / "logs")
        
        # 设置数据集子目录
        self.train_img_dir = str(Path(self.data_dir) / "train" / "images")
        # 根据标签格式选择目录名
        label_dir_name = "annotations" if self.label_format == "voc" else "labels"
        self.train_label_dir = str(Path(self.data_dir) / "train" / label_dir_name)
        
        # 验证集：使用test集（GC10DET没有单独的val集）
        self.val_img_dir = str(Path(self.data_dir) / "test" / "images")
        self.val_label_dir = str(Path(self.data_dir) / "test" / label_dir_name)
        
        # 测试集
        self.test_img_dir = str(Path(self.data_dir) / "test" / "images")
        self.test_label_dir = str(Path(self.data_dir) / "test" / label_dir_name)
    
    def set_seed(self) -> None:
        """设置随机种子，确保结果可复现。
        
        设置以下随机种子：
        - Python random
        - NumPy
        - PyTorch CPU
        - PyTorch CUDA
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"[Config] 随机种子已设置: {self.seed}")
    
    def setup_experiment(self) -> Path:
        """创建实验目录结构。
        
        Returns:
            Path: 实验目录路径
        """
        base_dir = Path("experiments") / self.experiment_name
        
        # 创建目录
        (base_dir / "models").mkdir(parents=True, exist_ok=True)
        (base_dir / "logs").mkdir(parents=True, exist_ok=True)
        (base_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        
        print(f"[Config] 实验目录: {base_dir}")
        return base_dir
    
    def print_config(self) -> None:
        """打印当前配置。"""
        print("\n" + "=" * 60)
        print("当前配置")
        print("=" * 60)
        
        print(f"\n[模型参数]")
        print(f"  输入尺寸: {self.img_size}")
        print(f"  批量大小: {self.batch_size}")
        print(f"  训练轮数: {self.epochs}")
        print(f"  学习率: {self.learning_rate}")
        
        print(f"\n[滑动窗口]")
        print(f"  窗口大小: {self.window_size}")
        print(f"  窗口步长: {self.window_stride}")
        print(f"  重叠率: {self.window_overlap*100:.0f}%")
        
        print(f"\n[数据集]")
        print(f"  路径: {self.data_dir}")
        print(f"  原图尺寸: {self.original_img_width} x {self.original_img_height}")
        print(f"  类别数: {len(self.class_names)}")
        
        print(f"\n[预处理方法]")
        print(f"  方法: {self.preprocess_method}")
        
        print(f"\n[设备]")
        print(f"  设备: {self.device}")
        print(f"  混合精度: {self.use_amp}")
        
        print("=" * 60 + "\n")
    
    def validate(self) -> bool:
        """验证配置是否有效。
        
        Returns:
            bool: 配置是否有效
        """
        errors = []
        
        # 检查数据集路径
        if not os.path.exists(self.data_dir):
            errors.append(f"数据集路径不存在: {self.data_dir}")
        
        # 检查窗口参数
        if self.window_size > self.original_img_width or self.window_size > self.original_img_height:
            errors.append(f"窗口大小({self.window_size})大于原图尺寸({self.original_img_width}x{self.original_img_height})")
        
        if self.window_stride >= self.window_size:
            errors.append(f"窗口步长({self.window_stride})应小于窗口大小({self.window_size})")
        
        if errors:
            print("[Config] 配置错误:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# 创建全局配置实例
config = Config()


# ============================================================
# 快速配置预设
# ============================================================

def use_smart_crop():
    """使用智能裁剪模式（训练推荐）。"""
    config.preprocess_method = "smart_crop"
    config.img_size = 512
    config.batch_size = 8
    print("[Config] 已切换到智能裁剪模式")


def use_sliding_window():
    """使用滑动窗口模式（推理推荐）。"""
    config.preprocess_method = "sliding_window"
    config.window_size = 1024
    config.window_stride = 512
    config.img_size = 512
    print("[Config] 已切换到滑动窗口模式")


def use_resize():
    """使用直接缩放模式（不推荐）。"""
    config.preprocess_method = "resize"
    config.img_size = 640
    config.batch_size = 4
    print("[Config] 已切换到直接缩放模式（注意：缺陷会变小）")


if __name__ == "__main__":
    # 测试配置
    config.print_config()
    config.set_seed()
    config.setup_experiment()
    print(f"配置验证: {'通过' if config.validate() else '失败'}")
