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
import json
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
    img_size: int = 640              # [R35] 640切片+640输入，无二次缩放
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
    # xinpaide: 4096 x 2736
    original_img_width: int = 4096
    original_img_height: int = 2736
    
    # 滑动窗口参数
    # [R35] 640切片：missing占20%, dirty占47%, dirty最大356<640不被切
    window_size: int = 640           # 窗口大小（=img_size，无二次缩放）
    window_stride: int = 320         # 窗口步长（50%重叠，确保缺陷不被切断）
    
    # 智能裁剪参数（训练时使用）
    crop_padding: int = 100          # 裁剪时的padding（确保缺陷完整）
    min_crop_size: int = 512         # 最小裁剪尺寸
    
    # ============================================================
    # 数据集参数
    # ============================================================
    
    # [R24] 切换到xinpaide数据集（两类缺陷：dirty, missing）
    data_dir: str = r"d:\deeplearning\learn\xinpaide"
    
    # 标签格式: "voc" (XML) 或 "yolo" (TXT)
    label_format: str = "yolo"
    
    # xinpaide实际只用2类：missing和dirty
    # 参见 d:\deeplearning\learn\xinpaide\dirty\labels\classes.txt：
    #   0=ghosting, 1=faint, 2=fuzziness, 3=missing, 4=dirty, 5=scratch, 6=body_scratch
    # YOLO原始ID: missing=3, dirty=4
    # 重新映射为模型class_id: missing=0, dirty=1
    class_names: List[str] = field(default_factory=lambda: ['missing', 'dirty'])

    # 类名 → 模型class_id 映射
    # 注意：这里key是类名(str)，不是YOLO原始ID(int)。
    # YOLO原始ID → 类名 → 模型class_id 的转换见 dataset.py:_read_yolo_labels 的 yolo_id_to_name。
    class_name_map: Dict = field(default_factory=lambda: {
        'missing': 0,   # YOLO原始ID 3 → 类名 'missing' → 模型class_id 0
        'dirty': 1,     # YOLO原始ID 4 → 类名 'dirty'  → 模型class_id 1
    })

    # [A9] YOLO 原始数字 ID → 类名（xinpaide classes.txt 全 7 类），单一事实来源。
    # 其余模块（dataset.py / sahi_inference.py）从这里 import，不再各自硬编码。
    yolo_id_to_name: Dict = field(default_factory=lambda: {
        0: 'ghosting', 1: 'faint', 2: 'fuzziness',
        3: 'missing', 4: 'dirty', 5: 'scratch', 6: 'body_scratch'
    })

    # 最大目标数量
    max_objects: int = 10  # xinpaide每张图只有1个缺陷，10足够
    
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

    # [A11] 检测框扩张后处理（默认关闭；重新训练后按新数据统计再决定是否启用）
    # 历史 R40/R41 用 1.3x(missing)/1.1x(dirty) 补偿框偏小，但 B1 诊断发现
    # R41 模型 missing 框仅偏小 ~7%（非历史 50%），扩张已过度补偿，故默认关闭。
    use_bbox_expansion: bool = False  # 是否启用扩张（开关）
    expand_missing: float = 1.3       # missing 类扩张系数（历史值，保留）
    expand_dirty: float = 1.1         # dirty 类扩张系数（历史值，保留）

    # [A14] dirty 类 dy 偏移补偿（后处理整体 y 平移，默认关闭）
    # overnight_20260927 训练后 dirty 框系统性向下偏移：val dy median=+20.8px、
    # train dy median=+13.6px（100% 为正），宽高比已正常（~0.98），属平移偏差。
    # 标定协议：val dirty 前 10 张定参、后 10 张验证（不偷看验证答案）。
    # [标定结论 2026-09-29] 标定值 -26.3px；后 10 张验证：TP/FP/FN 无变化
    # （补偿前 P=R=0.900 IoU=0.646 → 补偿后 P=R=0.900 IoU=0.665）。
    # 原因：偏移量两极分化（一半图 +0~5px、一半 +25~50px，std=18.5），
    # 固定平移对大偏移图有益、对小偏移图过度纠正，净收益≈0。故保持默认关闭。
    use_dy_compensation: bool = False  # 是否启用 dy 补偿（开关）
    dy_comp_dirty: float = -26.3       # dirty 类 y 平移像素（负=上移；标定值，未启用）

    # [平台 2026-09-29] 每张原图最终保留的检测框数（SAHI 全局 NMS 后按分数取 top-k）。
    # 默认 1 = 评估口径（xinpaide 每张原图只有 1 个缺陷，make_report/mAP 同口径）。
    # 真实产线可能一图多缺陷，平台 UI 可调到 3/5，不影响评估脚本默认值。
    max_dets_per_image: int = 1

    # [平台 2026-09-29] 平台默认 checkpoint（相对项目根），供 defect_platform 使用。
    # 优先级：命令行 --model > 环境变量 SAHI_MODEL_PATH > 本配置 > sahi_inference.MODEL_PATH
    default_ckpt: str = r"experiments\overnight_20260927\models\best_model.pth"
    
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
    # [R35] 启用滑动窗口：原图4096×2736，缺陷中位209px
    # 1024切片让缺陷占比12-27%，P2特征图上32px清晰可学
    preprocess_method: str = "sliding_window"
    
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
        
        # 设置数据集子目录（xinpaide结构）
        if self.label_format == "yolo":
            # [R24] 合并dirty和missing两个子集，按8:2划分训练/验证
            # 训练集：dirty前80张 + missing前80张 = 160张
            # 验证集：dirty后20张 + missing后20张 = 40张
            self.train_img_dirs = [
                str(Path(self.data_dir) / "dirty" / "images"),
                str(Path(self.data_dir) / "missing" / "images"),
            ]
            self.train_label_dirs = [
                str(Path(self.data_dir) / "dirty" / "labels"),
                str(Path(self.data_dir) / "missing" / "labels"),
            ]
            self.val_img_dirs = [
                str(Path(self.data_dir) / "dirty" / "images"),
                str(Path(self.data_dir) / "missing" / "images"),
            ]
            self.val_label_dirs = [
                str(Path(self.data_dir) / "dirty" / "labels"),
                str(Path(self.data_dir) / "missing" / "labels"),
            ]
            self.test_img_dirs = self.val_img_dirs
            self.test_label_dirs = self.val_label_dirs
            # 每个子集取前N张作为训练，剩余作为验证
            self.train_samples_per_subset = 80
            self.val_samples_per_subset = 20
        else:
            # GC10DET VOC格式
            self.train_img_dir = str(Path(self.data_dir) / "train" / "images")
            self.train_label_dir = str(Path(self.data_dir) / "train" / "annotations")
            self.val_img_dir = str(Path(self.data_dir) / "test" / "images")
            self.val_label_dir = str(Path(self.data_dir) / "test" / "annotations")
            self.test_img_dir = str(Path(self.data_dir) / "test" / "images")
            self.test_label_dir = str(Path(self.data_dir) / "test" / "annotations")
    
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

    def save_config(self, base_dir=None) -> Path:
        """[C2] 保存配置快照到实验目录（文件名带时间戳），保证实验可复现。

        Args:
            base_dir: 目标目录（Path 或 str），默认 experiments/<experiment_name>。

        Returns:
            Path: 保存的配置文件路径
        """
        base_dir = Path(base_dir) if base_dir else Path("experiments") / self.experiment_name
        base_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base_dir / f"config_{ts}.json"
        # 用 __dict__ 而非 dataclasses.asdict，以包含 __post_init__ 动态属性
        # （train_img_dirs / val_img_dirs / window_overlap 等）
        snapshot = dict(self.__dict__)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)
        print(f"[Config] 配置快照已保存: {path}")
        return path
    
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
