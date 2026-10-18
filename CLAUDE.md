# CLAUDE.md — 瓶身缺陷检测项目 长期规则

本文件是本项目在 Claude Code 中的长期工作规则。每次会话开始应先读本文件 + HANDOFF.md + DECISIONS.md。

## 项目定位
自研两阶段缺陷检测器（ResNet18 + FPN(P2/P3/P4) + RPN + ROI Head），
针对瓶身两类缺陷 `missing`/`dirty`，SAHI 滑动窗口切片推理。
核心文件：`config.py` / `dataset.py` / `model_fpn.py` / `train.py` / `sahi_inference.py` / `utils.py`。

## 数据划分（硬约束）
- 按**原图**做 80/20 划分 train/val，`seed=42`，结果落盘为 `split.json`。
- **禁止**按切片（窗口）随机划分 train/val —— 多个切片可能来自同一原图，按切片分会导致同原图跨 train/val 泄露。
- 评估**禁止用训练图**。

## 评估规范（硬约束）
- 指标用 **mAP@0.5**。
- 流程：SAHI 切片推理 → 框映射回原图坐标系 → 全局 NMS 合并 → **按原图**计算。
- 报告**每类**的 AP、precision、recall（不只给 mAP）。
- AP 计算锚定 recall=0、precision 补 1（标准 VOC/COCO）；改后数字不可与之前直接比。

## 训练规则
- **背景窗口保留**：参与 RPN 分类损失（提供负样本）；ROI Head 的**正样本跳过**背景窗口。
  - 待办 P0.4：`train.py` 中 `if n_obj == 0: continue` 把背景窗口整个跳过，需修复为「背景窗口参与 RPN cls loss，ROI 正样本跳过」。
- **Warmup（已修好）**：warmup 期间线性从 ~0 升到 target，并真正写入两个参数组（RPN 5x / 其他 1x）的 lr。
- **RPN 负样本配比**：保持 pos:neg = 1:3，加日志监控负样本采样；召回率异常再调。
- **数据增强**：只保留水平翻转；垂直翻转移除（瓶身有方向性）。

## 推理后处理
- **1.3x（missing）/ 1.1x（dirty）检测框扩张保留**，不要删，但要**配置化**（移入 config，不硬编码在 `sahi_inference.py`）。

## 代码维护规则
- **死代码不删，只标记**（加注释说明「已弃用/死路径」）。
- **GC10DET 残余隔离**：VOC 路径、4 类映射等旧逻辑保持隔离，**默认只跑 missing/dirty 两类**。
- 改动 proposal 逻辑时注意：train/validate/inference 三处逻辑高度重复，P0 阶段统一抽成公共函数。

## 工作流
- 长任务用 `tmux` / `nohup`，日志写文件（不要只打屏幕）。
- **每阶段结束更新 HANDOFF.md / STATUS.md 并 git commit**。
- **运行环境**：必须用 `D:\Python3.14\python.exe`（torch+cu126 / PyQt5 装在该解释器上）。
- **平台回归**（改动 `defect_platform/` 或 `sahi_inference.py` 后必做）：
  - `python defect_platform/smoke_ui.py` —— 界面链路冒烟（不弹窗，5 步检查）
  - `python defect_platform/smoke_test.py --n 1` —— 产出效果图，肉眼核对框
- **推理口径单一来源**：任何入口（评估脚本 / 平台 / 冒烟）都必须走
  `sahi_inference.sahi_inference_image()`，不要另写一份切片/后处理逻辑。

## 关键数据尺寸（参考）
- 原图 4096×2736；切片 640×640，stride 320。
- 缺陷尺寸：dirty ≈ 304px，missing ≈ 104px（中位数约 209px）。
- 数据集：dirty / missing 各 100 张，每图 1 个缺陷。
