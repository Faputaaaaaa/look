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

## 训练规则
- **背景窗口保留**：参与 RPN 分类损失（提供负样本）；ROI Head 的**正样本跳过**背景窗口。
  - 当前 bug：`train.py` 中 `if n_obj == 0: continue` 把背景窗口整个跳过，需修复为「背景窗口参与 RPN cls loss，ROI 正样本跳过」。
- **Warmup 保留并修好**：当前 `WarmupCosineScheduler` 是空操作（`step()` 在 warmup 期间不更新 optimizer lr）。需真正更新**两个参数组**（RPN 5x / 其他 1x）的 lr。

## 推理后处理
- **1.3x（missing）/ 1.1x（dirty）检测框扩张保留**，不要删，但要**配置化**（移入 config，不硬编码在 `sahi_inference.py`）。

## 代码维护规则
- **死代码不删，只标记**（加注释说明「已弃用/死路径」）。
- **GC10DET 残余隔离**：VOC 路径、4 类映射等旧逻辑保持隔离，**默认只跑 missing/dirty 两类**。
- 改动 proposal 逻辑时注意：train/validate/inference 三处逻辑高度重复，P0 阶段统一抽成公共函数。

## 工作流
- 长任务用 `tmux` / `nohup`，日志写文件（不要只打屏幕）。
- **每阶段结束更新 HANDOFF.md / STATUS.md 并 git commit**。

## 关键数据尺寸（参考）
- 原图 4096×2736；切片 640×640，stride 320。
- 缺陷尺寸：dirty ≈ 304px，missing ≈ 104px（中位数约 209px）。
- 数据集：dirty / missing 各 100 张，每图 1 个缺陷。
