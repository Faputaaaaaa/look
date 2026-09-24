# HANDOFF.md — 交接状态

> 记录当前进度，供下次新会话快速恢复。恢复时先读本文件 + CLAUDE.md + DECISIONS.md。

## 当前阶段
**P0.1、P0.2 已完成**，接下来 P0.3（修 warmup）。今日（2026-09-24）时间盒：11:45～13:30，只做可中断低风险任务。

## 已完成
- 通读 6 个核心文件（config / utils / model_fpn / dataset / train / sahi_inference，共约 4300 行）和 5 个诊断脚本。
- 核对数据集实际结构：dirty / missing 各 100 张，原图 4096×2736，每图 1 个缺陷。
- 输出完整审查报告（代码现状 / 缺失模块 / 潜在 bug / 待完善项）。
- **P0.1** 按原图分层划分 train/val，`seed=42`，落盘 `split.json`（脚本 `make_split.py`）。
  - 原图 ID 用 `{子集}/{文件名}`（如 `dirty/frame_0000.jpg`），因 dirty/missing 文件名同为 frame_XXXX.jpg 会重名。
  - 原图 ID 映射存在：切片是运行时生成，窗口样本 `meta['img_path']` 即来源原图，按原图分后无切片级泄露。
  - 结果：dirty 80/20、missing 80/20，val 各 20 张（≥15，达标）。
- **P0.2** 修 mAP 泄露（`sahi_inference.py:evaluate_map`）。
  - 旧实现：`max_imgs=50` 取每个子集「排序后最后 25 张」当验证集，但真验证集只有后 20 张 → 混入 5 张训练图。
  - 新实现：只读 `split.json` 的 val 原图（禁止训练图）；SAHI 合并后按原图算 mAP@0.5，报告每类 AP/P/R。
  - 冒烟测试通过：4 张 dirty 全链路无报错；missing 图解析+GT 正常；train∩val 重叠=0。
  - 已知小瑕疵（未改，非本次范围）：AP 用 `np.trapezoid(precision, recall)` 没锚定 recall=0 起点，会略低估 AP（纯 TP 时 AP=0.75 而非 1.0）。改它属于「评估协议」变更，留给下次确认。

## 未完成（按优先级）

### P0（必须先做）
- ~~**P0.1** 按原图 80/20 划分 train/val，`seed=42`，落盘 `split.json`。~~ ✅ 已完成
- ~~**P0.2** 修 mAP 评估 train/val 泄露（`sahi_inference.py:evaluate_map`：改为只读 `split.json` 的 val 原图）。~~ ✅ 已完成
- **P0.3** 修 warmup（`WarmupCosineScheduler` 真正更新两个参数组 lr，而非只改日志）。
- **P0.3** 修 warmup（`WarmupCosineScheduler` 真正更新两个参数组 lr，而非只改日志）。
- **P0.4** 背景窗口参与 RPN 分类损失，ROI 正样本跳过（改 `train.py` 的 `n_obj==0` 跳过逻辑）。
- **P0.5** 统一 proposal 提取逻辑（train / validate / inference 三处重复，抽成公共函数）。

### P1
- `_cls_weight` 动态化（`train.py` 当前硬编码 3 类，换 GC10DET 会崩）。
- 1.3x / 1.1x 扩张配置化。
- 补齐 `requirements.txt`、config 落盘、训练中 mAP 追踪。
- `np.trapezoid` 兼容（NumPy 2.0 才有的 API，旧版是 `np.trapz`）。
- YOLO ID 映射统一为单一事实来源（现散落 4 处）。

### P2
- stale 注释清理（只标记，不删死代码）。
- anchor 缓存（避免每 batch 重复生成）。
- 死代码标记（`multi_class_focal_loss`、`_spatial_uniform_sample` 等）。
- GC10DET 残余隔离收尾。

## 下一步
**P0.3**：修 `WarmupCosineScheduler`，warmup 期间真正更新 optimizer 两个参数组 lr，日志打印真实 lr。

## 关键决策
全部按 CLAUDE.md 的默认规则执行（见 DECISIONS.md 决策表）。

## 风险（当前未修复）
1. **mAP 泄露未修** —— 现有 mAP 数字含约 20% 训练图（50 张评估图里 10 张是训练图），不可信。
2. **warmup 伪生效** —— 前 2 epoch 学习率恒为初始值，`get_last_lr()` 返回的斜坡只是日志假象。
3. **背景窗口被跳过** —— `n_obj==0` 的样本整体 `continue`，背景负样本完全没参与训练。
4. **proposal 逻辑三处重复** —— 改一处易漏另外两处（train / validate / inference）。

## 下次恢复方式
1. 开新会话。
2. 先读 HANDOFF.md、CLAUDE.md、DECISIONS.md。
3. **不要重读全部代码，不要 `/resume` 旧对话**。
4. 从 P0.1 开始。
