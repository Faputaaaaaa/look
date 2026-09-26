# HANDOFF.md — 交接状态

> 记录当前进度，供下次新会话快速恢复。恢复时先读本文件 + CLAUDE.md + DECISIONS.md + STATUS.md。

## 当前阶段
今日（2026-09-26）「正确性修复 + 训练前优化 + 可复现工具」7 项任务 **全部完成**。未训练、未全量 SAHI、未碰 P0.5 提案统一与死代码清理。

## 已完成（本次会话 2026-09-26）
- **A13 背景窗口负样本日志监控** —— `RPNTrainer.reset_stats()` + `compute_loss` 统计 + 每 epoch 打印。smoke：24 窗口正 anchor=102,984 / 负 5,391,384 / 背景窗口负 5,376,000（**99.7%**）。
- **A10 窗口内 bbox 裁剪** —— `_get_bboxes_in_window` clip 到 [0,640]、clip 后 <4px 丢弃；`sahi_inference` 扩张后 clip 回原图。smoke：144 个窗口 GT 越界 0，85 个贴边（clip 确实在起作用）。
- **C1+C2 requirements.txt + config 落盘** —— 锁定 torch/torchvision/numpy/Pillow/matplotlib/tqdm（注明 CUDA 12.6 索引）；`config.save_config()` 落盘 `config_<ts>.json`。
- **C3 checkpoint 续训** —— `save_checkpoint`/`load_checkpoint`（model+optimizer+epoch+stage+config）+ `--resume`/`--exp`；`train_model` 重构为单全局 epoch 循环。注：resume **不恢复 scheduler**（LR 余弦从新周期开始，已知限制）。
- **B1 根因排查 + B6/B7/B8 诊断** —— `diagnose_bbox.py`（原始框回归误差统计），报告到 `experiments/diagnosis/`。关键发现见「风险」。
- **C5 验证结果图生成** —— `evaluate_and_visualize.py`（TP/FP/FN 标注，`--n` 控制张数）。
- **A7/A8/A9 顺手项** —— `_cls_weight` 动态化；np.trapezoid 已随 AP 标准实现解决；YOLO ID 映射统一到 `config.yolo_id_to_name`。
- **问题1（A11）扩张配置化 + 默认关闭** —— `config.use_bbox_expansion=False`（开关）+ `expand_missing`/`expand_dirty`（系数保留，代码路径不删）。重训后按新数据重标定。
- **问题2 背景窗口负样本限流** —— `max_neg_per_bg_window=64`；负样本主要来自含缺陷窗口，背景窗口每窗口×每层最多补 64。smoke：负 anchor 5.39M → 18,063（背景窗口占比 99.7% → 8.5%）。

## 未完成（按优先级）
- **A6 = P0.5** 统一 proposal 提取（train/validate/inference 三处重复）。今天明确不碰。
- **A11** 1.3x/1.1x 扩张配置化 + 重新标定（依赖 B1 结论，见 QUESTIONS）。
- **A12** stale 注释清理（只标记不删）。
- **B4/B5** anchor 缓存 + 死代码标记（P2）。
- **C4** GC10DET 残余隔离收尾（P2）。

## 下一步
1. 用户确认 QUESTIONS（1.3x/1.1x 是否重标定、背景窗口负样本是否限流）。
2. 然后 **P0.5** 统一 proposal 提取（改一处三处同步，是当前最大维护风险）。

## 风险（当前未解决）
1. **背景窗口负样本未限流**：A13 监控确认背景窗口贡献 99.7% 负 anchor（每背景窗口全量 ~672k 负 anchor），可能淹没正样本；需观察训练召回率是否异常。
2. **1.3x 扩张过度补偿**：B1 实测 R41 模型 missing 框仅偏小 ~7%（历史 R40 是 50%），1.3x 现会把框放大到 ~1.21x 超 GT。
3. **dirty +34px 向下中心偏移**：B1 新发现，对称扩张无法修正，需定位 ROI bbox 回归 dy 偏置（可能涉及正样本分配 / CIoU 绝对坐标空间 / roi_align 对齐）。
4. **proposal 逻辑三处重复**（P0.5 未做）。
5. **C3 resume 不恢复 scheduler 状态**（LR 余弦重启，非阻断但需知悉）。

## 关键决策
见 DECISIONS.md（均已确认）。本次会话新问题见 QUESTIONS.md（第 5、6 条）。

## 下次恢复方式
1. 开新会话。
2. 先读 HANDOFF.md、CLAUDE.md、DECISIONS.md、STATUS.md。
3. 不要重读全部代码，不要 `/resume` 旧对话。
4. 从「下一步」开始。
