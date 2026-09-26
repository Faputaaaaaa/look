# HANDOFF.md — 交接状态

> 记录当前进度，供下次新会话快速恢复。恢复时先读本文件 + CLAUDE.md + DECISIONS.md + STATUS.md。

## 当前阶段
今日（2026-09-26）「正确性修复 + 训练前优化 + 可复现工具 + 问题确认」全部完成，**含 P0.5 提案统一**。未训练、未全量 SAHI。所有改动已提交（git log 见 STATUS）。

## 已完成
- **A13 背景窗口负样本日志监控** —— `RPNTrainer.reset_stats()` + `compute_loss` 统计 + 每 epoch 打印两个负样本来源。
- **A10 窗口内 bbox 裁剪** —— `_get_bboxes_in_window` clip 到 [0,640]、<4px 丢弃；`sahi_inference` 扩张后 clip 回原图。
- **C1+C2 requirements.txt + config 落盘** —— 锁版本 + `config.save_config()` 落盘 `config_<ts>.json`。
- **C3 checkpoint 续训** —— `save_checkpoint`/`load_checkpoint` + `--resume`/`--exp`；`train_model` 单全局 epoch 循环。注：resume 不恢复 scheduler（已知限制）。
- **B1 根因排查 + B6/B7/B8 诊断** —— `diagnose_bbox.py`，报告到 `experiments/diagnosis/`。
- **C5 验证结果图生成** —— `evaluate_and_visualize.py`（TP/FP/FN 标注）。
- **A7/A8/A9 顺手项** —— `_cls_weight` 动态化；np.trapezoid 已随 AP 解决；YOLO ID 映射统一到 `config.yolo_id_to_name`。
- **问题1（A11）扩张配置化 + 默认关闭** —— `config.use_bbox_expansion=False` + `expand_missing`/`expand_dirty`（代码路径保留）。
- **问题2 背景窗口负样本限流** —— `max_neg_per_bg_window=64`；负 anchor 5.39M → 18,063（背景窗口 99.7% → 8.5%）。
- **P0.5 统一 proposal 提取** —— `extract_rpn_proposals` + `assign_levels_and_roi_align` 抽到 `model_fpn.py`，train/validate/inference 三处共用（净减 218 行）。smoke：proposal 数 3181（与旧注释 ~3200 一致）。

## 未完成（按优先级）
- **A11-余 扩张系数重新标定** —— 等重新训练后按新数据统计 bbox 偏差再决定（当前默认关闭）。
- **A12** stale 注释清理（只标记不删）。
- **B4/B5** anchor 缓存 + 死代码标记（P2）。
- **C4** GC10DET 残余隔离收尾（P2）。

## 下一步
1. **重新训练**（改动已就位：扩张默认关闭、背景窗口限流 64、bbox clip、checkpoint 续训）。用 `python train.py --exp <name>`。
2. 训练中观察 `[A13] RPN采样` 日志（含缺陷窗口 vs 背景窗口负 anchor 占比）。
3. 训练后跑 `diagnose_bbox.py` 重测 bbox 回归误差（尤其 dirty 的 +34px dy 是否改善），再决定扩张系数。
4. 用 `evaluate_and_visualize.py` 出验证图 + `sahi_inference.py` 算 mAP。

## 风险（当前未解决）
1. **dirty +34px 向下中心偏移**：B1 实测（16 个 TP，dy mean=+31.5px / median=+32.6px / std=12.3px，全为正），对称扩张无法修正，需训练后重测是否改善（可能涉及 ROI bbox 回归 dy 偏置）。
2. **1.3x 扩张过度补偿**：R41 模型 missing 框仅偏小 ~7%，1.3x 会放大到 ~1.21x 超 GT。已默认关闭。
3. **[新发现] validate 仍跳过背景窗口（n_obj==0）**：train_one_epoch 已按 P0.4 让背景窗口参与 RPN 负样本损失，但 `validate` 仍 `if n_obj==0: continue`，导致 train/val loss 在不同样本集上计算、不可直接比。建议下次让 validate 与 train 一致处理背景窗口。
4. **C3 resume 不恢复 scheduler 状态**（LR 余弦重启，非阻断）。

## 关键决策
见 DECISIONS.md。QUESTIONS.md 全部已确认（含第 5、6 条：扩张默认关闭、背景窗口限流 64）。

## 下次恢复方式
1. 开新会话。
2. 先读 HANDOFF.md、CLAUDE.md、DECISIONS.md、STATUS.md。
3. 不要重读全部代码，不要 `/resume` 旧对话。
4. 从「下一步」开始（重新训练）。
