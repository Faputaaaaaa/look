# STATUS — 当前状态快照

> 快速一览，细节见 HANDOFF.md。

## 当前运行中的训练（2026-09-27 01:38 启动，无人值守）
- 实验: `overnight_20260927`，20 epoch，每 3 epoch 轻量 mAP 验证，重试上限 3。
- 日志: `exp_logs/train_20260927_013821.log`
- 进度（最后确认）: epoch 2/20，loss 4.99→3.66 下降中，~268s/epoch ≈ 80 分钟总时长。
- checkpoint: `experiments/overnight_20260927/models/latest.pth`（每 epoch 更新）+ `best_model.pth`。
- 完成后自动: `diagnose_bbox.py` → `evaluate_and_visualize.py` → `make_report.py`（生成 `TRAIN_REPORT.md` + `val_metrics.json`）。

## 本次会话完成（2026-09-27）
- validate 对齐背景窗口（与 train 同一规则，loss 可直接比较）。
- checkpoint 保存/恢复 model + optimizer + scheduler + epoch + RNG。
- NaN 检测 + OOM 处理（保存 checkpoint 后优雅停止）。
- 每 epoch 保存 latest.pth；`--total-epochs`、`--val-metrics-every` 参数。
- `run_all.sh`（自动恢复循环）+ `make_report.py`（报告生成）。

## 待办（训练完成后 / 下次）
- 读 TRAIN_REPORT.md + val_metrics.json 评估结果。
- 重测 bbox 偏差，决定扩张系数是否重新标定（A11-余）。
- P2 收尾：A12 stale 注释、B4/B5 anchor 缓存/死代码标记、C4 GC10DET 隔离。

## 阻塞点
无。
