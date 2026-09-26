# STATUS — 当前状态快照

> 快速一览，细节见 HANDOFF.md。

## 本次会话全部完成（2026-09-26）
| 项 | 内容 | commit |
|---|---|---|
| A13 | 背景窗口负样本日志监控 | 79c6974 |
| A10 | 窗口内 bbox clip | 93f421d |
| C1+C2 | requirements.txt + config 落盘 | 00de540 |
| C3 | checkpoint 续训 + --resume | 7f9b902 |
| B1/B6/B7/B8 | bbox 回归误差诊断 | febc15f |
| C5 | 验证结果图生成 | 63e5739 |
| A7/A9 | _cls_weight 动态 + YOLO ID 统一 | 26a70cf |
| 问题1+2 | 扩张默认关闭 + 背景窗口限流64 | c73cbe6 |
| P0.5 | 统一 proposal 提取 | a56d202 |

## 待办（下次）
- A11-余：扩张系数重新标定（重训后）。
- A12 stale 注释、B4/B5 anchor 缓存/死代码标记、C4 GC10DET 隔离（P2）。
- validate 背景窗口处理与 train 对齐（新发现，见 HANDOFF 风险 3）。

## 下一步
重新训练 → 重测 bbox 偏差 → 决定扩张系数。

## 阻塞点
无。QUESTIONS.md 全部已确认。
