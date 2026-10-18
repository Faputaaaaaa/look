# STATUS — 当前状态快照

> 快速一览，细节见 HANDOFF.md。

## 最新状态（2026-09-29）
- 训练 `overnight_20260927`（20 epoch）已完成，**mAP@0.5 = 0.7788**。
  - missing AP=0.917 (P/R=0.95)；dirty AP=0.640 (P/R=0.80)。
  - 报告: `experiments/overnight_20260927/TRAIN_REPORT.md` + `val_metrics.json`。
- **A14 dy 补偿实验已完成，结论：无净收益，默认关闭。**
  - 标定（val dirty 前 10 张）: -26.3px；验证（后 10 张，未参与定参）:
    TP/FP/FN 无变化（P=R=0.900），IoU 0.646→0.665。
  - 根因: dirty 偏移两极分化（一半 +0~5px、一半 +25~50px），非固定偏移；
    部分 FN 是形状问题（框瘦高），固定平移治不了。
  - 开关代码已就位并冒烟测试通过（`config.use_dy_compensation`，默认 False）。

## 待办（下次）
- 决定 dirty 提升方向：① 查标注一致性（推荐，成本最低）② 形状回归修复 ③ 接受现状。
- A11-余: missing 扩张若启用建议 1.08x（dirty 已无需扩张）。
- P2 收尾: A12 stale 注释、B4/B5 anchor 缓存、C4 GC10DET 隔离。

## 平台（defect_platform）— 2026-09-29
- 静态图检测平台已可用：`defect_platform\start_platform.bat` 双击启动
  （若报错，用 `D:\Python3.14\python.exe defect_platform\platform_ui.py`）。
- 不开界面也能看效果：`python defect_platform/smoke_test.py --n 1`
  → 出图到 `defect_platform/outputs/smoke_<ts>/`（全图/缩略/放大裁剪 + summary.txt）。
- 实测基线：单图 SAHI 21~25s（117 切片，4060 Laptop），missing IoU 0.74 / dirty 0.83。
- 已修：默认权重口径（原 `sahi_inference.MODEL_PATH` 指向旧实验）、UI 主线程假死、
  临时 jpg 压缩损失、RGBA 花屏；requirements 补 PyQt5 + opencv-python。
- 平台侧待办：速度优化（batch/FP16/stride）、一图多缺陷（Top-K 与 NMS 重新标定）、
  实时视频/相机（`detector.detect_bgr` 已是内存数组入口，可直接复用）。

## 阻塞点
无。
