# STATUS — 当前状态快照

> 快速一览，细节见 HANDOFF.md。

## 最新状态（2026-10-19）
- **多缺陷训练 PR3 v2 已合入**（基于 8bf0346；opt-in，默认不影响旧训练口径）：
  - `roi_training.py`：GT proposal 注入 + GT 覆盖采样（仅训练/验证 loss，推理仍只用 RPN）；
    `manifest_training.py`：PR2 manifest → 训练数据（同图多实例、截断缺陷不误标背景、test_unseen_sku 禁入训练）。
  - `train.py` 新入口 `--manifest/--split-json/--data-root/--gt-aware-roi`；manifest 模式禁止 `--resume`/旧 val-metrics。
  - `dataset.py` 超 max_objects 不再静默截断；`config.use_gt_aware_roi_training=False`。
  - 验收：单测 49/49（PR1 7 + PR2 16 + PR3 26）；train.py --help 冒烟通过。
  - 注意：补丁 3 处 hunk 因空行空格差异 git apply 失败，已按 .rej 手工等价应用。
  - 已推：ACAMLab/Defect-Supervised（c4d312b）、Faputaaaaaa/look（003072c，快进）。
  - **待真实多缺陷 manifest 到位后**：`python train.py --manifest ... --split-json ... --data-root ... --exp pr3_multi_01`，并用 PR2 工具按新口径评估。
- **多缺陷评测 PR2 已合入**（纯新增模块，不改 split.json/训练/平台）：
  - `quality/`（manifest 契约 + 类别感知多实例 AP50/P/R + 分组隔离划分 + 整 SKU 留出）、
    `tools/`（build_group_split / export_multi_predictions / evaluate_multi_instance）、`docs/`、16 项单测。
  - 验收：单测 16/16、全套 23/23；合成 CLI 联调（多类别全检率 1.0）与真实图+overnight 权重全链路联调通过。
  - 待数据到位：制作真实多缺陷 manifest（见 `docs/PR2_DATA_EVAL.md` 合同）。
- **多缺陷推理 PR1 已合入**（为"一图多缺陷"数据集做准备，当前单缺陷数据下行为中性）：
  - 窗口内 NMS → 类别感知 NMS（不同类重叠框共存）；窗口枚举去重 117→96 切片（单图 15~17s）；
  - 移除 `final >= MAX_DET*2` 提前截断；新增 `inference_utils.py` + 7 个无权重回归测试。
  - 回归：单测 7/7、smoke_ui 5/5、smoke_test（missing IoU 0.742 / dirty 0.825，与打补丁前一致）。
  - 顺手修复 smoke_ui.py GBK 控制台 emoji 打印崩溃（退出码 127）。
- **GitHub 同步（2026-10-19）**：`Faputaaaaaa/look`（公开）main @ 003072c（PR1+PR2+PR3 快进合入）；
  `ACAMLab/Defect-Supervised`（私有）main @ c4d312b（PR1+PR2+PR3，历史已用 filter-branch 抹除 `.trae/` 个人笔记）。
  - 已上传：核心代码、`defect_platform/`、诊断脚本、`requirements.txt`、文档。
  - 未上传：数据集、`experiments/`、`exp_logs/`、平台 `outputs/`、`.workbuddy/`。
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
- 多缺陷数据集到位后：制作 manifest（`docs/PR2_DATA_EVAL.md` 合同）→ 分组划分 → 导出预测 → 评测；Top-K 标定（`config.max_dets_per_image`）。
- 决定 dirty 提升方向：① 查标注一致性（推荐，成本最低）② 形状回归修复 ③ 接受现状。
- A11-余: missing 扩张若启用建议 1.08x（dirty 已无需扩张）。
- P2 收尾: A12 stale 注释、B4/B5 anchor 缓存、C4 GC10DET 隔离。

## 平台（defect_platform）— 2026-09-29
- 静态图检测平台已可用：`defect_platform\start_platform.bat` 双击启动
  （若报错，用 `D:\Python3.14\python.exe defect_platform\platform_ui.py`）。
- 不开界面也能看效果：`python defect_platform/smoke_test.py --n 1`
  → 出图到 `defect_platform/outputs/smoke_<ts>/`（全图/缩略/放大裁剪 + summary.txt）。
- 实测基线：单图 SAHI 15~17s（96 切片，去重后；原 117 切片 21~25s），missing IoU 0.74 / dirty 0.83。
- 已修：默认权重口径（原 `sahi_inference.MODEL_PATH` 指向旧实验）、UI 主线程假死、
  临时 jpg 压缩损失、RGBA 花屏；requirements 补 PyQt5 + opencv-python。
- 平台侧待办：速度优化（batch/FP16/stride）、一图多缺陷 Top-K 标定（NMS/窗口/截断三处已修，见最新状态）、
  实时视频/相机（`detector.detect_bgr` 已是内存数组入口，可直接复用）。

## 阻塞点
无。
