# HANDOFF.md — 交接状态

> 记录当前进度，供下次新会话快速恢复。恢复时先读本文件 + CLAUDE.md + DECISIONS.md + STATUS.md。

## 当前阶段
2026-09-27 夜间训练 `overnight_20260927`（20 epoch）已完成并出报告（TRAIN_REPORT.md + val_metrics.json，mAP@0.5=0.7788）。2026-09-29 完成 A14 dy 补偿实验（标定+诚实验证，结论：无净收益，保持默认关闭）。

## 已完成（2026-09-27 及之前）
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
- **夜间训练 + 无人值守工具** —— `run_all.sh`（自动恢复循环）+ `make_report.py`；validate 对齐背景窗口；checkpoint 含 scheduler/RNG。

## 已完成（新增 2026-09-29）
- **后处理补跑** —— run_all.sh 当夜后处理未执行，已用 best_model.pth（epoch 15）补跑 diagnose_bbox / evaluate_and_visualize / make_report。
- **A14 dy 补偿实验** —— config 加 `use_dy_compensation`/`dy_comp_dirty` 开关 + sahi_inference 平移代码（冒烟测试通过）。标定协议：val dirty 前 10 张定参（-26.3px）、后 10 张验证。**验证结论：TP/FP/FN 无变化（P=R=0.900），IoU 仅 0.646→0.665，无净收益，默认关闭。** 临时脚本 `_diag_train_dy.py`、`_a14_calib_verify.py`，日志 `exp_logs/diag_train_dy.log`、`exp_logs/a14_calib_verify.log`。
- **A15 dirty 标注质量核查** —— **实锤：dirty 标注是固定尺寸模板框（w std=13px、77% 在中位±15px 内），且 y 贴放两极分化（~40% 图 GT 高 15~50px）**。24 张需复核（15 张贴高 + 10 张形状离群 + frame_0042 形状不匹配）。报告+清单+裁剪图: `experiments/label_check/`（LABEL_CHECK_REPORT.md / dirty_gt_stats.txt / crop_*.jpg）。脚本 `_check_dirty_labels.py`。**dirty AP 0.64 的瓶颈是标注质量，调模型无解。**

## 关键数据（overnight_20260927, best_model.pth）
- mAP@0.5=0.7788；missing AP=0.917 (P/R=0.95)；dirty AP=0.640 (P/R=0.80)。
- missing dy≈-4px（无偏移）；dirty dy 两极分化：一半图 +0~5px、一半 +25~50px（std≈18），**不是统一偏移**。
- dirty 的 4 个 FN 中至少 1 个（frame_0042）dy≈0 但 IoU=0.43——形状不匹配（框瘦高），与平移无关。
- 背景窗口负 anchor 占比 9.1%（限流生效）；train loss 5.06→2.26。

## 未完成（按优先级）
- **dirty 标注修正**（A15 已确认根因）：按 `experiments/label_check/LABEL_CHECK_REPORT.md` 清单人工复核 24 张（A 组 15 张下移、B 组 10 张重画、frame_0042 重点），修正后重训。不修正则接受 dirty 0.64 现状。
- **A11-余 扩张系数重新标定** —— 按本次诊断：dirty w/h 比≈0.98 已正常，无需扩张；missing 偏小 ~8%，若启用建议 1.08x（非历史 1.3x）。
- **A12** stale 注释清理（只标记不删）。
- **B4/B5** anchor 缓存 + 死代码标记（P2）。
- **C4** GC10DET 残余隔离收尾（P2）。

## 下一步
1. 决定 dirty 提升方向（见上"未完成"第 1 条，建议先查标注一致性，成本最低）。
2. 如继续训练：`python train.py --exp <name>`（改动已就位）。
3. P2 收尾项见上。

## 风险（当前未解决）
1. **dirty 偏移两极分化未解决**：非固定偏移，固定平移补偿无效（A14 已证伪）；部分 FN 是形状问题（框瘦高）而非平移。
2. **C3 resume 不恢复 scheduler 状态**（LR 余弦重启，非阻断）。
3. **轻量验证与全量评估协议不一致**：训练中每 3 epoch 的 val-metrics 与事后 make_report 的 mAP 差异大（0.27 vs 0.78，轻量版 missing 记 0），对比时注意口径。

## 关键决策
见 DECISIONS.md。QUESTIONS.md 全部已确认（含第 5、6 条：扩张默认关闭、背景窗口限流 64）。新增：A14 补偿标定值 -26.3px 但验证无净收益 → 默认关闭（2026-09-29）。

## 下次恢复方式
1. 开新会话。
2. 先读 HANDOFF.md、CLAUDE.md、DECISIONS.md、STATUS.md。
3. 不要重读全部代码，不要 `/resume` 旧对话。
4. 从「下一步」开始。

## 平台（2026-09-29 新增）
- **defect_platform/** —— 静态图缺陷检测平台 MVP，已跑通全流程冒烟测试。
  - 源自微信 defect demo 包（F:\微信下载\...\defect\defect，YOLO+海康相机），
    以 mainUI.py 为骨架迭代，推理引擎换成项目自训练 Faster R-CNN。
  - `detector.py` 适配器复用 sahi_inference 完整管线（与 mAP 0.7788 评估同口径，自测结果逐字段一致）。
  - `platform_ui.py`：打开图片/文件夹批量 → SAHI 检测 → 画框 + 右侧信息面板（统计/明细/A11/A14 开关）→ 保存结果图 + 导出 txt。
  - `start_platform.bat` 双击启动。新依赖：PyQt5。
  - **两个坑已记录在 defect_platform/README.md**：①import 必须 torch 先于 Qt（c10.dll WinError 1114）；②目录不能叫 platform/（与标准库重名）。
  - 待办：实时相机/RTSP 版（原包有现成参考）、线扫相机接入、性能优化。

## 平台二次审查（2026-09-29 晚）
- 审查结论：首版 MVP 与主项目**融合是对的**（同一 sahi_inference 代码路径，框逐字段一致），
  但工程上有明显问题，已全部修掉：
  1. **口径坑（最重要）**：`sahi_inference.MODEL_PATH` 默认指向旧实验 `exp_20260812_093621`，
     `python make_report.py` 单独跑会静默用旧权重 → 与 STATUS 的 0.7788 不一致。
     现改为从 `config.default_ckpt`（overnight_20260927 best）派生为绝对路径。
  2. **UI 假死**：单图 SAHI 21~25s（117 切片）在主线程跑 → 窗口"未响应"。
     现推理与模型加载都在 QThread，带切片级进度条与「停止」。
  3. **临时文件**：`detect_pil` 写临时 jpg（quality=95）。现直推内存图
     `detect_bgr` / `sahi_inference_image`，零磁盘 IO、零压缩损失。
  4. 其他：RGBA 图花屏（IMREAD_COLOR）、窗口缩放不重绘、批量循环引用未定义变量、
     跨线程改 Qt 控件、top-1 硬编码（→ `config.max_dets_per_image`）、
     阈值不可调（→ 滑块）、无 GT 对比（→ 绿框 + IoU 列）。
- 新增：`viz.py`（无 Qt 依赖的画框/IO 工具）、`smoke_test.py`（出效果图）、
  `smoke_ui.py`（界面回归）、`defect_platform/__init__.py`。
- 环境：须用 `D:\Python3.14\python.exe`（torch+cu126、PyQt5 都在它上面）；
  `start_platform.bat` 改为自动探测装了 torch+PyQt5 的解释器。
