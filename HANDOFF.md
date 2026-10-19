# HANDOFF.md — 交接状态

> 记录当前进度，供下次新会话快速恢复。恢复时先读本文件 + CLAUDE.md + DECISIONS.md + STATUS.md。

## 当前阶段
2026-09-27 夜间训练 `overnight_20260927`（20 epoch）已完成并出报告（TRAIN_REPORT.md + val_metrics.json，mAP@0.5=0.7788）。2026-09-29 完成 A14 dy 补偿实验（标定+诚实验证，结论：无净收益，保持默认关闭）。

## 已完成（新增 2026-10-19）
- **多缺陷训练 PR3 v2**（交付包 `look-PR3-v2-适配最新提交-8bf0346`，opt-in，默认不影响旧训练口径）：
  1. `roi_training.py`：训练/验证 loss 专用 **GT proposal 注入 + GT 覆盖采样**——每个真实缺陷保证一个正例（防"极差 proposal 保底错标正类"）；**推理仍只用 RPN**。
  2. `manifest_training.py`：从 PR2 manifest + group split 构建训练数据：同图多实例多类别、跨窗严重截断缺陷**不误标背景**、负窗每图限 2、`test_unseen_sku` 禁入训练；复用 `unique_sahi_windows` 与 SAHI 同口径。
  3. `train.py`：`--manifest/--split-json/--data-root/--gt-aware-roi` 入口；manifest 模式自动开 GT-aware、禁止 `--resume` 与旧 val-metrics 混用；默认无参数走旧路径。
  4. `dataset.py` 超 `max_objects` 从静默截断改为报错；`config.use_gt_aware_roi_training=False`。
  5. 验收：**单测 49/49**（PR1 7 + PR2 16 + PR3 26）；train.py --help 冒烟通过。
  6. 补丁 3 处 hunk（config.py、train.py parse_args、main 数据加载）因空行空格差异 `git apply` 失败，已按 .rej 手工等价应用并删除 .rej。
  7. 已推：ACAMLab/Defect-Supervised（c4d312b）、Faputaaaaaa/look（003072c，快进）。
- **多缺陷推理 PR1**（微信文件 `look-pr1-multidefect-inference.patch`，为后续"一图多缺陷"数据集做准备）：
  1. `model_fpn.py` 窗口内 NMS → **类别感知 NMS**（`class_aware_keep`）：不同类重叠框共存，同类重复才抑制。
  2. `sahi_inference.py` 窗口枚举 → `unique_sahi_windows` 去重：4096×2736 下 117 切片 → **96 切片**（老代码靠边窗口重复推理 ~43%），覆盖范围不变；单图耗时 21~25s → 15~17s。
  3. 移除 `if len(final) >= MAX_DET * 2: break` 提前截断（原会漏掉后面的缺陷类别）。
  4. 新增 `inference_utils.py` + `tests/test_inference_utils.py`（7 个无权重回归测试，`python -m unittest discover -s tests -t . -v`）。
- **平台冒烟修复**：`smoke_ui.py` 的 ✅/❌ emoji 在 GBK 控制台打印 UnicodeEncodeError → 5 步全过但退出码 127；已换 ASCII 标记。
- **回归通过**：单测 7/7；`smoke_ui.py` 5/5 退出码 0；`smoke_test.py --n 1` → 96 切片/张，missing IoU 0.742 / dirty 0.825（与打补丁前一致）。效果图 `defect_platform/outputs/smoke_20261019_172330/`。
- **结论：当前单缺陷数据集下该补丁行为中性**（不同类框重叠基本不发生、top-k 兜底不变），无需重新评估 mAP 口径；等新数据集到位后需做 Top-K 标定（`config.max_dets_per_image`）与按新数据重定评估口径。
- **多缺陷评测 PR2**（交付包 `look-PR1-PR2-阶段改造交付包.zip` 的 02 补丁）：纯新增独立模块，**不改 split.json / train.py / model_fpn.py / 平台**：
  1. `quality/`：`manifest.py`（多缺陷 GT manifest 契约校验）、`metrics.py`（类别感知多实例匹配 + VOC all-point AP50 + 多实例/多类别全检率、正常瓶误报率）、`split.py`（按 capture_group 隔离分组划分 + 整 SKU 留出 heldout + 泄漏检测）。
  2. `tools/`：`build_group_split.py`（生成分组划分，拒绝覆盖 split.json）、`export_multi_predictions.py`（走 `sahi_inference_image` 单一口径，低 score_floor 导出）、`evaluate_multi_instance.py`（离线评测，无需 torch）。
  3. `docs/PR2_DATA_EVAL.md` + `docs/技术架构方案.md`（五阶段路线：PR3 多缺陷数据/训练、PR4 批处理实时化、PR5 未见瓶型泛化）。
  4. `tests/test_quality.py` 16 项单测。
- **PR2 验收**：单测 16/16、全套 23/23、patch `git apply --check` 通过；CLI 联调：合成 manifest（一图两类+留出 SKU）build_group_split/evaluate 通过（multi_class_all_found=1.0）；真实图+overnight 权重 export/evaluate 全链路通过（top 检出 dirty@0.999 命中 GT，score_floor=0.05 下 7 个低分 FP 属预期）。**注意：真实多缺陷标注的 manifest 尚未制作**，等数据到位后按 `docs/PR2_DATA_EVAL.md` 的 contract 建。

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
2. 多缺陷数据集到位后：制作 manifest（按 `docs/PR2_DATA_EVAL.md` 合同）→ `tools.build_group_split` 分组划分 → 用 `train.py --manifest ... --split-json ... --data-root ...` 训练（自动 GT-aware）→ 训练后用 PR2 工具按新口径评估（multi_class_all_found_rate 等）。
3. PR3 训练前先复核 dirty 标注一致性；GT 注入不解决 RPN 召回问题，训练后要报告 RPN recall。
3. 如继续训练：`python train.py --exp <name>`（改动已就位）。
4. P2 收尾项见上。

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
