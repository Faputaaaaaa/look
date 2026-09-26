# HANDOFF.md — 交接状态

> 记录当前进度，供下次新会话快速恢复。恢复时先读本文件 + CLAUDE.md + DECISIONS.md。

## 当前阶段
**P0.1–P0.4 + AP 标准实现 已完成**。今日（2026-09-26）时间盒 15:00～20:00，做「正确性修复 + 训练前优化 + 可复现工具」，不训练、不全量 SAHI、不碰 P0.5 提案统一和死代码清理。

## 今日任务进度 (2026-09-26)
- [x] **A13 背景窗口负样本日志监控** —— `RPNTrainer.reset_stats()` + `compute_loss` 统计 + `train_one_epoch` 每 epoch 打印。
  - **关键发现**：背景窗口贡献 **99.7% 的负 anchor**（smoke：24 窗口，正 102,984 vs 负 5,391,384，其中背景窗口负 5,376,000）。背景窗口负样本未限流（每背景窗口全量 ~672k 负 anchor），见「风险」。
- [x] **A10 窗口内 bbox 裁剪** —— `_get_bboxes_in_window` clip 到 `[0,640]`、clip 后 <4px 丢弃；`sahi_inference` 扩张后 clip 回原图。smoke：144 个窗口 GT 越界 0，其中 85 个贴边（说明 clip 确实在起作用，修前这些 GT 会越界）。
- [ ] **C1 requirements.txt + C2 config 落盘**
- [ ] **C3 checkpoint 续训**
- [ ] **B1 根因排查 + B6/B7/B8 诊断**
- [ ] **C5 验证结果图生成**
- [ ] **A7/A8/A9 顺手项**

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
  - AP 计算已按标准 VOC 修好（见下方「AP 标准实现」），不再留小瑕疵。
- **P0.3** 修 warmup（`train.py:WarmupCosineScheduler`）。
  - 旧实现：`step()` 在 warmup 期间只 `current_epoch+=1`，从不写 optimizer；`get_last_lr()` 返回假斜坡。
  - 新实现：捕获各参数组 target_lr（RPN 5x / 其他 1x），warmup 期间线性从 ~0 升到 target 并**真正写入 optimizer 两个参数组**；`get_last_lr()` 返回真实 lr。
  - 冒烟测试通过：隔离测试 lr 从 ~0→0.5×→1×→余弦退火；模型 forward+backward+optimizer.step+scheduler.step 无报错。
- **AP 标准实现**（用户确认）：`evaluate_map` 用 VOC all-point AP，锚定 recall=0（precision 补 1）、recall=1（precision 补 0）。
  - 新增 `voc_ap()` 助手；冒烟测试纯 TP 得 AP=1.0（旧版 np.trapezoid 只得 0.75）。
  - **注意：改后 AP/mAP 数字不可与旧版直接比，旧版系统性偏低。**
- **P0.4** 背景窗口参与 RPN 分类损失（`train.py:train_one_epoch`）。
  - 旧：`n_obj==0` 整个跳过背景窗口，负样本没参与训练。新：空 GT 参与 RPN cls/bbox 损失，跳过 ROI Head；`num_valid_samples` 计入背景窗口。
  - 冒烟测试通过：12 窗口前向+反向，total_loss=5.81 无 NaN。
  - 注意：背景窗口 RPN 负样本目前是全量负 anchor（`sample_anchors` num_pos=0 时不截断），需加日志监控（见 P1）。

## 未完成（按优先级）

### P0（必须先做）
- ~~**P0.1** 按原图 80/20 划分 train/val，`seed=42`，落盘 `split.json`。~~ ✅ 已完成
- ~~**P0.2** 修 mAP 评估 train/val 泄露（`sahi_inference.py:evaluate_map`：改为只读 `split.json` 的 val 原图）。~~ ✅ 已完成
- ~~**P0.3** 修 warmup（`WarmupCosineScheduler` 真正更新两个参数组 lr，而非只改日志）。~~ ✅ 已完成
- ~~**P0.4** 背景窗口参与 RPN 分类损失，ROI 正样本跳过（改 `train.py` 的 `n_obj==0` 跳过逻辑）。~~ ✅ 已完成
- **P0.5** 统一 proposal 提取逻辑（train / validate / inference 三处重复，抽成公共函数）。

### P1
- `_cls_weight` 动态化（`train.py` 当前硬编码 3 类，换 GC10DET 会崩）。
- 1.3x / 1.1x 扩张配置化。
- 补齐 `requirements.txt`、config 落盘、训练中 mAP 追踪。
- `np.trapezoid` 兼容（NumPy 2.0 才有的 API，旧版是 `np.trapz`）。
- YOLO ID 映射统一为单一事实来源（现散落 4 处）。
- RPN 负样本采样加日志监控（背景窗口 num_pos=0 时全量负 anchor，需观察召回率是否异常再调 pos:neg）。

### P2
- stale 注释清理（只标记，不删死代码）。
- anchor 缓存（避免每 batch 重复生成）。
- 死代码标记（`multi_class_focal_loss`、`_spatial_uniform_sample` 等）。
- GC10DET 残余隔离收尾。

## 下一步
**P0.5**：统一 proposal 提取逻辑（train / validate / inference 三处重复，抽成公共函数）。

## 关键决策
全部按 CLAUDE.md 的默认规则执行（见 DECISIONS.md 决策表）。

## 风险（当前未修复）
1. **proposal 逻辑三处重复** —— 改一处易漏另外两处（train / validate / inference）。
2. **背景窗口负样本未限流** —— num_pos=0 时 `sample_anchors` 返回全量负 anchor，可能淹没正样本；需加日志监控负采样情况。

## 下次恢复方式
1. 开新会话。
2. 先读 HANDOFF.md、CLAUDE.md、DECISIONS.md。
3. **不要重读全部代码，不要 `/resume` 旧对话**。
4. 从 P0.4 开始。
