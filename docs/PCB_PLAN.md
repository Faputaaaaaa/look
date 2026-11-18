# PCB 基准线实验计划（PCB_PLAN）

> 公开基准（PKU-Market-PCB）负责「能和前人横向比」；瓶身集负责「真实场景真的有用」。
> 两条腿并行、互不替代。本文档是 PCB 线的唯一计划来源，每阶段结束勾选。

## 1. 定位与铁律

- **不改瓶身线**：xinpaide 配置/划分/实验口径（0.7788）一个字节不动，PCB 全部通过
  `dataset_profile` 开关隔离，默认仍是 xinpaide。
- **不按切片划分**：用数据集官方板级隔离划分（板 01,04,05,06,07,08,09 → train；
  10,11,12 → val），防板级泄漏（同板图像 99.99% 相同，按图随机分会虚高）。
- **评估禁止用训练图**：一切评估只读 val（152 图，板 10/11/12）。
- **指标主口径**：mAP@0.5 + 每类 AP50 + micro P/R；辅助：相对面积分档
  [<0.05%, 0.05~0.10%, >0.10%]（不硬套 COCO AP_small）。
- **PCB 与瓶身数字分表呈现，不直接比。**

## 2. 数据集档案（2026-11-18 实测）

- 版本：**PKU-Market-PCB-corrected**（KeenForgeAI，HF DOI 10.57967/hf/10554）。
  图像与上游逐字节一致；修正的是打包（YOLO/VOC 格式 + 板级隔离划分）。
  原版 693（COCO-only + 泄漏划分）仅留作对照。
- 引用：Huang & Wei, arXiv:1901.08204 (2019) + corrected 版（论文两篇都引）。
- 划分：train 541 / val 152；泄漏断言已通过并落盘 `data/pcb/pcb_split.json`
  （train/val 文件名不相交、板 ID 不相交、693 图全覆盖）。
- 实测（train）：框 2185 个（val 768）；每图 1~6 框，均值 4.04（**多实例**）；
  sqrt(面积) 中位 64.9px / p10 43.4 / p90 87.9；面积占比中位 0.0686%；
  10 块板 = 10 种图像尺寸（2240×2016 ~ 3056×2464）→ **必须 per-image 读尺寸**。

## 3. 已定决策（锚框匹配率脚本的产出）

`data/pcb/anchor_match.json`，关键行：

| 模式 | 输入 | anchor 集 | cov>=0.3 | cov>=0.5 | IoU 中位 |
|---|---|---|---|---|---|
| resize | 640 | xinpaide_cur | **0.218** | 0.033 | 0.208 |
| resize | 640 | pcb_cand | 0.931 | 0.821 | 0.731 |
| **resize** | **1024** | **pcb_cand** | **1.000** | **0.981** | **0.790** |
| sahi | 640 | pcb_cand | 1.000 | 0.997 | 0.798 |

**决策**：
1. **PCB anchor 集 = `pcb_cand`**：P2[16,24,32,48,64] P3[32,48,64,96,128]
   P4[64,96,128,192,256]，ratio [0.5,1,2]（每层 15 anchor）。resize 与 sahi
   两种模式**共用同一套**（消融只变一个变量）。`pcb_cand_plus`（+ratio 3）
   中位 IoU 只 +0.004，弃用。
2. **基线输入 = 整图 letterbox 1024**（resize_1024 下 GT 像素中位 23.4px，
   p10 15.4 / p90 31.7——已属 COCO「small」以下，这正是小缺陷研究点的证据）。
3. **SAHI 消融列 = 640 原生窗口**（GT 保持 43~88px）。
4. 论文论据：xinpaide_cur 在 resize_640 下只有 21.8% 的 GT 能过 pos 阈值——
   「现有大 anchor + 缩小输入」会饿死 RPN，这是小 anchor + 滑窗路线的量化动机
   （E04/E05 证据素材）。

## 4. 阶段清单

### Phase 0 — 数据与协议冻结（E01）【2026-11-18 完成】
- [x] `tools/prepare_pcb.py`：ImageSets → `data/pcb/pcb_split.json`（板 ID、
      每图尺寸、归一化框、泄漏断言、统计）
- [x] `tools/check_pcb_anchors.py`：全 train GT 在 {640,1024}×3 套 anchor 下
      匹配率 → 定 img_size=1024 + anchor=pcb_cand
- [x] 协议冻结：板级划分 541/152、mAP50 主指标、seed=42、评估只用 val 板

### Phase 1 — 干净基线 + 错检分析（E02/E03/E04/E05）
- [x] config 加 `dataset_profile`（默认 xinpaide 不变）+ `train.py --dataset pcb`
- [x] `dataset.py`：修 per-image 尺寸（`_get_bboxes_in_window` 改用实际图尺寸）、
      file-list 加载（读 pcb_split.json）、负窗限流配置化、未知类跳过加警告
- [x] `model_fpn.py`：FPN_CONFIG_PCB + `get_active_fpn_config()`（6 处引用切换）；
      INFERENCE_CONFIG_PCB（max_detections=20）+ RPN 公开 anchor_configs 属性
- [x] `sahi_inference.py`：MODEL_CLS_TO_NAME 从 config.class_names 派生；
      扩张/dy 补偿加 `dataset_profile=='xinpaide'` 守卫
- [ ] 基线训练：resize_1024、无 SAHI、无 GT-aware、负窗不限流、30 epoch
      （运行中：`exp/pcb_baseline_resize1024`，日志 `exp_logs/pcb_baseline_resize1024.log`）
- [x] `tools/evaluate_pcb.py`：推理 → predictions JSON → `quality/metrics.py`
      （per-class AP50 + micro P/R + 相对面积分档）
- [ ] **错检分类表**：按 类 × 尺寸档 × 失败场景（贴边/重叠/低分背景窗）统计
      FN/FP，导出可视化 crop——创新点的原材料
- [ ] E05 分辨率对照：resize_640（pcb_cand）vs resize_1024，量化分辨率贡献

### Phase 2 — 消融与创新点（E06~E09）
- [ ] 消融矩阵（每行独立重训）：基线 → +SAHI(640原生窗) → +anchor 细化 → +GT-aware
- [ ] **强对照行**：整图 resize 到与 SAHI 等效分辨率（如 1536）——证明收益来自
      滑窗机制本身，不只是分辨率
- [ ] 创新点从 Phase 1 错检分析中定，先验证再写

### Phase 3 — 主实验与验证（E10~E14）
- [ ] PCB 主对比表：文献数字 + 我们的基线 + 我们的方法（注明划分/输入尺寸口径；
      有余力再复现一个 YOLO11n）
- [ ] 跨数据集验证：PCB 上有效的改进 → 瓶身集验证（E11，瓶身口径不动）
- [ ] 效率表：参数量/FLOPs/单图耗时（PR4 平台实测数据）
- [ ] 多种子 3 次（核心对照）

### Phase 4 — 可选加固（E16~E18，P2）
- [ ] 光照/噪声鲁棒性扰动实验（瓶身拍照质量差场景加分）
- [ ] 产线实测演示（PR4 实时平台）

## 5. 协议纪律（每阶段必守）

- dev/val 分离：日常迭代用 train 里留出的板（建议板 01）当 dev；**val（板
  10/11/12）只在定版时跑一次**写进论文。
- 操作阈值纪律：在 dev/val 上选固定置信度阈值 → 独立测试集量误报/漏检
  （`quality/metrics.py` 的 `operating_score` 就是干这个的）。
- 每阶段收尾：git commit + 更新 HANDOFF/STATUS + `config.save_config()` 快照。
- 运行环境：`D:\Python3.14\python.exe`；数据脚本用 `-I` 隔离模式。

## 6. 交付物清单（论文素材）

1. 图1 数据集分析（每类框数、尺寸分布、面积占比分档——素材已在
   `data/pcb/pcb_split.json`）
2. 表1 主实验对比（per-class AP50 + mAP50 + 参数量/耗时，PCB 与瓶身分表）
3. 表2 消融 + 简单替代对照（含「等效分辨率整图」强对照行）
4. 图3 按缺陷尺寸分档的性能曲线
5. 图4 准确率-延迟（PR4 平台实测）
6. 图5 可视化对比（同图：GT / 基线 / Ours + 失败案例放大）
7. 图6 逐类改善/退化（不只平均成绩）
8. 论文结构参照「4. Experiments」八节模板（Datasets→Implementation→Comparison→
   Ablation→Challenging defects→Generalization→Efficiency→Qualitative）
