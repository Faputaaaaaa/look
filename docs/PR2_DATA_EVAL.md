# PR2：多缺陷评估 + 未见瓶型验证的最小可信基线

本 PR **只新增独立模块，不改变** `train.py`、`model_fpn.py`、原来的 `split.json`、`make_report.py` 及静态图平台，从而可以独立审查与回退；**不宣称精度提升**。

## 与旧版的差别

旧版按 `dirty/frame_XXXX.jpg` 与 `missing/frame_XXXX.jpg` 单类别目录做 160/40 原图切分，且最终 `max_dets_per_image=1`。旧报告的 mAP@0.5=0.7788 是单缺陷验证口径。新模块为**不同位置多个缺陷**提供 class-aware instance TP/FP/FN、AP50、全检率、严格集合正确率、**同瓶多实例 / 多类别全检率**与正常瓶误报率，并支持整瓶型未见（heldout SKU）评测。**不可将这两个 mAP 当作严格可比结果。**

## Manifest 合同（需人工真实标注，不可把无标签当作正常）

```json
{
  "classes": ["missing", "dirty"],
  "images": [
    {
      "image_id": "sku_A/a1.png",
      "path": "sku_A/a1.png",
      "sku_id": "sku_A",
      "capture_group": "physical_bottle_0001_session_01",
      "defects": [
        {"class_name": "missing", "bbox": [12, 20, 40, 56]},
        {"class_name": "dirty", "bbox": [90, 40, 110, 80]}
      ]
    },
    {
      "image_id": "sku_B/b1.png",
      "path": "sku_B/b1.png",
      "sku_id": "sku_B",
      "capture_group": "physical_bottle_0002_session_02",
      "defects": []
    }
  ]
}
```

每行一张原始**完整图**，多个独立实例用多个 bbox。坐标是左上角为原点的原图绝对像素 `xyxy`。同物理瓶的所有帧、裁剪来源和不同采集 session **必须共用 `capture_group`**（示例中的 ID 仅用来区分物理瓶，不应按 session 给同一瓶重新编号）。一个 SKU 可以有很多 group；整 SKU 留出才能用于严格“未见瓶型”评价。未标注完成的图片不应列入此 manifest。实例框若互相重叠可共存，匹配按类别分别做。不支持同一框真实多标签，这是后续独立研究方向。

## 命令（从仓库根目录执行）

```bash
python -m unittest discover -s tests -p 'test_quality.py' -v

# 提供至少两个非留出采集组；留出 SKU 的样本不得出现在 train/val。
python -m tools.build_group_split --manifest dataset_manifest.json \
  --out experiments/group_split.json --heldout-sku sku_B --seed 42

# 必须有真实数据与 checkpoint；仍调用仓库唯一的 SAHI 推理函数。
python -m tools.export_multi_predictions --manifest dataset_manifest.json \
  --split-json experiments/group_split.json --partition val --data-root D:/bottles \
  --model experiments/overnight_20260927/models/best_model.pth \
  --score-floor 0.05 --max-dets 30 --out experiments/multi_predictions.json

python -m tools.evaluate_multi_instance --manifest dataset_manifest.json \
  --predictions experiments/multi_predictions.json --split-json experiments/group_split.json \
  --partition val --score 0.5 --out experiments/multi_metrics.json
```

`python -m tools...` 只要求 `tools/` 文件夹可按 Python namespace package 导入，推荐以仓库根目录为工作目录；这里不依赖 PyQt。

## 重要已知限制

1. 这只提供多实例**评测**与数据拆分；还没有升级 ROI 训练策略、更没有改进模型能力。`max_dets=30` 不代表切片内候选不受 `INFERENCE_CONFIG['max_detections']=5` 限制。
2. SAHI 若尚未合并 PR1，现有 ROI NMS 可能抑制跨类别重叠框；因此建议 PR1→PR2 顺序合并，但 PR2 的纯离线评估/数据拆分可先用。
3. 导出必须选低 `score_floor`，否则 AP 会受到被丢弃预测的影响；`--score` 是可调操作阈值，只影响 P/R 与瓶级验收，不改变离线 AP 计算。
4. 各 SKU 的类别数不同或验证集无该类时，某类 AP 返回 `null`，mAP 只计算有 GT 的类别。跨 SKU 可比较前先对齐类别分布。
5. split 只是按 `capture_group` 隔离，不保证每类别分层充分或每个 SKU 均有足够样本；优先让可划分的缺陷类同时出现在 train/val，并通过 `coverage_warnings` 显式报告不可满足的类别覆盖。少样本实验要汇报每 SKU 类别计数和置信区间，不能声称“已经验证泛化”。
6. 本 PR 的纯 Python 单测不要求真实图片或权重；完整精度和速度验收需要本机 CUDA 环境与真实图片。
7. 代码使用高质量开源算法研究思想，但不复制第三方仓库实现。参考 [SAHI](https://github.com/obss/sahi)、[EfficientAD](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html)；异常分支将在 PR5 单独研究。

## 多缺陷专用 KPI 说明

- `n_multi_instance`：GT 含 2 个以上缺陷实例的图片数；`multi_instance_all_found_rate`：这些图片的所有实例都被匹配的比例。
- `n_multi_class`：GT 至少含 2 种缺陷类别的图片数；`multi_class_all_found_rate`：这些图片的所有实例都被匹配的比例。**这是项目近期主目标的关键指标。**
- `multi_class_exact_set_rate` 还要求**没有额外误检**；即便多个 GT 都检出，有多余预测也不算完整正确。
- 当对应类别的测试图片为零时，相应 rate 为 `null`，不是 0% 或 100%；**不能用单缺陷验证集替代多类别共存验证集**。
