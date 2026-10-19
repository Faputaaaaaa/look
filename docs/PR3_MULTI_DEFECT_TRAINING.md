# PR3 v2：连接 PR2 Manifest 的多缺陷 GT-aware 训练

基准提交：`8bf034621ed9693e640198061e6e372bbbb89a9c`。PR1、PR2 已合入，**不要重复应用它们的补丁**。

此 PR 保留自研 ResNet18 + FPN + RPN + ROI Head，不引入外部完整检测框架。

## 改动

- `roi_training.py`：每个真实 GT 注入对应 ROI 提议；基于 ROI Align 的重排索引，保证每个 GT 都获得一个正样本，其他正负样本仍从 RPN 候选选；完全不修改真实推理。
- `manifest_training.py`：从 PR2 `quality.manifest` 读取已确认标注，严格校验 `quality.split` 的 `capture_group` / 未见 SKU 隔离。允许单图多实例多类别；窗口内同图多类 GT 可同时保留。
- 对跨窗严重截断的 GT：禁止把该窗口错当无缺陷负例；仅保留满足可见面积比例的已裁切 GT。默认负窗口每原图上限 2。
- `train.py`：`--manifest`、`--split-json`、`--data-root` 与 `--gt-aware-roi` 新入口；manifest 模式自动启用 GT-aware，不跑与本数据不一致的旧 val-metrics。
- `dataset.py`：不再静默截断超过 max_objects 的标签。
- `config.py`：显式实验开关 `use_gt_aware_roi_training=False`；原训练保持旧口径。
- `tests/test_roi_training.py` + `tests/test_manifest_training.py`：不依赖真实权重的采样、窗口坐标、数据划分和类别映射测试。

## 运行

```bash
python -m unittest discover -s tests -v
# 先按 docs/PR2_DATA_EVAL.md 生成 manifest 与 split
python train.py --manifest dataset_manifest.json --split-json experiments/group_split.json \
  --data-root D:/bottles --exp pr3_multi_sku --total-epochs 20
```

在 Windows 命令提示符中运行多行命令请改成单行，或使用 `^` 换行；`\` 用于 Unix shell。

## 重要约束

1. 默认旧训练路径不变；**开启 GT-aware 训练后 loss 变化，旧 checkpoint 不能直接作为新的精度结论**。
2. 避免把已有旧 `--resume` 的 optimizer/scheduler 状态拿来混合新的 manifest 训练；如需微调，应另行实现显式只加载网络权重的 `--init-weights`。
3. 严禁使用 `test_unseen_sku` 数据集训练或调阈值；旧 `split.json` 不会被覆盖。
4. 新 GT proposal 只用于训练/验证 loss，模型推理仍依赖 RPN；**如 RPN 无法召回缺陷，GT 注入本身无法修复推理时的召回问题**。
5. 目前没有真实联合缺陷标注与训练权重，因此 CPU 单测不能等同于端到端 mAP 提升、实时识别成功或跨瓶型泛化。
6. PR2 的 manifest `classes` 必须与现有 `config.class_names` 完全一致，当前模型只训练 `missing/dirty` 两类；扩展 7 类需独立重新配置和再训练。
7. 先复核 dirty 标注不一致问题，精度变化必须在干净的分组验证集上报告：每类 AP50、同瓶多缺陷全检率、正常瓶误检、RPN recall、端到端耗时。
8. 现有 train/validate 中仍计算旧式 ROI 正例样本再覆盖为 GT-aware 策略，主要是为了保留旧训练代码且减少改动风险；后续重构应抽出公共采样函数，避免维护两份实现。

## 验收状态

当前仅完成新模块的 CPU 单测及离线 patch 结构检查；**未在对方 Windows 4060 设备进行真实训练、GPU 回归或调参**。默认 `use_gt_aware_roi_training=False`，不会自动影响既有基线。
