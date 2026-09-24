# DECISIONS.md — 默认决策表

> 与 CLAUDE.md 保持一致。遇到不确定的决策，优先采用下表默认值。

| 主题 | 默认决策 |
|---|---|
| 数据划分 | 按原图 80/20，`seed=42`，落盘 `split.json`；禁止按切片随机分 |
| 原图 ID 表示 | `{子集}/{文件名}`，如 `dirty/frame_0000.jpg`（dirty/missing 文件名重名，需子集前缀唯一化） |
| 评估指标 | mAP@0.5，SAHI 合并后按原图算，报告每类 AP / precision / recall |
| 评估数据 | 只准用验证集 / 测试集，禁用训练图 |
| 背景窗口 | 保留，参与 RPN 分类损失；ROI 正样本跳过 |
| Warmup | 保留并修好，真正更新两个参数组 lr |
| 后处理扩张 | 1.3x(missing) / 1.1x(dirty) 保留，配置化 |
| 死代码 | 不删，只标记 |
| 数据集范围 | GC10DET 残余隔离，默认只跑 missing / dirty |
| 长任务 | tmux / nohup + 日志写文件 |
| 阶段收尾 | 每阶段更新 HANDOFF.md / STATUS.md 并 git commit |
