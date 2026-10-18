# 缺陷检测平台 — 静态图版

> 推理引擎：本项目自训练的 Faster R-CNN（`overnight_20260927`, mAP@0.5 = 0.7788）。
> 经 `detector.py` 复用 `sahi_inference.py` 的**完整 SAHI 管线**，平台上看到的框
> 与 `make_report.py` 验收时的框是同一条代码路径，口径一致。
>
> 当前阶段（用户设定）：**先做照片（静态图）**，实时视频/相机留到后续迭代。

## 启动

```bash
# 方式一：双击（自动挑一个装了 torch + PyQt5 的解释器）
defect_platform\start_platform.bat

# 方式二：命令行
D:\Python3.14\python.exe defect_platform\platform_ui.py
D:\Python3.14\python.exe defect_platform\platform_ui.py --model experiments\...\best_model.pth
```

> ⚠️ 本机有两个 Python：`D:\Python3.14`（装了 torch+cu126、PyQt5，**用这个**）和
> PATH 里先出现的其它 python（没装）。`start_platform.bat` 会逐个探测，
> 也可以设 `set PLATFORM_PY=D:\Python3.14\python.exe` 直接指定。
>
> 双击启动会带一个**黑色控制台窗口**（保留它是为了能看到 Python 报错），
> 关掉界面时它会一起退出，属正常现象。
>
> 📌 `start_platform.bat` 必须是 **GBK 编码 + CRLF 换行**。Windows 的 cmd.exe 解析
> LF-only 换行或 UTF-8 的 .bat 时 `call` / `goto` 会失效，表现为**双击后一闪而过**。
> 若用编辑器改过这个 bat，务必确认编码和换行；不确定就用上面的命令行方式启动。

## 两个冒烟脚本（不需要点界面）

| 脚本 | 作用 | 命令 |
|---|---|---|
| `smoke_test.py` | 跑 1~N 张图，直接产出**能看的效果图**（全图/缩略/放大裁剪 + summary.txt） | `python defect_platform/smoke_test.py --n 1` |
| `smoke_ui.py` | 不弹窗，把界面代码路径整条跑一遍（模型加载→读图→推理→渲染→统计），用于改 UI 后回归 | `python defect_platform/smoke_ui.py` |

`smoke_test.py` 常用参数：`--n 3`（每类 3 张）、`--dir <文件夹>`、`--topk 3`、
`--thresh 0.3`、`--expand`（开启 A11 扩张对比）、`--no-gt`。
产物落在 `defect_platform/outputs/smoke_<时间戳>/`。

## 界面功能

| 功能 | 说明 |
|---|---|
| 📂 打开图片 | 单张检测 |
| 📁 打开文件夹 | 载入图片列表，可逐张浏览 |
| ◀ / ▶ 上一张 / 下一张 | 切换图片；已检测过的会直接显示缓存结果 |
| 🚀 检测当前图 | SAHI 切片推理（后台线程，界面不卡，带进度条） |
| ⚡ 批量检测全部 | 顺序跑完列表；可随时「■ 停止」 |
| 💾 保存结果图 | 单张存一张；多张结果时自动存到 `outputs/batch_<时间戳>/` |
| 📋 导出明细 | tab 分隔 txt：图片 / 缺陷数 / 耗时 / 类别@置信度[框]|IoU |

右侧面板：
- **系统状态**：checkpoint、设备、耗时
- **推理参数**：置信度阈值滑块、Top-K（1/3/5）、A11 扩张、A14 dy 补偿、GT 叠加开关
- **检测统计**：总数 / missing / dirty
- **当前图明细**：类别、置信度、框尺寸、**与 GT 的 IoU**
- **图片列表**：每张图后面显示检出的缺陷数

## 文件说明

| 文件 | 作用 |
|---|---|
| `platform_ui.py` | 主程序（PyQt5；PySide6 可退回） |
| `detector.py` | 检测适配器：路径 / BGR ndarray / PIL 三种入参 → 统一 dict 输出 |
| `viz.py` | 图像 IO、画框、IoU 工具（**不依赖 Qt**，GUI 与脚本共用同一套画框逻辑） |
| `smoke_test.py` | 无界面推理冒烟，产出效果图 |
| `smoke_ui.py` | 无窗口界面冒烟 |
| `start_platform.bat` | 双击启动（自动探测解释器） |
| `outputs/` | 结果图输出目录（不进 git） |

## 依赖

项目 `requirements.txt` 已包含：`torch==2.12.0+cu126`、`PyQt5==5.15.11`、
`opencv-python==4.11.0.86`、`numpy`、`Pillow`。
torch 必须走 CUDA 官方源装（见 requirements.txt 头部注释），否则会装成 CPU 版。

## 性能基线（RTX 4060 Laptop 8GB）

- 模型加载 ≈ 1s；单张 4096×2736 原图 SAHI 推理 ≈ **21~25s**（117 个 640×640 切片）。
- 实测（val 集）：missing IoU 0.74 / dirty IoU 0.83（top-1，未开 A11/A14）。
- 后续优化方向（未做）：切片 batch 推理、FP16、GPU 内做切片预处理、增大 stride。

## 已知限制 / 下一步

1. **一图多缺陷**：默认 `config.max_dets_per_image = 1`（= 评估口径，数据集每图 1 个缺陷）。
   界面上可切 Top-K=3/5；真实产线若一图多缺陷，需重新标定该参数与 NMS。
2. **missing 框偏小 ~8%**：若启用 A11，建议把 `config.expand_missing` 从历史值 1.3
   改成 1.08 再验证（1.3x 是过补偿，已证）。
3. **dirty AP 0.64 的瓶颈是标注质量**（见 `experiments/label_check/`），不是模型。
4. **没有实时视频/相机**：下一步可接 RTSP/工业相机 + 抽帧复用本平台检测接口
   （`detector.detect_bgr` 已经是内存数组入口，直接可复用）。
5. 推理耗时 20s+ 对产线太慢，做视频前必须先解决速度问题（batch / 降 stride / TensorRT）。

## 技术要点（给后续维护者）

1. **import 顺序坑**：必须先 import torch（经 detector）再 import Qt，否则 Windows 上
   c10.dll 初始化失败（WinError 1114）。`platform_ui.py` 顶部有注释，勿调整。
2. **目录名坑**：不能叫 `platform/`（与 Python 标准库重名），故用 `defect_platform/`。
3. **推理跑在 QThread**：单图 20s+，早期版本在主线程跑会让窗口"未响应"。
   模型加载同样在后台线程（跨线程直接改 Qt 控件是未定义行为，已修）。
4. **默认 checkpoint**：统一来自 `config.default_ckpt`（→ 绝对路径），
   `sahi_inference.MODEL_PATH` 也由它派生，避免各脚本指向不同实验。
   覆盖顺序：`--model` > 环境变量 `SAHI_MODEL_PATH` > `config.default_ckpt`。
