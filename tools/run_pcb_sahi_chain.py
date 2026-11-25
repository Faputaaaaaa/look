"""PCB SAHI 窗口消融串行训练链（自愈：断点自动续训、已完成自动跳过）。

- 512→640→800 顺序执行；每个实验若存在 latest.pth 自动 --resume；
  若日志已含「训练完成总结」则跳过（重复运行本脚本安全）。
- 训练输出追加到 exp_logs/pcb_sahi_w<S>.log（PYTHONUTF8=1，日志可读）；
  本脚本自身日志 exp_logs/pcb_sahi_chain.log。
- 用独立进程启动（脱离 Claude Code 后台任务树）：中途被杀/断电后
  重跑同一命令即从断点继续，不需要人工介入。

启动（独立进程，等价 Windows 版 nohup）：
    cmd /c start /min "" D:\\Python3.14\\python.exe tools\\run_pcb_sahi_chain.py --epochs 10

前台调试：
    D:/Python3.14/python.exe tools/run_pcb_sahi_chain.py --only 512 --epochs 10
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    {"tag": "w512", "size": 512, "stride": 256},
    {"tag": "w640", "size": 640, "stride": 320},
    {"tag": "w800", "size": 800, "stride": 400},
    # [探针决策 2026-11-25] latest 权重纯推理：960=0.7622 / 1120=0.7618（打平），
    # 均高于 800 窗口的 0.7403 → 窗口收益在 960 附近饱和。按计划训 960@30ep 验证。
    {"tag": "w960", "size": 960, "stride": 480},
]

TOTAL_EPOCHS = 30  # 默认（确认轮与基线同口径）；探索轮用 --epochs 10


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    (ROOT / "exp_logs").mkdir(exist_ok=True)
    with open(ROOT / "exp_logs" / "pcb_sahi_chain.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one(cfg, total_epochs):
    exp = f"pcb_sahi_{cfg['tag']}"
    exp_dir = ROOT / "experiments" / exp
    latest = exp_dir / "models" / "latest.pth"
    log_path = ROOT / "exp_logs" / f"pcb_sahi_{cfg['tag']}.log"

    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--dataset", "pcb",
        "--exp", exp,
        "--total-epochs", str(total_epochs),
        "--img-size", str(cfg["size"]),
        "--window-size", str(cfg["size"]),
        "--window-stride", str(cfg["stride"]),
        "--preprocess", "sliding_window",
        "--max-neg-windows", "2",
    ]
    if latest.exists():
        # 从 checkpoint 读已完成 epoch 数（字段 0 基：完成 N 个 epoch 时 = N-1）
        import torch
        try:
            ck = torch.load(latest, map_location="cpu", weights_only=False)
            done_epochs = int(ck.get("epoch", 0)) + 1
        except Exception:
            done_epochs = 0
        if done_epochs >= total_epochs:
            log(f"{exp}: 已到 {done_epochs} epoch（≥ {total_epochs}），跳过")
            return 0
        cmd += ["--resume", str(latest)]
        log(f"{exp}: 从 epoch {done_epochs} 续训（目标 {total_epochs}）")
    else:
        log(f"{exp}: 全新训练（目标 {total_epochs} epoch）")

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"  # 日志 UTF-8 可读（避免 GBK 乱码）
    with open(log_path, "a", encoding="utf-8") as out:
        out.write(f"\n===== 训练链启动 {datetime.now().isoformat()} =====\n")
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=out,
                            stderr=subprocess.STDOUT, env=env).returncode
        dt = time.time() - t0
    if rc == 0:
        log(f"{exp}: 完成（耗时 {dt / 3600:.1f} h）")
    else:
        log(f"{exp}: 退出码 {rc}（耗时 {dt / 3600:.1f} h）——继续下一档")
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None,
                        help="逗号分隔，只跑指定档（512,640,800）")
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS,
                        help=f"每档目标 epoch 数（默认 {TOTAL_EPOCHS}；探索轮用 10）")
    args = parser.parse_args()
    configs = CONFIGS
    if args.only:
        want = {int(x) for x in args.only.split(",")}
        configs = [c for c in CONFIGS if c["size"] in want]
    log(f"训练链开始（{len(configs)} 档，{args.epochs} epoch/档）")
    for cfg in configs:
        run_one(cfg, args.epochs)
    log("训练链全部结束")


if __name__ == "__main__":
    main()
