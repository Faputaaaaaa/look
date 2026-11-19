"""PCB SAHI 窗口消融串行训练链（自愈：断点自动续训、已完成自动跳过）。

- 512→640→800 顺序执行；每个实验若存在 latest.pth 自动 --resume；
  若日志已含「训练完成总结」则跳过（重复运行本脚本安全）。
- 训练输出追加到 exp_logs/pcb_sahi_w<S>.log（PYTHONUTF8=1，日志可读）；
  本脚本自身日志 exp_logs/pcb_sahi_chain.log。
- 用独立进程启动（脱离 Claude Code 后台任务树）：中途被杀/断电后
  重跑同一命令即从断点继续，不需要人工介入。

启动（独立进程，等价 Windows 版 nohup）：
    cmd /c start /min "" D:\\Python3.14\\python.exe tools\\run_pcb_sahi_chain.py

前台调试：
    D:/Python3.14/python.exe tools/run_pcb_sahi_chain.py --only 512
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
]

TOTAL_EPOCHS = 30  # 与 resize 基线统一（docs/PCB_PLAN.md 消融矩阵口径）


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    (ROOT / "exp_logs").mkdir(exist_ok=True)
    with open(ROOT / "exp_logs" / "pcb_sahi_chain.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one(cfg):
    exp = f"pcb_sahi_{cfg['tag']}"
    exp_dir = ROOT / "experiments" / exp
    latest = exp_dir / "models" / "latest.pth"
    log_path = ROOT / "exp_logs" / f"pcb_sahi_{cfg['tag']}.log"

    if log_path.exists() and "训练完成总结" in log_path.read_text(
            encoding="utf-8", errors="ignore"):
        log(f"{exp}: 已完成（日志含完成标记），跳过")
        return 0

    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--dataset", "pcb",
        "--exp", exp,
        "--total-epochs", str(TOTAL_EPOCHS),
        "--img-size", str(cfg["size"]),
        "--window-size", str(cfg["size"]),
        "--window-stride", str(cfg["stride"]),
        "--preprocess", "sliding_window",
        "--max-neg-windows", "2",
    ]
    if latest.exists():
        cmd += ["--resume", str(latest)]
        log(f"{exp}: 发现 latest.pth，断点续训")
    else:
        log(f"{exp}: 全新训练")

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
    args = parser.parse_args()
    configs = CONFIGS
    if args.only:
        want = {int(x) for x in args.only.split(",")}
        configs = [c for c in CONFIGS if c["size"] in want]
    log(f"训练链开始（{len(configs)} 档，{TOTAL_EPOCHS} epoch/档）")
    for cfg in configs:
        run_one(cfg)
    log("训练链全部结束")


if __name__ == "__main__":
    main()
