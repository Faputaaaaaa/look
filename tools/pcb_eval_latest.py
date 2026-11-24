"""补评估：探索轮三档 latest.pth 的评估（autopilot 文件名 bug 修复后的补账）。

修复前 autopilot 拼了 latest_model.pth（实际是 latest.pth），探索轮只有
best 的评估结果。按审计协议（docs/PCB_PLAN.md §5.1）补跑 latest。
已存在的结果文件自动跳过，可重复运行。

用法（独立进程）：
    powershell Start-Process -WindowStyle Minimized -FilePath "D:\\Python3.14\\python.exe"
        -ArgumentList "tools\\pcb_eval_latest.py" -WorkingDirectory "D:\\deeplearning\\learn\\项目"
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZES = {512: 256, 640: 320, 800: 400}
LOG = ROOT / "exp_logs" / "pcb_eval_latest.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    for s, st in SIZES.items():
        out = ROOT / "data" / "pcb" / f"eval_sahi_w{s}_latest.json"
        if out.exists():
            log(f"w{s} latest 评估已存在，跳过")
            continue
        model = ROOT / "experiments" / f"pcb_sahi_w{s}" / "models" / "latest.pth"
        if not model.exists():
            log(f"[FAIL] w{s} latest.pth 不存在")
            continue
        log(f"评估 w{s} latest ...")
        cmd = [sys.executable, "-m", "tools.evaluate_pcb",
               "--model", str(model),
               "--mode", "sahi",
               "--img-size", str(s),
               "--window-size", str(s),
               "--window-stride", str(st),
               "--out", str(out)]
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"\n===== w{s} latest 补评估 =====\n")
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                                stderr=subprocess.STDOUT, env=env).returncode
        log(f"w{s} latest 评估 rc={rc}")
    log("补评估全部结束")


if __name__ == "__main__":
    main()
