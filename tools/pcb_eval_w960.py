"""w960@30ep 协议评估：best + latest（审计协议，报 val-mAP 更高者）。

结果文件存在自动跳过，可重复运行。独立进程启动（不受会话内存回收影响）。
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "exp_logs" / "pcb_eval_w960.log"
BASE = ROOT / "experiments" / "pcb_sahi_w960" / "models"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


def run_eval(tag, model, out):
    out = Path(out)
    if out.exists():
        d = json.loads(out.read_text(encoding="utf-8"))
        log(f"{tag} 已存在（mAP50={d['eval']['mAP50']:.4f}），跳过")
        return
    log(f"评估 {tag} ...")
    cmd = [sys.executable, "-m", "tools.evaluate_pcb",
           "--model", str(model), "--mode", "sahi",
           "--img-size", "960", "--window-size", "960",
           "--window-stride", "480", "--out", str(out)]
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n===== {tag} =====\n")
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                            stderr=subprocess.STDOUT, env=env).returncode
    if out.exists():
        d = json.loads(out.read_text(encoding="utf-8"))
        log(f"{tag} 完成 rc={rc}  mAP50={d['eval']['mAP50']:.4f}  "
            f"推理 {d['infer_time_s']['total']:.0f}s / 显存峰值 "
            f"{d['gpu_peak_vram_mb']:.0f}MB")
    else:
        log(f"{tag} 失败 rc={rc}")


def main():
    log("=" * 60)
    run_eval("w960_30ep_best", BASE / "best_model.pth",
             ROOT / "data" / "pcb" / "eval_sahi_w960_best.json")
    run_eval("w960_30ep_latest", BASE / "latest.pth",
             ROOT / "data" / "pcb" / "eval_sahi_w960_latest.json")
    log("w960 协议评估全部结束")


if __name__ == "__main__":
    main()
