"""确认轮最终评估 + 窗口探针（w800@30ep）。

流程：
1. w800@30ep best/latest 评估（审计协议：报 val-mAP 更高者）
2. 窗口探针：960/480 与 1120/560 纯推理（用 best 权重，不训练）——
   决策规则见 docs/PCB_PLAN.md：960 掉点 → 800 即最优；
   960 涨且 1120 掉 → 960 值得完整训练轮验证。

注意：探针存在上下文分布偏移（训练窗口 800），只用于决策门，
论文正式数字必须用训练窗口=推理窗口的模型。

用法：
    D:/Python3.14/python.exe tools/pcb_confirm_probe.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "exp_logs" / "pcb_confirm_probe.log"
BASE = ROOT / "experiments" / "pcb_sahi_w800" / "models"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


def run_eval(tag, model, size, stride, out):
    if out.exists():
        d = json.loads(out.read_text(encoding="utf-8"))
        log(f"{tag} 已存在（mAP50={d['eval']['mAP50']:.4f}），跳过")
        return 0
    log(f"评估 {tag} (window {size}/{stride}) ...")
    cmd = [sys.executable, "-m", "tools.evaluate_pcb",
           "--model", str(model), "--mode", "sahi",
           "--img-size", str(size), "--window-size", str(size),
           "--window-stride", str(stride), "--out", str(out)]
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
        log(f"{tag} 失败 rc={rc}（无结果文件）")
    return rc


def main():
    log("=" * 64)
    log("确认轮评估 + 窗口探针启动")
    # 1. 确认轮 best/latest
    run_eval("w800_30ep_best", BASE / "best_model.pth", 800, 400,
             ROOT / "data" / "pcb" / "eval_sahi_w800_confirm_best.json")
    run_eval("w800_30ep_latest", BASE / "latest.pth", 800, 400,
             ROOT / "data" / "pcb" / "eval_sahi_w800_confirm_latest.json")
    # 2. 窗口探针（best 权重，纯推理）
    run_eval("probe_960", BASE / "best_model.pth", 960, 480,
             ROOT / "data" / "pcb" / "eval_sahi_w800_probe960.json")
    run_eval("probe_1120", BASE / "best_model.pth", 1120, 560,
             ROOT / "data" / "pcb" / "eval_sahi_w800_probe1120.json")
    # 3. 窗口探针扩展（latest 权重——协议最优模型，800 窗口下 0.7403）
    run_eval("probe_960_latest", BASE / "latest.pth", 960, 480,
             ROOT / "data" / "pcb" / "eval_sahi_w800_probe960_latest.json")
    run_eval("probe_1120_latest", BASE / "latest.pth", 1120, 560,
             ROOT / "data" / "pcb" / "eval_sahi_w800_probe1120_latest.json")
    log("确认轮评估 + 窗口探针全部结束")


if __name__ == "__main__":
    main()
