"""PCB 全自动实验管线（无人值守，独立进程运行）。

流程：
1. 等待探索链（512/640/800 × 10 epoch，tools/run_pcb_sahi_chain.py 启动的
   独立训练）全部完成——按各实验 latest.pth 的 epoch 判定，不依赖链日志
2. 三档评估（best + latest，按审计协议报 val-mAP 更高者，见 docs/PCB_PLAN.md §5.1）
3. 自动选优胜档（mAP50 最高）→ 30 epoch 确认轮（从 10 epoch 续训，与
   resize 基线同轮数口径）
4. 确认轮评估 + 额外种子（seed=123）10 epoch 稳定性训练 + 评估
5. 全部结果落盘 data/pcb/autopilot_summary.json + 本日志

设计约束：
- 本脚本自身以独立进程启动（Start-Process 分离），不受会话内存回收影响
- 评估只在训练链结束后跑（无 GPU/内存竞争）
- 每步容错：单步失败记录进 summary 不中断后续

启动（独立进程）：
    powershell Start-Process -WindowStyle Minimized -FilePath "D:\\Python3.14\\python.exe"
        -ArgumentList "tools\\pcb_autopilot.py" -WorkingDirectory "D:\\deeplearning\\learn\\项目"
"""

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "exp_logs" / "pcb_autopilot.log"
SIZES = {512: 256, 640: 320, 800: 400}
EXPLORE_EPOCHS = 10
CONFIRM_EPOCHS = 30
SEED2 = 123


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ckpt_epoch(exp):
    """读 latest.pth 的 epoch 数（不存在返回 0）。"""
    p = ROOT / "experiments" / exp / "models" / "latest.pth"
    if not p.exists():
        return 0
    try:
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        return int(ck.get("epoch", 0))
    except Exception as e:
        log(f"  [warn] 读 {exp} checkpoint 失败: {e}")
        return 0


def wait_chain_done(timeout_h=44):
    """等三档 10-epoch 探索全部完成（按 checkpoint epoch 判定）。"""
    log(f"等待探索链完成（3 档 × {EXPLORE_EPOCHS} epoch，超时 {timeout_h}h）")
    deadline = time.time() + timeout_h * 3600
    while time.time() < deadline:
        states = {s: ckpt_epoch(f"pcb_sahi_w{s}") for s in SIZES}
        log("  进度: " + " ".join(f"w{s}={e}ep" for s, e in states.items()))
        if all(e >= EXPLORE_EPOCHS for e in states.values()):
            log("探索链全部完成")
            return states
        time.sleep(900)  # 15 分钟一轮
    log("[FAIL] 等待超时；按当前 checkpoint 状态继续（未完成的档评估会失败）")
    return {s: ckpt_epoch(f"pcb_sahi_w{s}") for s in SIZES}


def run_eval(size, model_name, model_path, out_path):
    """跑一次评估，返回 (mAP50, json_path)；失败返回 (None, None)。"""
    log(f"评估 w{size} {model_name} ...")
    cmd = [sys.executable, "-m", "tools.evaluate_pcb",
           "--model", str(model_path),
           "--mode", "sahi",
           "--img-size", str(size),
           "--window-size", str(size),
           "--window-stride", str(SIZES[size]),
           "--out", str(out_path)]
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                            encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONUTF8": "1"},
                            timeout=4 * 3600).returncode
    except subprocess.TimeoutExpired:
        log(f"  [FAIL] w{size} {model_name} 评估超时（>4h）")
        return None, None
    if rc != 0 or not out_path.exists():
        log(f"  [FAIL] w{size} {model_name} 评估失败 rc={rc}")
        return None, None
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        mAP = data["eval"]["mAP50"]
    except Exception as e:
        log(f"  [FAIL] w{size} {model_name} 结果解析失败: {e}")
        return None, None
    log(f"  w{size} {model_name}: mAP50={mAP:.4f}（{time.time() - t0:.0f}s）")
    return mAP, out_path


def eval_both(exp_tag, size, out_prefix):
    """按审计协议：best/latest 都评估，报 val-mAP 更高者。

    exp_tag: 实验后缀（如 'w512' / 'w640_s2'）；out_prefix: 结果 JSON 前缀。
    """
    exp = f"pcb_sahi_{exp_tag}"
    base = ROOT / "experiments" / exp / "models"
    entries = []
    for name in ("best", "latest"):
        p = base / f"{name}_model.pth"
        if not p.exists():
            continue
        mAP, jp = run_eval(size, name, p, ROOT / "data" / "pcb" / f"{out_prefix}_{name}.json")
        if mAP is not None:
            entries.append((mAP, jp))
    if not entries:
        return None
    return max(entries, key=lambda x: x[0])


def train(size, epochs, exp, seed=None, resume_ok=True):
    """训练一次，返回退出码。resume_ok 且 epoch<目标 时自动 --resume。"""
    latest = ROOT / "experiments" / exp / "models" / "latest.pth"
    done = ckpt_epoch(exp)
    cmd = [sys.executable, str(ROOT / "train.py"),
           "--dataset", "pcb", "--exp", exp,
           "--total-epochs", str(epochs),
           "--img-size", str(size), "--window-size", str(size),
           "--window-stride", str(SIZES[size]),
           "--preprocess", "sliding_window", "--max-neg-windows", "2"]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if done >= epochs:
        log(f"训练 {exp}: 已到 {done} epoch（≥ {epochs}），跳过")
        return 0
    if resume_ok and latest.exists():
        cmd += ["--resume", str(latest)]
        log(f"训练 {exp}: 从 epoch {done} 续训到 {epochs}")
    else:
        log(f"训练 {exp}: 全新（目标 {epochs} epoch, seed={seed}）")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    with open(ROOT / "exp_logs" / f"{exp}.log", "a", encoding="utf-8") as out:
        out.write(f"\n===== 自动管线启动 {datetime.now().isoformat()} =====\n")
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=out,
                            stderr=subprocess.STDOUT, env=env).returncode
        log(f"训练 {exp} 退出码 {rc}（{time.time() - t0:.0f}s）")
    return rc


def main():
    log("=" * 64)
    log("PCB 自动管线启动：探索评估 → 优胜确认(30ep) → 稳定性种子(seed2)")
    summary = {"started": datetime.now().isoformat(timespec="seconds"), "steps": {}}
    try:
        # 1. 等探索链
        states = wait_chain_done()
        summary["chain_state"] = states

        # 2. 三档评估
        results = {}
        for s in SIZES:
            try:
                r = eval_both(f"w{s}", s, f"eval_sahi_w{s}")
                results[str(s)] = {"mAP50": round(r[0], 4), "json": str(r[1])} if r else None
            except Exception:
                log(f"[FAIL] w{s} 评估异常:\n{traceback.format_exc()}")
                results[str(s)] = None
        summary["explore_eval"] = results

        valid = {int(s): v for s, v in results.items()
                 if v and v["mAP50"] is not None}
        if not valid:
            log("[FAIL] 三档评估全部失败，管线终止")
            summary["fatal"] = "no valid exploration evals"
        else:
            # 3. 选优胜
            winner = max(valid, key=lambda s: valid[s]["mAP50"])
            log(f"优胜档: w{winner} (mAP50={valid[winner]['mAP50']:.4f})")
            summary["winner"] = winner

            # 4. 确认轮 30 epoch（从 10 续训，与基线同轮数）
            rc = train(winner, CONFIRM_EPOCHS, f"pcb_sahi_w{winner}")
            summary["confirm_train_rc"] = rc
            if rc == 0:
                r = eval_both(f"w{winner}", winner, f"eval_sahi_w{winner}_confirm")
                summary["confirm_eval"] = {
                    "mAP50": round(r[0], 4), "json": str(r[1])} if r else None

            # 5. 稳定性种子（seed=123，10 epoch）
            rc2 = train(winner, EXPLORE_EPOCHS, f"pcb_sahi_w{winner}_s2",
                        seed=SEED2)
            summary["seed2_train_rc"] = rc2
            if rc2 == 0:
                r2 = eval_both(f"w{winner}_s2", winner, f"eval_sahi_w{winner}_s2")
                summary["seed2_eval"] = {
                    "mAP50": round(r2[0], 4), "json": str(r2[1])} if r2 else None
    except Exception:
        log(traceback.format_exc())
        summary["fatal"] = traceback.format_exc()
    finally:
        summary["finished"] = datetime.now().isoformat(timespec="seconds")
        out = ROOT / "data" / "pcb" / "autopilot_summary.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        log(f"管线结束，summary → {out}")


if __name__ == "__main__":
    main()
