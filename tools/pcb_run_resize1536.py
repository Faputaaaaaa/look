"""resize-1536 强对照训练（整图 letterbox，无 SAHI）。

目的：SAHI-960 的收益是否只来自「更高分辨率」？本对照把整图 resize 到
1536（缺陷 ~38px，接近 SAHI-960 窗口的有效信息量），其余全部同口径
（pcb profile、pcb_cand anchor、30 epoch、seed 42、审计协议评估）。

用法（独立进程）：
    powershell Start-Process -WindowStyle Minimized -FilePath "D:\\Python3.14\\python.exe"
        -ArgumentList "tools\\pcb_run_resize1536.py" -WorkingDirectory "D:\\deeplearning\\learn\\项目"
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "exp_logs" / "pcb_resize1536.log"

cmd = [sys.executable, str(ROOT / "train.py"),
       "--dataset", "pcb",
       "--exp", "pcb_resize1536",
       "--total-epochs", "30",
       "--img-size", "1536",
       "--preprocess", "resize"]
env = dict(os.environ)
env["PYTHONUTF8"] = "1"
with open(LOG, "a", encoding="utf-8") as out:
    out.write(f"\n===== resize1536 强对照训练 =====\n")
    rc = subprocess.run(cmd, cwd=str(ROOT), stdout=out,
                        stderr=subprocess.STDOUT, env=env).returncode
print(f"resize1536 训练退出码 {rc}")
