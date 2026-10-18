# -*- coding: utf-8 -*-
"""[平台] 界面冒烟：不弹窗口，把 UI 的代码路径整条跑一遍。

覆盖：模型后台加载 → 打开图片 → 后台推理 → 渲染结果图 → 统计/表格 → 收尾。
改了 UI 之后跑一下这个，能在不手动点界面的情况下发现回归（比如线程信号没接对、
表格列数不一致、图片渲染报错）。

用法：
    python defect_platform/smoke_ui.py [--model xxx.pth] [--cls dirty]
"""
import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from defect_platform import platform_ui as pu  # noqa: E402

try:
    from PyQt5.QtCore import QTimer
except ImportError:
    from PySide6.QtCore import QTimer

STATE = {'t0': 0.0, 'img_name': ''}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--cls', default='dirty', choices=['dirty', 'missing'])
    a = ap.parse_args()

    app = pu.QApplication([])
    w = pu.PlatformUI(a.model)

    def fail(msg):
        print('❌', msg)
        app.exit(1)

    def step_wait_model():
        if w.detector is None:
            QTimer.singleShot(200, step_wait_model)
            return
        print(f'[1/5] 模型加载 OK: {Path(w.detector.model_path).name} | {w.detector.postproc_state()}')

        import sahi_inference as si
        sp = si.load_split()
        img_id = [i for i in sp['val'] if i.startswith(f'{a.cls}/')][0]
        f, _, _ = si.resolve_image(img_id)
        w._set_files([Path(f)])
        w.goto(0)
        if w.cur_bgr is None:
            fail('读图失败')
            return
        STATE['img_name'] = Path(f).name
        print(f'[2/5] 打开图片 OK: {img_id} shape={w.cur_bgr.shape} '
              f'GT={len(w.gts.get(Path(f).name, []))} 个')

        STATE['t0'] = time.time()
        w.run_single()
        QTimer.singleShot(300, step_wait_detect)

    def step_wait_detect():
        if w.worker is not None and w.worker.isRunning():
            QTimer.singleShot(300, step_wait_detect)
            return
        name = STATE['img_name']
        dets = w.results.get(name)
        if dets is None:
            fail('没有检测结果（线程可能报错，见上方 traceback）')
            return
        print(f'[3/5] 推理 OK: {len(dets)} 个缺陷, {time.time()-STATE["t0"]:.1f}s '
              f'→ {[d["class_name"] + "@" + format(d["confidence"], ".3f") for d in dets]}')

        if w.v2.pixmap() is None or w.v2.pixmap().isNull():
            fail('结果图未渲染到界面')
            return
        print(f'[4/5] 界面渲染 OK: 结果图 {w.v2.pixmap().width()}x{w.v2.pixmap().height()}, '
              f'明细表 {w.table.rowCount()} 行')

        expect_cols = w.table.columnCount()
        if expect_cols != 4:
            fail(f'明细表列数异常: {expect_cols}')
            return
        print('[5/5] 全部通过 ✅')
        app.quit()

    QTimer.singleShot(200, step_wait_model)
    return app.exec_() if hasattr(app, 'exec_') else app.exec()


if __name__ == '__main__':
    sys.exit(main())
