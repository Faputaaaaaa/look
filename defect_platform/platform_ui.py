# -*- coding: utf-8 -*-
"""[平台融合 MVP] 静态图缺陷检测平台。

推理引擎：项目自训练 Faster R-CNN（overnight_20260927, mAP@0.5=0.7788），
经 defect_platform/detector.py 复用 sahi_inference 完整 SAHI 管线（与评估同口径）。

2026-09-29 审查后重写（相对首版 MVP 的改动）:
1. **推理搬到 QThread**：单图 SAHI ~117 切片、GPU 上约 20s，旧版在主线程跑会导致
   界面假死（Windows 标题栏"未响应"）。现改为后台线程 + 切片级进度 + 可中止。
2. **模型加载也走后台线程**（旧版用 threading.Timer 直接改 Qt 控件，跨线程不安全）。
3. **直推内存图像**：走 detector.detect_bgr → sahi_inference_image，不再写临时 jpg。
4. **文件列表 + 上一张/下一张**：打开文件夹后可逐张浏览、单独检测。
5. **参数可调**：置信度阈值滑块、Top-K、A11 扩张 / A14 dy 补偿开关。
6. **GT 叠加对比**：图片来自数据集时自动找 YOLO 标签，绿框画 GT 并算 IoU，
   便于迭代时直观判断"偏了还是漏了"。
7. 修掉：PNG/RGBA 花屏（统一 IMREAD_COLOR）、窗口缩放不重绘、批量循环外引用
   未定义变量等缺陷。

依赖: PyQt5（或 PySide6） + opencv-python + 项目 requirements.txt
启动: 双击 start_platform.bat 或 python defect_platform/platform_ui.py
"""
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

# [重要] 必须先加载 torch（经 detector→sahi_inference）再加载 Qt：
# Windows 上 Qt 先加载会与 torch 的 c10.dll 冲突（WinError 1114）。勿调整此顺序。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from defect_platform.detector import FasterRCNNDetector  # noqa: E402

# 优先 PyQt5，退回 PySide6
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QFileDialog, QSplitter, QRadioButton, QGroupBox,
        QGridLayout, QFormLayout, QMessageBox, QTableWidget, QTableWidgetItem,
        QHeaderView, QListWidget, QProgressBar, QSlider, QComboBox, QCheckBox,
        QAbstractItemView,
    )
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    QT_LIB = 'PyQt5'
except ImportError:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QFileDialog, QSplitter, QRadioButton, QGroupBox,
        QGridLayout, QFormLayout, QMessageBox, QTableWidget, QTableWidgetItem,
        QHeaderView, QListWidget, QProgressBar, QSlider, QComboBox, QCheckBox,
        QAbstractItemView,
    )
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtCore import Qt, QThread
    from PySide6.QtCore import Signal as pyqtSignal
    QT_LIB = 'PySide6'

from defect_platform.viz import (  # noqa: E402
    CLASS_COLORS, IMG_EXTS, OUT_DIR, imread_u, imwrite_u,
    load_gts, best_gt_iou, draw_vis,
)


class ImageView(QLabel):
    """可自适应缩放重绘的图像标签。"""

    def __init__(self, placeholder=''):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background:#151922; border:1px solid #384152; color:#666")
        self._bgr = None
        self._ph = placeholder

    def set_image(self, bgr):
        self._bgr = bgr
        self._render()

    def clear_image(self):
        self._bgr = None
        self.clear()
        self.setText(self._ph)

    def _render(self):
        if self._bgr is None:
            return
        rgb = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()


# ==================== 后台线程 ====================
class LoadWorker(QThread):
    """模型加载（约 1~3s），放后台避免启动瞬间卡顿。"""
    sig_ok = pyqtSignal(object)
    sig_err = pyqtSignal(str)

    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            self.sig_ok.emit(FasterRCNNDetector(self.model_path))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.sig_err.emit(f'{type(e).__name__}: {e}')


class DetectWorker(QThread):
    """SAHI 推理后台线程：items = [(name, bgr), ...]。"""
    sig_item = pyqtSignal(int, str, object, float)   # idx, name, dets, seconds
    sig_slice = pyqtSignal(int)                      # 全局进度百分比 0~100
    sig_done = pyqtSignal(int, float)                # 完成张数, 总耗时
    sig_err = pyqtSignal(str)

    def __init__(self, detector, items, score_thresh=None, max_dets=None):
        super().__init__()
        self.detector = detector
        self.items = items
        self.score_thresh = score_thresh
        self.max_dets = max_dets
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        t_all = time.time()
        n_ok = 0
        n_items = max(1, len(self.items))
        for idx, (name, bgr) in enumerate(self.items):
            if self._stop:
                break

            def _prog(done, total, _i=idx):
                if self._stop:
                    return
                self.sig_slice.emit(min(100, int(100.0 * (_i + done / max(1, total)) / n_items)))

            try:
                t0 = time.time()
                dets = self.detector.detect_bgr(
                    bgr, score_thresh=self.score_thresh, max_dets=self.max_dets,
                    progress=_prog,
                )
                self.sig_item.emit(idx, name, dets, time.time() - t0)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.sig_err.emit(f'{name}: {type(e).__name__}: {e}')
                break
        self.sig_done.emit(n_ok, time.time() - t_all)


# ==================== 主界面 ====================
class PlatformUI(QMainWindow):
    def __init__(self, model_path=None):
        super().__init__()
        self.setWindowTitle("缺陷检测平台 — Faster R-CNN + SAHI（静态图）")
        self.resize(1400, 880)
        self.setMinimumSize(1200, 760)

        self.detector = None
        self.model_path_arg = model_path
        self.files = []            # list[Path]
        self.cur_idx = -1
        self.cur_bgr = None
        self.results = {}          # name -> list[dict]
        self.gts = {}              # name -> [(x1,y1,x2,y2,cls)]
        self.times = {}            # name -> seconds
        self.worker = None
        self.loader = None

        self._init_ui()
        self.load_model()

    # ---------------- UI ----------------
    def _init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        root = QVBoxLayout(main)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- 工具栏 ----
        tool = QHBoxLayout()
        self.btn_open = QPushButton("📂 打开图片"); self.btn_open.clicked.connect(self.open_file)
        self.btn_dir = QPushButton("📁 打开文件夹"); self.btn_dir.clicked.connect(self.open_dir)
        self.btn_prev = QPushButton("◀ 上一张"); self.btn_prev.clicked.connect(lambda: self.goto(self.cur_idx - 1))
        self.btn_next = QPushButton("下一张 ▶"); self.btn_next.clicked.connect(lambda: self.goto(self.cur_idx + 1))
        self.run_btn = QPushButton("🚀 检测当前图"); self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_single)
        self.batch_btn = QPushButton("⚡ 批量检测全部"); self.batch_btn.setEnabled(False)
        self.batch_btn.clicked.connect(self.run_batch)
        self.stop_btn = QPushButton("■ 停止"); self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detect)
        self.btn_save = QPushButton("💾 保存结果图"); self.btn_save.clicked.connect(self.save_res)
        self.btn_export = QPushButton("📋 导出明细"); self.btn_export.clicked.connect(self.export_txt)
        for b in (self.btn_open, self.btn_dir, self.btn_prev, self.btn_next, self.run_btn,
                  self.batch_btn, self.stop_btn, self.btn_save, self.btn_export):
            tool.addWidget(b)
        tool.addStretch()

        # ---- 中部：左视图 + 右面板 ----
        disp = QSplitter(Qt.Horizontal)
        left = QSplitter(Qt.Horizontal)
        self.v1 = ImageView("原图")
        self.v2 = ImageView("检测结果")
        left.addWidget(self.v1)
        left.addWidget(self.v2)
        left.setStretchFactor(0, 1)
        left.setStretchFactor(1, 1)

        panel = QWidget()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(360)
        ply = QVBoxLayout(panel)
        ply.setContentsMargins(4, 4, 4, 4)

        # 系统状态
        g1 = QGroupBox("系统状态")
        l1 = QVBoxLayout()
        self.lbl_model = QLabel("模型: 加载中..."); self.lbl_model.setWordWrap(True)
        self.lbl_device = QLabel("设备: -")
        self.lbl_state = QLabel("状态: 请打开图片")
        self.lbl_time = QLabel("耗时: -")
        for w in (self.lbl_model, self.lbl_device, self.lbl_state, self.lbl_time):
            l1.addWidget(w)
        g1.setLayout(l1); ply.addWidget(g1)

        # 推理参数
        g5 = QGroupBox("推理参数")
        f5 = QFormLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(5, 95)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(self._on_thresh)
        self.lbl_thresh = QLabel("阈值 0.10")
        f5.addRow("置信度", self.slider)
        f5.addRow("", self.lbl_thresh)
        self.cmb_topk = QComboBox()
        self.cmb_topk.addItems(["1（评估口径）", "3", "5"])
        f5.addRow("Top-K", self.cmb_topk)
        from config import config as _cfg
        self.r_a11_off = QRadioButton("关（默认）"); self.r_a11_off.setChecked(True)
        self.r_a11_on = QRadioButton(f"开 {_cfg.expand_missing}x/{_cfg.expand_dirty}x")
        box_a11 = QHBoxLayout(); box_a11.addWidget(self.r_a11_off); box_a11.addWidget(self.r_a11_on)
        f5.addRow("A11 扩张", box_a11)
        self.r_a14_off = QRadioButton("关（默认）"); self.r_a14_off.setChecked(True)
        self.r_a14_on = QRadioButton("开 -26px")
        box_a14 = QHBoxLayout(); box_a14.addWidget(self.r_a14_off); box_a14.addWidget(self.r_a14_on)
        f5.addRow("A14 dy", box_a14)
        self.chk_gt = QCheckBox("叠加 GT 绿框（有标签时）")
        self.chk_gt.setChecked(True)
        self.chk_gt.stateChanged.connect(self._render_result)
        f5.addRow("", self.chk_gt)
        g5.setLayout(f5); ply.addWidget(g5)

        # 检测统计
        g2 = QGroupBox("检测统计")
        l2 = QVBoxLayout()
        self.lbl_total = QLabel("总缺陷数: 0")
        self.lbl_missing = QLabel("missing: 0"); self.lbl_missing.setStyleSheet("color:#ffc800")
        self.lbl_dirty = QLabel("dirty: 0"); self.lbl_dirty.setStyleSheet("color:#ff5050")
        for w in (self.lbl_total, self.lbl_missing, self.lbl_dirty):
            l2.addWidget(w)
        g2.setLayout(l2); ply.addWidget(g2)

        # 明细
        g3 = QGroupBox("当前图明细")
        l3 = QVBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["类别", "置信度", "尺寸(px)", "IoU(GT)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        l3.addWidget(self.table)
        g3.setLayout(l3); ply.addWidget(g3, 1)

        # 文件列表
        g6 = QGroupBox("图片列表")
        l6 = QVBoxLayout()
        self.listw = QListWidget()
        self.listw.currentRowChanged.connect(self._on_list_row)
        l6.addWidget(self.listw)
        g6.setLayout(l6); ply.addWidget(g6, 1)

        ply.addStretch()
        disp.addWidget(left)
        disp.addWidget(panel)
        disp.setStretchFactor(0, 3)
        disp.setStretchFactor(1, 1)

        # ---- 底部状态 ----
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(260)
        self.lbl_status = QLabel("初始化...")
        self.lbl_status.setStyleSheet("background:#1b2029; color:#ddd; padding:5px")
        bottom.addWidget(self.progress)
        bottom.addWidget(self.lbl_status, 1)

        root.addLayout(tool)
        root.addWidget(disp, 1)
        root.addLayout(bottom)
        # 开关改动后需重新检测才生效，给个明确提示（放在最后连接，避免初始化误触发）
        for r in (self.r_a11_on, self.r_a14_on, self.r_a11_off, self.r_a14_off):
            r.toggled.connect(self._warn_recheck)
        self.slider.valueChanged.connect(self._warn_recheck)
        self.cmb_topk.currentIndexChanged.connect(self._warn_recheck)
        self._apply_qss()

    def _warn_recheck(self, *_a):
        if self.results:
            self._set_status("⚠ 参数已改：对已检测图片需重新检测才会更新")

    def _apply_qss(self):
        self.setStyleSheet("""
        QWidget { background:#0f1115; color:#ddd; font-family:Microsoft YaHei UI; font-size:12px; }
        QLabel { background:transparent; border:none; color:#ccc; }
        QGroupBox { border:1px solid #384152; border-radius:6px; margin-top:10px;
                    padding-top:8px; font-weight:bold; color:#8ab4f8; }
        QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
        QPushButton { background:#232834; border:1px solid #384152; padding:6px 12px;
                      border-radius:4px; }
        QPushButton:hover { background:#2b3140; }
        QPushButton:disabled { background:#1b2029; color:#666; }
        QTableWidget, QListWidget { background:#151922; border:1px solid #384152;
                                    gridline-color:#2a2f3a; }
        QHeaderView::section { background:#232834; color:#ccc; border:none; padding:4px; }
        QRadioButton, QCheckBox { padding:2px; }
        QProgressBar { background:#151922; border:1px solid #384152; border-radius:4px;
                       text-align:center; color:#ccc; }
        QProgressBar::chunk { background:#3d7eff; }
        """)

    # ---------------- 模型 ----------------
    def load_model(self):
        self.lbl_status.setText("模型加载中...")
        self.loader = LoadWorker(self.model_path_arg)
        self.loader.sig_ok.connect(self._on_model_ok)
        self.loader.sig_err.connect(self._on_model_err)
        self.loader.start()

    def _on_model_ok(self, det):
        self.detector = det
        st = det.postproc_state()
        self.lbl_model.setText(f"模型: {Path(det.model_path).name}")
        self.lbl_device.setText(f"设备: {st['device']} | 窗口 {st['window/stride']}")
        self.run_btn.setEnabled(True)
        self._set_status(f"✅ 模型就绪 | 阈值 {st['score_thresh']} | Top-K {st['top_k']} | "
                         f"A11 {st['A11 bbox扩张']} | A14 {st['A14 dy补偿']}")

    def _on_model_err(self, msg):
        self._set_status(f"❌ 模型加载失败: {msg}")
        self.lbl_model.setText("模型: 加载失败")
        QMessageBox.critical(self, "模型加载失败",
                             f"{msg}\n\n检查：① 是否用装了 torch 的解释器启动 "
                             f"② config.default_ckpt 路径是否存在")

    # ---------------- 打开 / 导航 ----------------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not path:
            return
        self._set_files([Path(path)])
        self.goto(0)

    def open_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not d:
            return
        files = sorted([p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS])
        if not files:
            QMessageBox.information(self, "提示", "该文件夹里没有图片")
            return
        self._set_files(files)
        self.goto(0)

    def _set_files(self, files):
        self.files = files
        self.results = {}
        self.gts = {}
        self.times = {}
        self.listw.clear()
        for p in files:
            self.listw.addItem(p.name)
        self.batch_btn.setEnabled(len(files) > 1)

    def _on_list_row(self, row):
        if 0 <= row < len(self.files) and row != self.cur_idx:
            self.goto(row)

    def goto(self, idx):
        if not self.files or idx < 0 or idx >= len(self.files):
            return
        self.cur_idx = idx
        self.listw.blockSignals(True)
        self.listw.setCurrentRow(idx)
        self.listw.blockSignals(False)

        p = self.files[idx]
        img = imread_u(p)
        if img is None:
            self._set_status(f"❌ 无法读取: {p.name}")
            return
        self.cur_bgr = img
        self.v1.set_image(img)

        name = p.name
        if name not in self.gts:
            self.gts[name] = load_gts(p)
        if name in self.results:
            self._render_result()
            self._update_stats(self.results[name])
            self.lbl_time.setText(f"耗时: {self.times.get(name, 0):.1f}s")
            self.lbl_state.setText(f"状态: {name}（已检测）")
        else:
            self.v2.clear_image()
            self._update_stats([])
            self.lbl_time.setText("耗时: -")
            self.lbl_state.setText(f"状态: {name} 已加载 | GT {len(self.gts[name])} 个")
        self._set_status(f"{idx+1}/{len(self.files)}  {p}")

    # ---------------- 检测 ----------------
    def _apply_postproc(self):
        from config import config
        config.use_bbox_expansion = self.r_a11_on.isChecked()
        config.use_dy_compensation = self.r_a14_on.isChecked()

    def _thresh(self):
        return self.slider.value() / 100.0

    def _topk(self):
        return [1, 3, 5][self.cmb_topk.currentIndex()]

    def _on_thresh(self, _v=None):
        self.lbl_thresh.setText(f"阈值 {self._thresh():.2f}")

    def run_single(self):
        if self.cur_bgr is None:
            QMessageBox.information(self, "提示", "请先打开图片")
            return
        if self.detector is None:
            QMessageBox.information(self, "提示", "模型尚未就绪")
            return
        self._start_worker([(self.files[self.cur_idx].name, self.cur_bgr)], batch=False)

    def run_batch(self):
        if not self.files:
            return
        if self.detector is None:
            QMessageBox.information(self, "提示", "模型尚未就绪")
            return
        cur_name = self.files[self.cur_idx].name if (self.cur_idx >= 0 and self.cur_bgr is not None) else None
        items = []
        for p in self.files:
            img = self.cur_bgr if p.name == cur_name else imread_u(p)
            if img is not None:
                items.append((p.name, img))
        if not items:
            return
        self._start_worker(items, batch=True)

    def _start_worker(self, items, batch):
        self._apply_postproc()
        self.worker = DetectWorker(self.detector, items,
                                   score_thresh=self._thresh(), max_dets=self._topk())
        self.worker.sig_item.connect(self._on_item)
        self.worker.sig_slice.connect(self._on_slice)
        self.worker.sig_done.connect(lambda n, t: self._on_done(n, t, batch))
        self.worker.sig_err.connect(self._on_err)
        self._busy(True)
        self.worker.start()

    def stop_detect(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._set_status("⏹ 正在停止（等当前切片跑完）...")

    def _busy(self, busy):
        for b in (self.btn_open, self.btn_dir, self.run_btn, self.batch_btn, self.btn_prev, self.btn_next):
            b.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    def _on_slice(self, pct):
        self.progress.setValue(pct)

    def _on_item(self, idx, name, dets, sec):
        self.results[name] = dets
        self.times[name] = sec
        self._set_status(f"✅ {name} → {len(dets)} 个缺陷，耗时 {sec:.1f}s")
        # 若检测的就是当前显示的图，立即刷新
        if self.files and 0 <= self.cur_idx < len(self.files) and self.files[self.cur_idx].name == name:
            self._render_result()
            self._update_stats(dets)
            self.lbl_time.setText(f"耗时: {sec:.1f}s")
            self.lbl_state.setText(f"状态: {name} 检测完成")
        self.listw.blockSignals(True)
        if 0 <= idx < self.listw.count():
            self.listw.item(idx).setText(f"{name}  ({len(dets)})")
        self.listw.blockSignals(False)

    def _on_done(self, n, total_sec, batch):
        self._busy(False)
        self.progress.setValue(100)
        tot = sum(len(v) for v in self.results.values())
        if batch:
            self._set_status(f"✅ 批量完成：{n} 张 / 共 {tot} 个缺陷 / 总耗时 {total_sec:.1f}s "
                             f"（均值 {total_sec/max(1,n):.1f}s）")
        else:
            self._set_status(f"✅ 检测完成：{n} 张 / {tot} 个缺陷 / 耗时 {total_sec:.1f}s")

    def _on_err(self, msg):
        self._busy(False)
        self._set_status(f"❌ {msg}")

    # ---------------- 渲染 / 统计 ----------------
    def _render_result(self, _v=None):
        name = self.files[self.cur_idx].name if self.files and self.cur_idx >= 0 else None
        if name is None or name not in self.results or self.cur_bgr is None:
            return
        vis = draw_vis(self.cur_bgr, self.results[name], self.gts.get(name), self.chk_gt.isChecked())
        self.v2.set_image(vis)

    def _update_stats(self, dets):
        n_m = sum(1 for d in dets if d['class_name'] == 'missing')
        n_d = sum(1 for d in dets if d['class_name'] == 'dirty')
        self.lbl_total.setText(f"总缺陷数: {len(dets)}")
        self.lbl_missing.setText(f"missing: {n_m}")
        self.lbl_dirty.setText(f"dirty: {n_d}")

        name = self.files[self.cur_idx].name if self.files and self.cur_idx >= 0 else None
        gts = self.gts.get(name, [])
        self.table.setRowCount(len(dets))
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d['bbox']
            iou = best_gt_iou(d['bbox'], gts, d['class_name'])
            self.table.setItem(i, 0, QTableWidgetItem(d['class_name']))
            self.table.setItem(i, 1, QTableWidgetItem(f"{d['confidence']:.3f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{x2-x1} x {y2-y1}"))
            self.table.setItem(i, 3, QTableWidgetItem("-" if iou is None else f"{iou:.3f}"))

    def _set_status(self, text):
        self.lbl_status.setText(text)

    # ---------------- 保存 / 导出 ----------------
    def save_res(self):
        if not self.results or self.cur_bgr is None:
            QMessageBox.information(self, "提示", "还没有检测结果")
            return
        multi = len(self.results) > 1
        if multi:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            d = OUT_DIR / f'batch_{stamp}'
            d.mkdir(parents=True, exist_ok=True)
            n = 0
            for p in self.files:
                if p.name not in self.results:
                    continue
                img = self.cur_bgr if (self.files[self.cur_idx].name == p.name) else imread_u(p)
                if img is None:
                    continue
                vis = draw_vis(img, self.results[p.name], self.gts.get(p.name), self.chk_gt.isChecked())
                if imwrite_u(d / f"res_{Path(p.name).stem}.jpg", vis):
                    n += 1
            self._set_status(f"✅ 已保存 {n} 张结果图到 {d}")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "保存结果图", "result.jpg", "JPEG (*.jpg)")
            if not path:
                return
            vis = draw_vis(self.cur_bgr, self.results[self.files[self.cur_idx].name],
                           self.gts.get(self.files[self.cur_idx].name), self.chk_gt.isChecked())
            if imwrite_u(path, vis):
                self._set_status(f"✅ 已保存: {path}")

    def export_txt(self):
        if not self.results:
            QMessageBox.information(self, "提示", "还没有检测结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出明细", "results.txt", "Text (*.txt)")
        if not path:
            return
        lines = ["图片\t缺陷数\t耗时s\t明细(类别@置信度[x1,y1,x2,y2]|IoU)"]
        for p in self.files:
            name = p.name
            if name not in self.results:
                continue
            dets = self.results[name]
            gts = self.gts.get(name, [])
            parts = []
            for d in dets:
                iou = best_gt_iou(d['bbox'], gts, d['class_name'])
                parts.append(f"{d['class_name']}@{d['confidence']:.3f}{list(d['bbox'])}"
                             f"|IoU={'-' if iou is None else f'{iou:.3f}'}")
            lines.append(f"{name}\t{len(dets)}\t{self.times.get(name, 0):.1f}\t"
                         f"{'; '.join(parts) if parts else '无'}")
        Path(path).write_text("\n".join(lines), encoding='utf-8')
        self._set_status(f"✅ 明细已导出: {path}")

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None, help='checkpoint 路径（默认 config.default_ckpt）')
    args, _unknown = ap.parse_known_args()

    app = QApplication(sys.argv)
    w = PlatformUI(args.model)
    w.show()
    sys.exit(app.exec_() if hasattr(app, 'exec_') else app.exec())


if __name__ == '__main__':
    main()
