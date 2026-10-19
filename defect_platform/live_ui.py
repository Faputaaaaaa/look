# -*- coding: utf-8 -*-
"""Modern live-monitoring console for the custom two-stage bottle defect model.

Launch from platform_ui's "实时监测" button or run:
    python defect_platform/live_ui.py --source 0
    python defect_platform/live_ui.py --source D:/video.mp4
    python defect_platform/live_ui.py --source rtsp://...
The video preview is asynchronous; inference FPS is reported separately.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# [Windows c10.dll] Torch via detector MUST load before Qt.
from defect_platform.detector import FasterRCNNDetector
from defect_platform.live_core import (
    FrameJob, LatestFrameMailbox, MotionBoxes, bounded_roi, global_detections,
)

try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QTimer
    from PyQt5.QtGui import QImage, QPainter, QColor, QPen, QFont, QPixmap
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QLineEdit, QFileDialog, QGroupBox, QGridLayout, QSlider,
        QComboBox, QFrame, QMessageBox, QSizePolicy,
    )
except ImportError:
    from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal, QRectF, QPointF, QTimer
    from PySide6.QtGui import QImage, QPainter, QColor, QPen, QFont, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QLineEdit, QFileDialog, QGroupBox, QGridLayout, QSlider,
        QComboBox, QFrame, QMessageBox, QSizePolicy,
    )


class CaptureWorker(QThread):
    info = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self._running = True
        self.latest = LatestFrameMailbox()

    def stop(self):
        self._running = False
        self.latest.close()

    def run(self):
        cap = None
        try:
            # Prefer DirectShow on Windows webcams; fallback to backend default.
            if isinstance(self.source, int) and os.name == "nt":
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = None
            if cap is None:
                cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.error.emit(f"视频源无法打开：{self.source}")
                return
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # backend-dependent
                if isinstance(self.source, int):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            except Exception:
                pass
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.info.emit(
                f"视频源已连接 · {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} · 标称{fps:.0f}fps"
                if fps > 0 else "视频源已连接")
            seq = 0
            last_emit = 0.0
            errors = 0
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    errors += 1
                    if errors > 5:
                        self.info.emit("视频结束或视频源断开")
                        break
                    self.msleep(50)
                    continue
                errors = 0
                now = time.monotonic()
                # Continue reading frames to avoid downstream capture buffering.
                if now - last_emit < 1/25:
                    continue
                last_emit = now
                if frame.ndim != 3 or frame.shape[2] != 3:
                    continue
                seq += 1
                self.latest.offer((frame, seq, now))
        except Exception as ex:
            self.error.emit(f"读取视频失败：{type(ex).__name__}: {ex}")
        finally:
            self.latest.close()
            if cap is not None:
                cap.release()


class InferenceWorker(QThread):
    ready = pyqtSignal(str)
    results = pyqtSignal(int, int, float, object, float, int)
    error = pyqtSignal(str)

    def __init__(self, model_path=None, detector=None):
        super().__init__()
        self.model_path = model_path
        self.detector = detector
        self.inbox = LatestFrameMailbox()
        self._running = True

    def offer(self, job):
        self.inbox.offer(job)

    def stop(self):
        self._running = False
        self.inbox.close()

    def run(self):
        try:
            if self.detector is None:
                self.detector = FasterRCNNDetector(self.model_path)
            self.ready.emit(f"模型已加载 · {self.detector.device}")
        except Exception as ex:
            self.error.emit(f"模型加载失败：{type(ex).__name__}: {ex}")
            return
        while self._running:
            job = self.inbox.take(timeout=0.2)
            if job is None or not self._running:
                continue
            try:
                x1,y1,x2,y2 = bounded_roi(job.image.shape, job.roi)
                crop = job.image[y1:y2, x1:x2]
                start = time.monotonic()
                dets = self.detector.detect_bgr(
                    crop, score_thresh=job.threshold, max_dets=job.topk)
                seconds = time.monotonic() - start
                dets = global_detections(dets, (x1,y1,x2,y2), job.image.shape)
                if self._running:
                    self.results.emit(job.session, job.sequence, job.timestamp,
                                      dets, seconds, len(dets))
            except Exception as ex:
                self.error.emit(f"视频推理失败：{type(ex).__name__}: {ex}")


def _pos(event):
    return event.position() if hasattr(event, "position") else event.pos()


class VideoCanvas(QWidget):
    """Keep-aspect video, accurate coordinate mapping, thin defect rectangles."""
    roi_changed = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(520,360)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self._image = None
        self._size = None
        self._boxes = []
        self._roi = None
        self._drag = None
        self._start = None

    def set_frame(self, bgr):
        h,w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._image = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888).copy()
        self._size = (w,h)
        self.update()

    def set_boxes(self, boxes):
        self._boxes = boxes
        self.update()

    def set_roi(self, roi):
        self._roi = roi
        self.update()

    def _fit(self):
        if not self._size:
            return QRectF()
        w,h = self._size
        scale = min(self.width()/w, self.height()/h)
        rw,rh = w*scale,h*scale
        return QRectF((self.width()-rw)/2, (self.height()-rh)/2, rw, rh)

    def _to_frame(self, point):
        r = self._fit()
        if r.isEmpty():
            return None
        w,h = self._size
        x = min(w, max(0, (point.x()-r.left())/r.width()*w))
        y = min(h, max(0, (point.y()-r.top())/r.height()*h))
        return (int(round(x)), int(round(y)))

    def _on_screen(self, bbox):
        r = self._fit()
        w,h = self._size
        x1,y1,x2,y2 = bbox
        return QRectF(r.left()+x1*r.width()/w,
                      r.top()+y1*r.height()/h,
                      (x2-x1)*r.width()/w,
                      (y2-y1)*r.height()/h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#111C2C"))
        if self._image is None:
            painter.setPen(QColor("#8DA1B8"))
            painter.setFont(QFont("Microsoft YaHei UI", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待摄像头或视频连接\n支持本地摄像头 / 视频文件 / RTSP")
            return
        rect = self._fit()
        painter.drawImage(rect, self._image)
        painter.setFont(QFont("Microsoft YaHei UI", 9))
        for box in self._boxes:
            name = box.get("class_name", "defect")
            color = QColor("#FB7185" if name == "dirty" else "#28D0A7")
            painter.setPen(QPen(color, 1.5))
            painter.setBrush(Qt.NoBrush)
            target = self._on_screen(box["bbox"]).intersected(rect)
            if target.width()<1 or target.height()<1:
                continue
            painter.drawRect(target)
            text = f"{name}  {box.get('confidence',0):.2f}"
            top = max(rect.top()+2, target.top()-23)
            label_rect = QRectF(target.left(), top, max(96,len(text)*8), 21)
            painter.fillRect(label_rect, color)
            painter.setPen(QColor("#102131"))
            painter.drawText(label_rect.adjusted(5,0,-3,0), Qt.AlignVCenter, text)
        if self._roi is not None:
            pen=QPen(QColor("#FBBF24"), 1.5, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._on_screen(self._roi))
        if self._drag is not None:
            painter.setPen(QPen(QColor("#60A5FA"), 2, Qt.DashLine))
            painter.setBrush(QColor(67,152,245,32))
            painter.drawRect(self._on_screen(self._drag))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._size:
            self._start = self._to_frame(_pos(event))
            self._drag = None

    def mouseMoveEvent(self, event):
        if self._start is not None:
            other = self._to_frame(_pos(event))
            x1,y1 = self._start
            x2,y2 = other
            self._drag = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
            self.update()

    def mouseReleaseEvent(self, event):
        if self._start is not None:
            other = self._to_frame(_pos(event))
            x1,y1 = self._start
            x2,y2 = other
            roi = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
            self._start = None
            self._drag = None
            if roi[2]-roi[0]>=64 and roi[3]-roi[1]>=64:
                self._roi = roi
                self.roi_changed.emit(roi)
            self.update()


class LiveWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, model_path=None, detector=None, default_source="0"):
        super().__init__()
        self.setWindowTitle("PrintSight · 实时丝印质检")
        self.resize(1440, 900)
        self.setMinimumSize(1020,670)
        self._capture = None
        self._capture_id = 0
        self._epoch = 0
        self._frame = None
        self._tracker = MotionBoxes(max_age=2.5)
        self._preview_count = 0
        self._preview_t0 = time.monotonic()
        self._last_offer = -1.0
        self._source_live = False
        self._closing = False
        self._infer = InferenceWorker(model_path=model_path,detector=detector)
        self._build_ui(default_source)
        self._infer.ready.connect(self._on_model_ready)
        self._infer.results.connect(self._on_result)
        self._infer.error.connect(self._on_infer_error)
        self._infer.finished.connect(self._on_thread_finished)
        self._infer.start()
        # GUI owns the display clock: no queued 25fps Qt signals to accumulate.
        self._display_timer=QTimer(self)
        self._display_timer.timeout.connect(self._poll_capture)
        self._display_timer.start(40)

    def _label(self, text, object_name=None):
        x=QLabel(text)
        if object_name: x.setObjectName(object_name)
        return x

    def _card(self, title):
        c=QGroupBox(title)
        c.setObjectName("card")
        return c

    def _build_ui(self, source):
        central=QWidget();self.setCentralWidget(central)
        body=QVBoxLayout(central)
        body.setContentsMargins(24,20,24,18)
        body.setSpacing(16)
        header=QHBoxLayout()
        brand=QVBoxLayout()
        title=self._label("PrintSight  /  丝印智能质检","brand")
        subtitle=self._label("实时视觉监测     ·     自研 ResNet18 + FPN + RPN/ROI","subtle")
        brand.addWidget(title);brand.addWidget(subtitle)
        header.addLayout(brand,1)
        self.model_status=self._label("模型加载中","state")
        header.addWidget(self.model_status)
        body.addLayout(header)

        source_row=QHBoxLayout()
        source_row.setSpacing(10)
        source_row.addWidget(self._label("视频源"))
        self.source=QLineEdit(source)
        self.source.setPlaceholderText("摄像头：0 / 1  ·  视频文件路径  ·  rtsp://地址")
        source_row.addWidget(self.source,1)
        self.pick=QPushButton("选择视频");self.pick.clicked.connect(self._choose_file)
        self.start=QPushButton("开始监测");self.start.setObjectName("primary")
        self.start.clicked.connect(self._start_capture)
        self.stop=QPushButton("停止");self.stop.setEnabled(False);self.stop.clicked.connect(self._stop_capture)
        for button in (self.pick,self.start,self.stop):source_row.addWidget(button)
        body.addLayout(source_row)

        middle=QHBoxLayout();middle.setSpacing(18)
        scene=QVBoxLayout();scene.setSpacing(8)
        self.canvas=VideoCanvas();self.canvas.roi_changed.connect(self._change_roi)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        scene.addWidget(self.canvas,1)
        bottom=QHBoxLayout()
        self.state=self._label("等待视频源连接","subtle")
        bottom.addWidget(self.state,1)
        self.preview_fps=self._label("预览：— fps","metric")
        bottom.addWidget(self.preview_fps)
        scene.addLayout(bottom)
        middle.addLayout(scene,1)

        panel=QVBoxLayout();panel.setSpacing(12)
        status=self._card("检测概览")
        grid=QGridLayout(status)
        grid.setContentsMargins(18,18,18,16)
        self.defect_num=self._label("0","bigCount")
        self.det_latency=self._label("—","metric")
        self.infer_speed=self._label("—","metric")
        grid.addWidget(self.defect_num,0,0)
        grid.addWidget(self._label("当前可信检测框","subtle"),1,0)
        grid.addWidget(self.det_latency,0,1)
        grid.addWidget(self._label("每次推理耗时","subtle"),1,1)
        grid.addWidget(self.infer_speed,2,0,1,2)
        panel.addWidget(status)

        opts=self._card("识别参数")
        ol=QVBoxLayout(opts);ol.setContentsMargins(18,16,18,16);ol.setSpacing(10)
        score_row=QHBoxLayout()
        score_row.addWidget(self._label("置信度阈值"))
        self.score_label=self._label("0.50","metric")
        score_row.addWidget(self.score_label)
        ol.addLayout(score_row)
        self.score=QSlider(Qt.Horizontal)
        self.score.setMinimum(5);self.score.setMaximum(95);self.score.setValue(50)
        self.score.valueChanged.connect(lambda v:self.score_label.setText(f"{v/100:.2f}"))
        ol.addWidget(self.score)
        select=QHBoxLayout()
        select.addWidget(self._label("最多显示"))
        self.topk=QComboBox(); self.topk.addItems(["3","5","10","20"])
        self.topk.setCurrentText("10")
        select.addWidget(self.topk,1)
        ol.addLayout(select)
        self.clear_roi=QPushButton("清除检测区域 · 全画面")
        self.clear_roi.clicked.connect(lambda:self._change_roi(None))
        ol.addWidget(self.clear_roi)
        ol.addWidget(self._label("在视频上拖拽矩形，可只检测瓶身丝印区域。\n不缩小像素尺寸；可显著减少需要计算的切片。","helper"))
        panel.addWidget(opts)

        legend=self._card("结果图例")
        lg=QVBoxLayout(legend);lg.setContentsMargins(18,14,18,14)
        lg.addWidget(self._label("●  missing  /  漏印","green"))
        lg.addWidget(self._label("●  dirty      /  脏污","red"))
        lg.addWidget(self._label("黄色虚线：当前 ROI 范围","helper"))
        lg.addWidget(self._label("细框为最近检测 + 短时光流跟踪，\n跟踪失败或结果过期会自动隐藏。","helper"))
        panel.addWidget(legend)
        self.detail=self._label("尚无检测结果","helper")
        self.detail.setWordWrap(True)
        panel.addWidget(self.detail)
        panel.addStretch()
        panel_wrap=QWidget()
        panel_wrap.setFixedWidth(328)
        panel_wrap.setLayout(panel)
        middle.addWidget(panel_wrap)
        body.addLayout(middle,1)

        self.notice=self._label("注意：视频预览 FPS ≠ AI 检测 FPS。全图慢速模型无法保证产线级实时检出。","footer")
        body.addWidget(self.notice)
        self.setStyleSheet("""
        QMainWindow,QWidget { background:#F3F6FB; color:#22334B; font-family:"Microsoft YaHei UI"; font-size:13px; }
        QLabel { background:transparent; }
        QLabel#brand { font-size:23px; font-weight:800; color:#152B4A; }
        QLabel#subtle,QLabel#helper { color:#718099; font-size:12px; }
        QLabel#helper { line-height:1.5; }
        QLabel#state { background:#E8F2FE; border-radius:12px; color:#1F65AA; padding:9px 14px; font-weight:bold; }
        QLabel#bigCount { font-size:40px; font-weight:800; color:#182A43; }
        QLabel#metric { font-size:14px; font-weight:700; color:#215F92; }
        QLabel#green { color:#178C77; font-weight:700; }
        QLabel#red { color:#CB465D; font-weight:700; }
        QLabel#footer { color:#718099; font-size:12px; }
        QGroupBox#card { background:#FFFFFF; border:1px solid #DDE5F0; border-radius:13px; margin-top:10px;
                         font-weight:700; color:#304562; padding-top:8px; }
        QGroupBox#card::title { subcontrol-origin:margin; left:14px; padding:0 6px; background:#FFFFFF; }
        QPushButton { background:#FFFFFF; border:1px solid #D4DFEB; color:#344B69;
                     padding:10px 17px; border-radius:9px; font-weight:700; }
        QPushButton:hover { background:#E8F4FC; border-color:#7CAFD4; }
        QPushButton:disabled { background:#E9EEF4; color:#9BAAC0; }
        QPushButton#primary { background:#237FC1; color:white; border:1px solid #237FC1; }
        QPushButton#primary:hover { background:#1467A7; }
        QLineEdit,QComboBox { background:white; border:1px solid #D4DFEB; padding:10px; border-radius:9px; }
        QSlider::groove:horizontal { height:6px; background:#DAE6F1; border-radius:3px; }
        QSlider::handle:horizontal { background:#2788C9; width:17px; margin:-5px 0; border-radius:8px; }
        """)

    def _on_model_ready(self, desc):
        self.model_status.setText(desc)
        self.state.setText("模型就绪，选择摄像头或视频并开始")
    def _on_infer_error(self, msg):
        self.state.setText(msg)
        self.model_status.setText("模型或推理异常")

    def _choose_file(self):
        path,_=QFileDialog.getOpenFileName(self,"选择视频","",
                            "视频文件 (*.mp4 *.avi *.mkv *.mov);;全部文件 (*)")
        if path:self.source.setText(path)

    def _start_capture(self):
        if self._capture is not None and self._capture.isRunning():
            self.state.setText("当前视频源尚未停止"); return
        entry=self.source.text().strip()
        if not entry:
            self.state.setText("请输入摄像头编号、视频文件或 RTSP 地址");return
        src=int(entry) if entry.isdigit() else entry
        if isinstance(src,str) and "://" not in src and not Path(src).is_file():
            self.state.setText("找不到视频文件："+src);return
        self._epoch+=1
        self._capture_id+=1
        cid=self._capture_id
        self._tracker.clear()
        self.canvas.set_boxes([])
        self.canvas.set_roi(None)
        self._frame=None
        self._source_live=True
        self._last_offer=-1.
        cap=CaptureWorker(src)
        self._capture=cap
        cap.info.connect(self.state.setText)
        cap.error.connect(self.state.setText)
        cap.finished.connect(lambda cid=cid:self._capture_finished(cid))
        cap.start()
        self.stop.setEnabled(True)
        self.start.setEnabled(False)

    def _stop_capture(self):
        self._source_live=False
        self._epoch+=1
        self._tracker.clear()
        self.canvas.set_boxes([])
        self.defect_num.setText("0")
        if self._capture is not None and self._capture.isRunning():
            self._capture.stop()
            self.state.setText("正在停止视频源…")
        else:
            self.start.setEnabled(True)
        self.stop.setEnabled(False)

    def _capture_finished(self,cid):
        if cid != self._capture_id:return
        self._source_live=False
        self._epoch+=1
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        self._capture=None

    def _change_roi(self,roi):
        self.canvas.set_roi(roi)
        self._epoch+=1
        self._tracker.clear()
        self.canvas.set_boxes([])
        self.defect_num.setText("0")
        if roi:
            w,h=roi[2]-roi[0],roi[3]-roi[1]
            self.state.setText(f"ROI 已设置：{w}×{h} px；旧推理结果不再展示")
        else:
            self.state.setText("ROI 已清除：恢复检测完整画面")

    def _poll_capture(self):
        cap=self._capture
        if cap is None or not self._source_live:
            return
        item=cap.latest.take(timeout=0)
        if item is None:
            return
        frame,seq,ts=item
        self._on_frame(self._capture_id,frame,seq,ts)

    def _on_frame(self, cid,frame,seq,ts):
        if cid!=self._capture_id or not self._source_live:return
        self._frame=frame
        self.canvas.set_frame(frame)
        self._tracker.ingest(seq,ts,frame)
        boxes=self._tracker.visible(ts)
        self.canvas.set_boxes(boxes)
        self.defect_num.setText(str(len(boxes)))
        self._preview_count+=1
        if ts-self._preview_t0>=1:
            self.preview_fps.setText(f"预览：{self._preview_count/(ts-self._preview_t0):.1f} fps")
            self._preview_t0=ts;self._preview_count=0
        if ts-self._last_offer>=0.25:
            self._last_offer=ts
            job=FrameJob(self._epoch,seq,ts,frame.copy(),self.canvas._roi,
                         self.score.value()/100,int(self.topk.currentText()))
            self._infer.offer(job)

    def _on_result(self, epoch, seq, stamp, dets,seconds,n):
        if epoch!=self._epoch or not self._source_live:return
        self.det_latency.setText(f"{seconds:.2f} s")
        self.infer_speed.setText(f"AI 推理频率 ≤ {1/max(seconds,0.001):.2f} fps")
        ok=self._tracker.install(seq,stamp,dets)
        if not ok:
            self.canvas.set_boxes([])
            self.defect_num.setText("0")
            self.state.setText(f"推理结果已过期（{seconds:.1f}s）：继续预览，避免把旧框画在新瓶上。请缩小 ROI。")
            return
        visible=self._tracker.visible()
        self.canvas.set_boxes(visible)
        self.defect_num.setText(str(len(visible)))
        self.detail.setText(" · ".join(f"{d['class_name']} {d['confidence']:.2f}" for d in visible) or "本次未发现缺陷")
        self.state.setText(f"检测更新：{n} 个框 · 用光流跟随当前画面；仅为短时跟踪")

    def _on_thread_finished(self):
        if self._closing: QTimer.singleShot(50,self.close)

    def closeEvent(self,event):
        if not self._closing:
            self._closing=True
            self._stop_capture()
            self._infer.stop()
        capture_busy=self._capture is not None and self._capture.isRunning()
        infer_busy=self._infer.isRunning()
        if capture_busy or infer_busy:
            event.ignore()
            QTimer.singleShot(250,self.close)
        else:
            self.closed.emit()
            event.accept()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="0",help="camera index, video file, or RTSP URL")
    ap.add_argument("--model",default=None)
    args=ap.parse_args()
    app=QApplication(sys.argv)
    window=LiveWindow(model_path=args.model,default_source=args.source)
    window.show()
    sys.exit(app.exec_() if hasattr(app,"exec_") else app.exec())


if __name__=="__main__":
    main()
