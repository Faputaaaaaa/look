# -*- coding: utf-8 -*-
"""[平台融合] Faster R-CNN 检测适配器。

把项目自训练的 Faster R-CNN（overnight_20260927, mAP@0.5=0.7788）包成
平台统一的检测接口：图进 → [(bbox, class_name, confidence)] 出。

复用 sahi_inference.py 的完整 SAHI 管线（640 窗口 stride 320 + 全局 NMS
+ top-k + A11 扩张/A14 dy 补偿开关），保证平台推理结果与评估口径一致。

用法:
    from defect_platform.detector import FasterRCNNDetector
    det = FasterRCNNDetector()                       # 默认 config.default_ckpt
    det = FasterRCNNDetector("path/to/best_model.pth")
    dets = det.detect_path("xxx.jpg")   # 文件路径
    dets = det.detect_bgr(bgr_ndarray)  # OpenCV 读的图（平台主用）
    dets = det.detect_pil(pil_img)      # PIL Image

2026-09-29 修订（平台审查）:
1. 内存直推：新增 detect_bgr / detect_pil 走 sahi_inference_image()，
   不再落临时 jpg（旧实现有 quality=95 压缩损失 + 一次磁盘 IO）。
2. checkpoint 来源统一：命令行/环境变量/config.default_ckpt，去掉硬编码优先级混乱。
3. 阈值与 top-k 可在实例上设置（UI 滑块），默认沿用 sahi 全局（= 评估口径）。
"""
import os
import sys
from pathlib import Path

# defect_platform/ 的上级是项目根，导入 sahi_inference / config
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image

import sahi_inference as si
from config import config


class FasterRCNNDetector:
    """统一检测接口（静态图 MVP）。

    对外只暴露:
        detect_path(img_path)  -> list[dict]
        detect_bgr(bgr_array)  -> list[dict]
        detect_pil(pil_image)  -> list[dict]
    dict 字段: bbox=(x1,y1,x2,y2) 原图像素 / class_name / confidence / class_id
    """

    def __init__(self, model_path=None, score_thresh=None, max_dets=None):
        self.model_path = model_path or default_ckpt()
        self.model = si.load_model(self.model_path)
        self.score_thresh = None if score_thresh is None else float(score_thresh)
        self.max_dets = None if max_dets is None else int(max_dets)
        self.class_names = {1: 'missing', 2: 'dirty'}
        self.device = str(si.DEVICE)

    # ---------- 对外主接口 ----------
    def detect_path(self, img_path, score_thresh=None, max_dets=None, progress=None):
        """对一张原图文件执行完整 SAHI 推理（内部同样走内存图像路径，口径一致）。"""
        return self.detect_pil(Image.open(str(img_path)), score_thresh, max_dets, progress)

    def detect_bgr(self, bgr, score_thresh=None, max_dets=None, progress=None):
        """对 OpenCV 读到的 BGR ndarray 推理（平台主路径，无临时文件）。"""
        if bgr is None:
            raise ValueError("detect_bgr: 图像为空")
        rgb = bgr[..., ::-1]  # BGR -> RGB（避免为这一件事引入 cv2 依赖）
        if not rgb.flags['C_CONTIGUOUS']:
            rgb = np.ascontiguousarray(rgb)
        return self.detect_pil(Image.fromarray(rgb), score_thresh, max_dets, progress)

    def detect_pil(self, pil_img, score_thresh=None, max_dets=None, progress=None):
        """对内存中的 PIL Image 执行推理（直推 sahi_inference_image）。"""
        return self._wrap(
            si.sahi_inference_image(
                self.model,
                pil_img.convert('RGB'),
                score_thresh=self._pick(self.score_thresh, score_thresh),
                max_dets=self._pick(self.max_dets, max_dets),
                progress=progress,
            ),
            score_thresh, max_dets)

    # ---------- 内部 ----------
    @staticmethod
    def _pick(inst_val, arg_val):
        """实例默认 < 调用参数（None 表示沿用 sahi 全局/配置）。"""
        return arg_val if arg_val is not None else inst_val

    def _wrap(self, dets, score_thresh=None, max_dets=None):
        out = []
        for x1, y1, x2, y2, s, c in dets:
            out.append({
                'bbox': (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                'class_name': self.class_names.get(int(c), str(c)),
                'confidence': float(s),
                'class_id': int(c),
            })
        return out

    def postproc_state(self):
        """返回当前后处理开关状态（供界面信息面板展示）。"""
        return {
            'A11 bbox扩张': '开' if config.use_bbox_expansion else '关',
            'A14 dy补偿': '开' if config.use_dy_compensation else '关',
            'score_thresh': self.score_thresh if self.score_thresh is not None else si.SCORE_THRESH,
            'top_k': self.max_dets if self.max_dets is not None else config.max_dets_per_image,
            'window/stride': f'{si.WINDOW}/{si.STRIDE}',
            'device': self.device,
        }


def default_ckpt():
    """平台默认 checkpoint 优先级：
        1) 环境变量 SAHI_MODEL_PATH
        2) config.default_ckpt（相对项目根；相对路径按项目根解析）
        3) experiments/overnight_20260927/models/{best,latest}.pth
        4) None → 交给 sahi_inference.load_model 用它的 MODEL_PATH 兜底
    """
    env = os.environ.get('SAHI_MODEL_PATH')
    if env and Path(env).exists():
        return env

    cands = []
    ck = getattr(config, 'default_ckpt', '')
    if ck:
        p = Path(ck)
        cands.append(p if p.is_absolute() else _ROOT / p)
    cands += [
        _ROOT / 'experiments' / 'overnight_20260927' / 'models' / 'best_model.pth',
        _ROOT / 'experiments' / 'overnight_20260927' / 'models' / 'latest.pth',
    ]
    for c in cands:
        if c.exists():
            return str(c)
    return None


if __name__ == '__main__':
    # 自测：拿 val 里 missing/dirty 各一张跑一遍
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--topk', type=int, default=None)
    a = ap.parse_args()

    det = FasterRCNNDetector(a.model, max_dets=a.topk)
    print('checkpoint:', det.model_path)
    print('postproc:', det.postproc_state())

    sp = si.load_split()
    for cls in ('missing', 'dirty'):
        pick = [i for i in sp['val'] if i.startswith(f'{cls}/')][:1]
        for img_id in pick:
            img_file, lbl_file, _ = si.resolve_image(img_id)
            print(f'{img_id} ->', det.detect_path(img_file))
            print('   GT:', si.load_gt(img_file, lbl_file))
