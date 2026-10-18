"""[R35] SAHI切片推理完整流程

流程：
1. 加载训练最优权重
2. 原图4096×2736滑动窗口切片(640×640, stride=320)
3. 单切片送模型推理(预处理与训练一致)
4. 切片内预测框坐标反向映射回原图坐标系
5. 全局NMS合并所有切片的预测框
6. 计算mAP、召回率、精确率
7. 可视化输出
"""
import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.ops import nms

from config import config
from model_fpn import TwoStageDetector, FPN_CONFIG, INFERENCE_CONFIG
from dataset import DefectDataset

# ============ 配置 ============
# [R40] 历史默认（保留备注，勿删）：experiments\exp_20260812_093621\models\best_model.pth
# [平台 2026-09-29] 默认 checkpoint 改为从 config.default_ckpt 派生，并转成**绝对路径**。
# 原因：旧默认是相对路径常量，`python make_report.py` 单独跑时会静默加载旧实验权重，
#       报告里的 mAP 与 STATUS.md（overnight_20260927, 0.7788）口径不一致。
# 优先级：load_model 显式入参 > 环境变量 SAHI_MODEL_PATH > 这里的默认值。
def _init_model_path():
    rel = getattr(config, 'default_ckpt', '') or r"experiments\exp_20260812_093621\models\best_model.pth"
    p = Path(rel)
    return str(p if p.is_absolute() else Path(__file__).resolve().parent / p)


MODEL_PATH = _init_model_path()
DATA_DIR = Path(r"d:\deeplearning\learn\xinpaide")
SPLIT_PATH = Path(__file__).resolve().parent / "split.json"  # 按原图划分的结果（P0.1）
SUBSETS = ['dirty', 'missing']
# [A9] 从 config 派生目标类 GT id → name 映射（单一事实来源）
YOLO_ID_TO_NAME = {yid: name for yid, name in config.yolo_id_to_name.items() if name in config.class_names}
# 模型输出: 0=BG, 1=missing, 2=dirty
MODEL_CLS_TO_NAME = {1: 'missing', 2: 'dirty'}

WINDOW = config.window_size   # 640
STRIDE = config.window_stride # 320
IMG_SIZE = config.img_size    # 640
SCORE_THRESH = INFERENCE_CONFIG['score_thresh']  # 0.10
NMS_THRESH = INFERENCE_CONFIG['nms_thresh']     # 0.30
MAX_DET = INFERENCE_CONFIG['max_detections']    # 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet标准化（与训练一致）
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def load_model(model_path=None):
    """加载训练好的模型。

    model_path 未给时，依次用环境变量 SAHI_MODEL_PATH、MODEL_PATH。
    """
    path = model_path or os.environ.get('SAHI_MODEL_PATH') or MODEL_PATH
    model = TwoStageDetector(num_classes=len(config.class_names))
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    # [R40] strict=False：旧权重可能没有 residual_proj（新加的残差投影层）
    # 该层零初始化时 residual=0，输出与旧模型完全一致，可安全加载。
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # 过滤掉roi_head.residual_proj（预期缺失，零初始化）
        real_missing = [k for k in missing if 'residual_proj' not in k]
        if real_missing:
            print(f"[警告] 缺失非residual_proj的key: {real_missing}")
        else:
            print(f"[info] residual_proj 用零初始化（旧权重无此参数），输出与旧模型一致")
    if unexpected:
        print(f"[警告] 意外key: {unexpected}")
    model.to(DEVICE)
    model.eval()
    print(f"模型加载: {path}")
    return model


def preprocess_crop(crop_img_pil):
    """单切片预处理（与训练一致）
    输入: PIL Image (640×640)
    输出: tensor [1,3,640,640]
    """
    arr = np.array(crop_img_pil, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = NORMALIZE(t)
    return t.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def infer_single_crop(model, crop_tensor, score_thresh=None):
    """单切片推理，返回切片内预测框列表

    Args:
        score_thresh: 可选，覆盖全局 SCORE_THRESH（平台 UI 调阈值时用）
    Returns: list of (x1,y1,x2,y2,score,cls_id) 切片内坐标
    """
    thr = float(SCORE_THRESH if score_thresh is None else score_thresh)
    out = model.inference(crop_tensor)
    boxes_list = out.get('boxes', [])
    scores_list = out.get('scores', [])
    labels_list = out.get('labels', [])

    results = []
    if isinstance(boxes_list, list) and len(boxes_list) > 0:
        b_boxes = boxes_list[0]
        b_scores = scores_list[0] if isinstance(scores_list, list) else scores_list
        b_labels = labels_list[0] if isinstance(labels_list, list) else labels_list
    elif isinstance(boxes_list, torch.Tensor):
        b_boxes = boxes_list[0] if boxes_list.dim() == 3 else boxes_list
        b_scores = scores_list[0] if isinstance(scores_list, torch.Tensor) and scores_list.dim() == 2 else scores_list
        b_labels = labels_list[0] if isinstance(labels_list, torch.Tensor) and labels_list.dim() == 2 else labels_list
    else:
        b_boxes = torch.tensor([])
        b_scores = torch.tensor([])
        b_labels = torch.tensor([])

    if isinstance(b_boxes, torch.Tensor) and b_boxes.numel() > 0:
        for i in range(b_boxes.size(0)):
            s = float(b_scores[i].item()) if isinstance(b_scores, torch.Tensor) else float(b_scores)
            if s < thr:
                continue
            c = int(b_labels[i].item()) if isinstance(b_labels, torch.Tensor) else int(b_labels)
            if c == 0:
                continue
            box = b_boxes[i].cpu().numpy()
            results.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), s, c))
    return results


def sahi_inference_image(model, img, score_thresh=None, max_dets=None, progress=None):
    """[平台 2026-09-29] 对内存中的 PIL Image 执行完整 SAHI 推理。

    与 sahi_inference() 共用同一条代码路径（评估 / 平台同口径），
    只把输入从「文件路径」换成「内存图像」，省掉平台侧写临时 jpg 的
    一次磁盘 IO 与 quality=95 的压缩损失。

    Args:
        score_thresh: 可选，覆盖全局 SCORE_THRESH
        max_dets:     可选，覆盖 config.max_dets_per_image（top-k 保留）
        progress:     可选回调 progress(done, total)，每 8 个切片回调一次
    Returns: list of (x1,y1,x2,y2,score,cls_id) 原图坐标系
    """
    iw, ih = img.size
    thr = float(SCORE_THRESH if score_thresh is None else score_thresh)
    keep_k = max(1, int(config.max_dets_per_image if max_dets is None else max_dets))

    # [平台 2026-09-29] 先枚举窗口，便于回调切片级进度（UI 进度条）。
    windows = []
    for y in range(0, ih, STRIDE):
        for x in range(0, iw, STRIDE):
            x2 = min(x + WINDOW, iw)
            y2 = min(y + WINDOW, ih)
            x1 = max(0, x2 - WINDOW)
            y1 = max(0, y2 - WINDOW)
            if x2 - x1 < 64 or y2 - y1 < 64:
                continue
            windows.append((x1, y1, x2, y2))

    all_dets = []
    total = max(1, len(windows))
    for _n, (x1, y1, x2, y2) in enumerate(windows, 1):
        if progress is not None and (_n % 8 == 0 or _n == total):
            progress(_n, total)

        crop = img.crop((x1, y1, x2, y2))
        if crop.size != (IMG_SIZE, IMG_SIZE):
            padded = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            padded.paste(crop, (0, 0))
            crop = padded

        crop_tensor = preprocess_crop(crop)
        crop_dets = infer_single_crop(model, crop_tensor, thr)

        for dx1, dy1, dx2, dy2, s, c in crop_dets:
            edges_touched = 0
            if dx1 < 5: edges_touched += 1
            if dy1 < 5: edges_touched += 1
            if dx2 > IMG_SIZE - 5: edges_touched += 1
            if dy2 > IMG_SIZE - 5: edges_touched += 1
            if edges_touched >= 3:
                continue

            ox1 = dx1 + x1
            oy1 = dy1 + y1
            ox2 = dx2 + x1
            oy2 = dy2 + y1
            all_dets.append((ox1, oy1, ox2, oy2, s, c))

    if not all_dets:
        return []

    boxes = torch.tensor([(d[0], d[1], d[2], d[3]) for d in all_dets], dtype=torch.float32)
    scores = torch.tensor([d[4] for d in all_dets], dtype=torch.float32)
    classes = torch.tensor([d[5] for d in all_dets], dtype=torch.long)

    final = []
    for cls in classes.unique().tolist():
        mask = classes == cls
        if mask.sum() == 0:
            continue
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        keep = nms(cls_boxes, cls_scores, NMS_THRESH)
        for i in keep:
            final.append((
                cls_boxes[i, 0].item(), cls_boxes[i, 1].item(),
                cls_boxes[i, 2].item(), cls_boxes[i, 3].item(),
                cls_scores[i].item(), cls
            ))
        if len(final) >= MAX_DET * 2:
            break

    # [R38] top-k：数据集每张原图只有 1 个缺陷 → 默认 max_dets_per_image=1（= 评估口径）。
    # [平台] 产线可能一图多缺陷，keep_k 可由 config.max_dets_per_image / 平台 UI 调大。
    final.sort(key=lambda x: -x[4])
    final = final[:keep_k]

    # [A14] dirty 类 dy 偏移补偿：预测框整体 y 平移（默认关闭）。
    # overnight_20260927 后 dirty 框系统性向下偏移（train/val 均 100% 为正），
    # 宽高正常，纯平移偏差，故整体上移 dy_comp_dirty 像素。
    # 越界 clip 由下方扩张段的边界裁剪统一处理。
    if config.use_dy_compensation and config.dy_comp_dirty != 0.0:
        shifted = []
        for x1, y1, x2, y2, s, c in final:
            if c == 2:  # dirty
                y1 += config.dy_comp_dirty
                y2 += config.dy_comp_dirty
            shifted.append((x1, y1, x2, y2, s, c))
        final = shifted

    # [R40] missing类检测框1.3x扩张 + [R41] dirty类1.1x扩张
    # R40: missing框系统性偏小50%，1.3x最佳(mAP 75%→86.9%)
    # R41: dirty类也有轻微偏小，1.1x适度补偿
    # [A11] 扩张后处理：系数从 config 读，开关 config.use_bbox_expansion（默认关闭）。
    # 历史 R40/R41 用 1.3x/1.1x 补偿框偏小，但 B1 诊断发现 R41 已不再偏小 50%，
    # 扩张过度补偿，故默认关闭；重新训练后按新数据统计再决定是否启用。代码路径保留。
    expanded = []
    for x1, y1, x2, y2, s, c in final:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if config.use_bbox_expansion:
            if c == 1:
                w *= config.expand_missing
                h *= config.expand_missing
            elif c == 2:
                w *= config.expand_dirty
                h *= config.expand_dirty
        nx1, ny1 = cx - w/2, cy - h/2
        nx2, ny2 = cx + w/2, cy + h/2
        # [A10] 扩张后 clip 回原图边界（贴边缺陷扩张可能越界）
        nx1 = max(0.0, nx1)
        ny1 = max(0.0, ny1)
        nx2 = min(float(iw), nx2)
        ny2 = min(float(ih), ny2)
        expanded.append((nx1, ny1, nx2, ny2, s, c))
    return expanded


def sahi_inference(model, img_path, progress=None):
    """对一张原图（文件路径）执行完整SAHI推理
    Returns: list of (x1,y1,x2,y2,score,cls_id) 原图坐标系
    """
    img = Image.open(img_path).convert('RGB')
    return sahi_inference_image(model, img, progress=progress)


def load_gt(img_path, label_path):
    """加载GT标签，返回原图坐标系bbox列表"""
    iw, ih = Image.open(img_path).size
    gts = []
    if not label_path.exists():
        return gts
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                yolo_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                name = YOLO_ID_TO_NAME.get(yolo_id)
                if name is None:
                    continue
                x1 = (cx - w/2) * iw
                y1 = (cy - h/2) * ih
                x2 = (cx + w/2) * iw
                y2 = (cy + h/2) * ih
                gts.append((x1, y1, x2, y2, name))
    return gts


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / max(union, 1e-6)


def voc_ap(rec, prec):
    """VOC 标准 AP（all-point interpolation）。

    锚定 recall=0（precision 补 1）、recall=1（precision 补 0），
    precision 做单调不减插值，再按 recall 分段积分。

    注意：改用此标准实现后，AP/mAP 数字与旧版（np.trapezoid 未锚定 recall=0）
    不可直接比较，旧版会系统性偏低。
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    # 单调不减插值：从右往左取 max
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def load_split():
    """读取 split.json（按原图划分，seed=42）。"""
    with open(SPLIT_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_image(img_id):
    """把 split.json 的原图 ID（如 'dirty/frame_0001.jpg'）解析为 (img_path, lbl_path, subset)。"""
    subset, fname = img_id.split('/', 1)
    img_file = DATA_DIR / subset / "images" / fname
    lbl_file = DATA_DIR / subset / "labels" / f"{Path(fname).stem}.txt"
    return img_file, lbl_file, subset


def evaluate_map(model, split='val', max_imgs=None):
    """评估 mAP@0.5（SAHI 合并后按原图算）。

    只读 split.json 的 val 原图（禁止用训练图）。流程：
    1. 每张原图 SAHI 切片推理 → 框映射回原图坐标系 → 全局 NMS 合并。
    2. 按原图算每类 TP/FP/FN，再算 AP / precision / recall。

    Args:
        model: 已加载的模型。
        split: 'val'（默认）或 'train'（仅过拟合诊断，读 split.json 的 train）。
        max_imgs: 可选，只评估前 N 张（冒烟测试用）；None=全量。
    """
    print(f"\n{'='*60}")
    print(f"SAHI切片推理 mAP评估  [split={split}, 只读 split.json]")
    print(f"{'='*60}")

    sp = load_split()
    val_ids = sp[split]
    if max_imgs is not None:
        val_ids = val_ids[:max_imgs]
    test_imgs = [resolve_image(img_id) for img_id in val_ids]
    print(f"评估原图数: {len(test_imgs)}  (split={split}, 来自 split.json)")

    stats = {name: {'tp': 0, 'fp': 0, 'fn': 0, 'scores': [], 'tps': []} for name in MODEL_CLS_TO_NAME.values()}

    for i, (img_file, lbl_file, subset) in enumerate(test_imgs):
        if i % 10 == 0:
            print(f"  进度: {i}/{len(test_imgs)}")
        dets = sahi_inference(model, str(img_file))
        gts = load_gt(img_file, lbl_file)

        for cls_name in MODEL_CLS_TO_NAME.values():
            cls_dets = [d for d in dets if MODEL_CLS_TO_NAME.get(d[5]) == cls_name]
            cls_gts = [g for g in gts if g[4] == cls_name]

            cls_dets.sort(key=lambda x: -x[4])

            matched_gt = set()
            for det in cls_dets:
                best_iou = 0
                best_gt = -1
                for j, gt in enumerate(cls_gts):
                    if j in matched_gt:
                        continue
                    iou = compute_iou(det[:4], gt[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j

                if best_iou >= 0.5 and best_gt >= 0:
                    matched_gt.add(best_gt)
                    stats[cls_name]['tp'] += 1
                    stats[cls_name]['scores'].append(det[4])
                    stats[cls_name]['tps'].append(1)
                else:
                    stats[cls_name]['fp'] += 1
                    stats[cls_name]['scores'].append(det[4])
                    stats[cls_name]['tps'].append(0)

            stats[cls_name]['fn'] += len(cls_gts) - len(matched_gt)

    print(f"\n{'='*60}")
    print("mAP结果")
    print(f"{'='*60}")
    aps = []
    for cls_name in sorted(MODEL_CLS_TO_NAME.values()):
        s = stats[cls_name]
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        n_pos = tp + fn  # 该类 GT 总数
        if s['scores'] and n_pos > 0:
            scores = np.array(s['scores'])
            tps = np.array(s['tps'])
            order = np.argsort(-scores)
            tps = tps[order]
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(1 - tps)
            rec = cum_tp / n_pos
            prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)
            ap = voc_ap(rec, prec)
        else:
            ap = 0.0
        aps.append(ap)
        print(f"  {cls_name}: TP={tp}, FP={fp}, FN={fn}, P={precision:.3f}, R={recall:.3f}, AP={ap:.3f}")

    mAP = np.mean(aps) if aps else 0
    print(f"\n  mAP@0.5 = {mAP:.4f} ({mAP*100:.1f}%)")
    print(f"{'='*60}")
    return mAP


def visualize_one(model, img_path, out_path, gt_path=None):
    """可视化单张图：将预测框（红）和GT原缺陷位置框（绿）画在同一张图上。

    [R40] 之前只画预测框，难以直观判断 IoU / 漏检 / 误检。
    现在GT和预测同图对比：
      - 红色实线框  = 预测框（含 score）
      - 绿色虚线框  = GT 真实缺陷位置
      - 左上角图例  = 颜色说明

    Args:
        model: 已加载的模型
        img_path: 原图路径
        out_path: 输出图片路径
        gt_path: GT标签文件路径（YOLO格式）。None时自动从img_path推断：
                 .../{subset}/images/frame_xxxx.jpg → .../{subset}/labels/frame_xxxx.txt
    """
    dets = sahi_inference(model, str(img_path))
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    # 字体
    try:
        font = ImageFont.truetype("arial.ttf", 40)
        font_small = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
        font_small = font

    # ===== 1. 画GT框（绿色虚线） =====
    # YOLO标签: class_id cx cy w h（归一化到原图）
    # 数据集 classes.txt: 3=missing, 4=dirty
    if gt_path is None:
        # 自动推断: images/xxx.jpg → labels/xxx.txt
        img_path = Path(img_path)
        subset = img_path.parent.parent.name  # 'dirty' or 'missing'
        label_dir = DATA_DIR / subset / "labels"
        gt_path = label_dir / f"{img_path.stem}.txt"

    gt_boxes = []
    if Path(gt_path).exists():
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                yolo_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                # 归一化 → 原图像素
                px_cx = cx * img_w
                px_cy = cy * img_h
                px_w = bw * img_w
                px_h = bh * img_h
                x1 = px_cx - px_w / 2
                y1 = px_cy - px_h / 2
                x2 = px_cx + px_w / 2
                y2 = px_cy + px_h / 2
                gt_name = YOLO_ID_TO_NAME.get(yolo_id, f"id{yolo_id}")
                gt_boxes.append((x1, y1, x2, y2, gt_name))

    GT_COLOR = (0, 255, 0)  # 绿色 = GT
    for x1, y1, x2, y2, name in gt_boxes:
        # 虚线效果：通过画多个短矩形模拟（Pillow无内置dashed）
        # 这里用粗实线 + 不同颜色区分，比虚线更清晰
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=8)
        # GT标签放在框左下角，避免和预测标签（左上角）冲突
        draw.text((x1, min(img_h - 50, y2 + 5)), f"GT: {name}",
                  fill=GT_COLOR, font=font_small)

    # ===== 2. 画预测框（红色实线） =====
    PRED_COLOR = (255, 0, 0)  # 红色 = 预测
    for det in dets:
        x1, y1, x2, y2, s, c = det
        name = MODEL_CLS_TO_NAME.get(c, str(c))
        draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=5)
        # 预测标签放在框左上角
        draw.text((x1, max(0, y1 - 45)), f"{name} {s:.2f}",
                  fill=PRED_COLOR, font=font)

    # ===== 3. 左上角图例 =====
    legend_x, legend_y = 20, 20
    legend_w, legend_h = 50, 20
    # 红色方块 + "Prediction"
    draw.rectangle([legend_x, legend_y, legend_x + legend_w, legend_y + legend_h],
                   outline=PRED_COLOR, width=4)
    draw.text((legend_x + legend_w + 10, legend_y - 5), "Prediction",
              fill=PRED_COLOR, font=font_small)
    # 绿色方块 + "Ground Truth"
    draw.rectangle([legend_x, legend_y + 50, legend_x + legend_w, legend_y + 50 + legend_h],
                   outline=GT_COLOR, width=4)
    draw.text((legend_x + legend_w + 10, legend_y + 45), "Ground Truth",
              fill=GT_COLOR, font=font_small)

    # ===== 4. 标题信息 =====
    n_pred = len(dets)
    n_gt = len(gt_boxes)
    info = f"P:{n_pred}  GT:{n_gt}"
    draw.text((img_w - 350, 20), info, fill=(255, 255, 255), font=font)

    img.save(out_path)
    print(f"可视化保存: {out_path}  (预测={n_pred}, GT={n_gt})")


if __name__ == "__main__":
    model = load_model()
    # 评估只读 split.json 的 val 原图（禁止训练图）
    mAP = evaluate_map(model, split='val')
    print("\n生成可视化（验证集）...")
    sp = load_split()
    for subset in SUBSETS:
        # 可视化取 val 原图（来自 split.json），避免用训练图
        val_subset = [x for x in sp['val'] if x.startswith(f"{subset}/")][-2:]
        for img_id in val_subset:
            img_file, _, _ = resolve_image(img_id)
            out = f"r40_vis_{subset}_{Path(img_id).name}"
            visualize_one(model, str(img_file), out)
