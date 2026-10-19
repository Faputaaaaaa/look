"""PR3: opt-in multi-instance SAHI-window training from PR2's manifest.

All patches of a physical bottle stay in the train or validation partition
selected by quality.split. No held-out SKU may be sampled for training.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import config
from quality.manifest import load_manifest
from quality.split import check_group_isolation
from inference_utils import unique_sahi_windows

_NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def window_defects(defects, window, *, min_visible_fraction=0.5, min_edge_px=4):
    """Return (clipped normalized targets, ambiguous) for a crop.

    An intersecting but mostly truncated/smaller-than-trainable object makes
    the whole crop ambiguous. Do not incorrectly label it as background.
    """
    x0, y0, x1, y1 = window
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("non-positive window dimensions")
    targets = []
    ambiguous = False
    for defect in defects:
        bx0, by0, bx1, by1 = defect["bbox"]
        ix0, iy0, ix1, iy1 = max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1)
        iw, ih = ix1 - ix0, iy1 - iy0
        if iw <= 0 or ih <= 0:
            continue
        vis = (iw * ih) / ((bx1 - bx0) * (by1 - by0))
        if vis < min_visible_fraction or min(iw, ih) < min_edge_px:
            ambiguous = True
            continue
        targets.append((
            ((ix0 + ix1) / 2 - x0) / width,
            ((iy0 + iy1) / 2 - y0) / height,
            iw / width, ih / height,
            defect["class_id"],
        ))
    return targets, ambiguous


class ManifestWindowDataset(Dataset):
    """Use exactly the same input contract as DefectDataset: img, n, padded boxes, meta.

    Only `train`/`val` are allowed: evaluating held-out SKUs must use PR2's
    separate export/evaluate tools, never the training loader.
    """
    def __init__(self, manifest_path, split_path, data_root, partition, *,
                 max_background_per_image=2, seed=42):
        if partition not in ("train", "val"):
            raise ValueError("only train/val partitions may be loaded for training")
        if max_background_per_image < 0:
            raise ValueError("max_background_per_image cannot be negative")
        self.manifest = load_manifest(manifest_path)
        self.split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        check_group_isolation(self.manifest, self.split)
        # Explicitly reject class-index ambiguity and silent dropping of unseen classes.
        if self.manifest["classes"] != list(config.class_names):
            raise ValueError("manifest.classes must exactly match config.class_names in order")
        self.partition = partition
        self.data_root = Path(data_root).resolve()
        self.window_size = int(config.window_size)
        self.stride = int(config.window_stride)
        self.img_size = int(config.img_size)
        self.max_objects = int(config.max_objects)
        if self.img_size != self.window_size:
            raise ValueError("manifest training currently requires img_size == window_size")
        self.records = []
        allowed = set(self.split[partition])
        rng = random.Random(seed)
        for row in self.manifest["images"]:
            if row["image_id"] not in allowed:
                continue
            source = Path(row["path"])
            path = (source if source.is_absolute() else self.data_root / source).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"missing manifest image: {path}")
            with Image.open(path) as im:
                width, height = im.size
            annotated = []
            for d in row["defects"]:
                if d["bbox"][2] > width or d["bbox"][3] > height:
                    raise ValueError(f"{row['image_id']}: bbox exceeds actual image dimensions")
                annotated.append({"bbox": d["bbox"], "class_id": config.class_name_map[d["class_name"]]})
            positive, background = [], []
            for crop in unique_sahi_windows(width, height, self.window_size, self.stride):
                targets, ambiguous = window_defects(annotated, crop)
                if ambiguous:
                    continue
                if len(targets) > self.max_objects:
                    raise ValueError(f"{row['image_id']}: {len(targets)} objects exceeds max_objects={self.max_objects}")
                record = (path, row["image_id"], crop, targets)
                (positive if targets else background).append(record)
            self.records.extend(positive)
            self.records.extend(rng.sample(background, min(len(background), max_background_per_image)))
        if not allowed or not self.records:
            raise ValueError(f"{partition}: empty partition or no usable training windows")
        if not allowed.issubset({r[1] for r in self.records}):
            raise ValueError("some selected image IDs generated no usable windows")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, image_id, (x0, y0, x1, y1), targets = self.records[idx]
        with Image.open(path) as f:
            crop = f.convert("RGB").crop((x0, y0, x1, y1))
        # On images smaller than the window we pad, rather than stretching GT.
        cw, ch = crop.size
        if (cw, ch) != (self.img_size, self.img_size):
            canvas = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
            canvas.paste(crop, (0, 0))
            crop = canvas
        # Correct for padding BEFORE flipping: normalized targets are now in
        # padded 640x640 coordinates rather than the visible-image coordinates.
        scale_x, scale_y = cw / self.img_size, ch / self.img_size
        arr = [[cx * scale_x, cy * scale_y, w * scale_x, h * scale_y, cls]
               for cx, cy, w, h, cls in targets]
        # PR3 experiments adopt only bbox-safe augmentation on the train partition.
        if self.partition == "train" and config.use_augmentation:
            if random.random() < 0.5:
                crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                arr = [[1.0 - a, b, w, h, c] for a, b, w, h, c in arr]
            factor = random.uniform(0.85, 1.15)
            crop = ImageEnhance.Contrast(crop).enhance(factor)
            crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.85, 1.15))
        padded = torch.zeros((self.max_objects, 5), dtype=torch.float32)
        if arr:
            padded[:len(arr)] = torch.tensor(arr, dtype=torch.float32)
        meta = {"img_path": str(path), "image_id": image_id, "window": (x0, y0, x1, y1), "original_size": (cw, ch)}
        return _NORMALIZE(self.to_tensor(crop)), torch.tensor(len(arr), dtype=torch.long), padded, meta


def make_manifest_loaders(manifest_path, split_path, data_root):
    train = ManifestWindowDataset(manifest_path, split_path, data_root, "train")
    val = ManifestWindowDataset(manifest_path, split_path, data_root, "val")
    common = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory)
    return DataLoader(train, shuffle=True, **common), DataLoader(val, shuffle=False, **common)
