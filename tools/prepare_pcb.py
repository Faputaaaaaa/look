"""PCB 基准数据准备：从 PKU-Market-PCB-corrected 的板级划分生成 pcb_split.json。

作用：
1. 把 ImageSets/train_board.txt / val_board.txt（板级隔离 541/152，与文献一致）
   落盘为 data/pcb/pcb_split.json —— PCB 线的划分单一事实来源
   （对齐项目根 split.json 的「划分必须落盘」惯例）。
2. 板级泄漏断言：train/val 文件名不相交 且 板 ID 不相交。
3. 每张图记录实际尺寸（10 种板尺寸，必须 per-image 读取）+ 归一化框，
   供 dataset.py 的 file-list 加载与 tools/check_pcb_anchors.py 使用。

纯标准库，无 torch/numpy 依赖。运行（-I 隔离模式，数据集目录不参与 sys.path）：
    D:/Python3.14/python.exe -I tools/prepare_pcb.py --src "D:\\百度网盘下载\\PKU-Market-PCB-corrected" --out data/pcb/pcb_split.json
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# Windows GBK 控制台会把 UTF-8 中文打成乱码（项目 HANDOFF 有记录），强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def percentile(sorted_vals, p):
    """sorted_vals 升序列表；线性插值百分位（等价 np.percentile 的默认口径）。"""
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def read_split(split_path):
    """读 ImageSets 文件：每行一个图片文件名（含 .jpg）。"""
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_image_size(xml_path):
    """从 VOC XML 读 <size>（PCB 有 10 种板尺寸，必须按图读，不能用全局常量）。"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path}: 缺少 <size>")
    return int(size.find("width").text), int(size.find("height").text)


def read_label_boxes(label_path):
    """读 YOLO 标签，返回 [[cls_id, cx, cy, w, h]]（归一化）。"""
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        boxes.append([int(parts[0])] + [float(x) for x in parts[1:5]])
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=r"D:\百度网盘下载\PKU-Market-PCB-corrected",
                        help="corrected 数据集根目录")
    parser.add_argument("--out", default="data/pcb/pcb_split.json",
                        help="输出 split JSON")
    parser.add_argument("--quiet", action="store_true", help="只输出 [OK]/[FAIL] 行")
    args = parser.parse_args()

    src = Path(args.src)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images_dir = src / "images"
    labels_dir = src / "labels"
    annot_dir = src / "annotations"
    for d in (images_dir, labels_dir, annot_dir):
        if not d.is_dir():
            print(f"[FAIL] 目录不存在: {d}")
            sys.exit(1)

    # ---- 1. 类名（从数据集 classes.txt 读，不硬编码） ----
    classes = [c for c in (src / "classes.txt").read_text(encoding="utf-8").splitlines()
               if c.strip()]
    if len(classes) != 6:
        print(f"[FAIL] classes.txt 应有 6 类，实际 {len(classes)}: {classes}")
        sys.exit(1)
    if not args.quiet:
        print(f"[info] classes ({len(classes)}): {', '.join(classes)}")

    # ---- 2. 读板级划分 ----
    train_names = read_split(src / "ImageSets" / "train_board.txt")
    val_names = read_split(src / "ImageSets" / "val_board.txt")
    checks = {}

    def board_of(name):
        return name.split("_")[0]

    train_boards = sorted(set(board_of(n) for n in train_names))
    val_boards = sorted(set(board_of(n) for n in val_names))

    overlap = set(train_names) & set(val_names)
    checks["train_val_files_disjoint"] = len(overlap) == 0
    checks["train_val_boards_disjoint"] = len(set(train_boards) & set(val_boards)) == 0

    # 与 README 记载的板级划分一致（train=01,04,05,06,07,08,09 / val=10,11,12）
    checks["boards_match_readme"] = (
        train_boards == ["01", "04", "05", "06", "07", "08", "09"]
        and val_boards == ["10", "11", "12"]
    )

    # 全部 693 张图都被划分覆盖（无漏分）
    all_on_disk = sorted(p.name for p in images_dir.glob("*.jpg"))
    covered = set(train_names) | set(val_names)
    checks["all_images_covered"] = set(all_on_disk) == covered

    if not all(checks.values()):
        for k, v in checks.items():
            print(f"[FAIL] {k} = {v}")
        print(f"[FAIL] 重叠文件: {sorted(overlap)[:5]}")
        sys.exit(1)
    if not args.quiet:
        print(f"[OK] 板级泄漏检查全部通过")
        print(f"      train={len(train_names)} 图 (板 {train_boards}) / "
              f"val={len(val_names)} 图 (板 {val_boards})")

    # ---- 3. 逐图记录尺寸 + 归一化框 ----
    def build_rows(names):
        rows = []
        for name in names:
            stem = Path(name).stem
            w, h = read_image_size(annot_dir / f"{stem}.xml")
            boxes = read_label_boxes(labels_dir / f"{stem}.txt")
            rows.append({
                "file": name,
                "board": board_of(name),
                "width": w,
                "height": h,
                "boxes": boxes,  # [[cls, cx, cy, w, h]] 归一化
            })
        return rows

    train_rows = build_rows(train_names)
    val_rows = build_rows(val_names)

    # ---- 4. 统计（尺寸分布 / 每类框数 / 面积占比） ----
    def collect_stats(rows):
        sqrt_areas = []
        area_fracs = []
        per_cls_sqrt = {i: [] for i in range(len(classes))}
        per_cls_frac = {i: [] for i in range(len(classes))}
        n_boxes_list = [len(r["boxes"]) for r in rows]
        for r in rows:
            for cls_id, cx, cy, w, h in r["boxes"]:
                px_w, px_h = w * r["width"], h * r["height"]
                sqrt_areas.append((px_w * px_h) ** 0.5)
                area_fracs.append((px_w * px_h) / (r["width"] * r["height"]))
                per_cls_sqrt[cls_id].append((px_w * px_h) ** 0.5)
                per_cls_frac[cls_id].append((px_w * px_h) / (r["width"] * r["height"]))
        return {
            "n_images": len(rows),
            "n_boxes": len(sqrt_areas),
            "boxes_per_image": {
                "min": min(n_boxes_list), "max": max(n_boxes_list),
                "mean": round(sum(n_boxes_list) / len(n_boxes_list), 2),
            },
            "sqrt_area_px": {
                "median": round(percentile(sorted(sqrt_areas), 0.5), 1),
                "p10": round(percentile(sorted(sqrt_areas), 0.1), 1),
                "p90": round(percentile(sorted(sqrt_areas), 0.9), 1),
            },
            "area_frac_median_pct": round(percentile(sorted(area_fracs), 0.5) * 100, 4),
            "per_class": {
                classes[i]: {
                    "n_boxes": len(per_cls_sqrt[i]),
                    "sqrt_area_median_px": round(percentile(sorted(per_cls_sqrt[i]), 0.5), 1),
                    "area_frac_median_pct": round(percentile(sorted(per_cls_frac[i]), 0.5) * 100, 4),
                }
                for i in range(len(classes))
            },
            "board_sizes": sorted({(r["width"], r["height"]) for r in rows}, reverse=True),
        }

    stats = {"train": collect_stats(train_rows), "val": collect_stats(val_rows)}

    # ---- 5. 落盘 ----
    payload = {
        "dataset": "PKU-Market-PCB-corrected",
        "source": str(src),
        "citation": "Huang & Wei, arXiv:1901.08204 (2019); "
                    "corrected release: KeenForgeAI/PKU-Market-PCB-corrected "
                    "(HF DOI 10.57967/hf/10554)",
        "classes": classes,
        "split_sources": {"train": "ImageSets/train_board.txt",
                          "val": "ImageSets/val_board.txt"},
        "created": datetime.now().isoformat(timespec="seconds"),
        "train": train_rows,
        "val": val_rows,
        "boards": {"train": train_boards, "val": val_boards},
        "checks": checks,
        "stats": stats,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    s = stats["train"]
    print(f"[OK] {out_path} 已生成")
    print(f"     train={s['n_images']} 图 / val={stats['val']['n_images']} 图")
    print(f"     框数: {s['n_boxes']} (train) / {stats['val']['n_boxes']} (val)")
    print(f"     每图框数: min={s['boxes_per_image']['min']} "
          f"max={s['boxes_per_image']['max']} mean={s['boxes_per_image']['mean']}")
    print(f"     sqrt(area): median={s['sqrt_area_px']['median']}px "
          f"p10={s['sqrt_area_px']['p10']}px p90={s['sqrt_area_px']['p90']}px")
    print(f"     面积占比中位: {s['area_frac_median_pct']}%")


if __name__ == "__main__":
    main()
