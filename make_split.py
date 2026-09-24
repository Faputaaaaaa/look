"""按原图 ID 分层划分 train/val，落盘 split.json。

用途：
    - dirty / missing 两类各自按 80/20 划分（seed=42，可复现）。
    - 划分单位是「原图」（frame_XXXX.jpg），不是切片/窗口。
    - 切片（滑动窗口）是运行时从原图动态生成的，每个窗口的
      meta['img_path'] 即来源原图，因此按原图划分后，同一原图的所有
      切片必然同属 train 或 val，不存在跨集合泄露。

原图 ID 约定：
    - 数据目录 xinpaide/{dirty,missing}/images 下文件名同为 frame_XXXX.jpg，
      两类会重名。为保持全局唯一，ID 统一用「子集相对路径」：
      dirty/frame_0000.jpg、missing/frame_0000.jpg 等。

用法：
    python make_split.py
输出：
    split.json（与 CLAUDE.md 约定格式一致）
"""
import json
import random
from pathlib import Path

SEED = 42
DATA_DIR = Path(r"d:\deeplearning\learn\xinpaide")
SUBSETS = ['dirty', 'missing']
TRAIN_RATIO = 0.8
MIN_VAL_PER_CLASS = 15


def build_split() -> dict:
    rng = random.Random(SEED)
    split = {
        "seed": SEED,
        "split_type": "stratified_by_image",
        "train": [],
        "val": [],
        "by_class": {},
    }

    for subset in SUBSETS:
        img_dir = DATA_DIR / subset / "images"
        # 原图 ID = 子集相对路径，保证 dirty/missing 之间不重名
        ids = sorted(f"{subset}/{f.name}" for f in img_dir.glob("*.jpg"))
        rng.shuffle(ids)

        n_val = int(round(len(ids) * (1 - TRAIN_RATIO)))
        val = sorted(ids[:n_val])
        train = sorted(ids[n_val:])

        split["by_class"][subset] = {"train": train, "val": val}
        split["train"].extend(train)
        split["val"].extend(val)

    split["train"] = sorted(split["train"])
    split["val"] = sorted(split["val"])
    return split


def verify(split: dict) -> None:
    """验证划分合法性（不跨集合、每类 val 数量达标）。"""
    train = set(split["train"])
    val = set(split["val"])

    # 1. train / val 不重叠
    overlap = train & val
    assert not overlap, f"train/val 存在重叠: {overlap}"

    # 2. 每个子集内部 train/val 也不重叠，且每类 val 达标
    for subset in SUBSETS:
        t = set(split["by_class"][subset]["train"])
        v = set(split["by_class"][subset]["val"])
        assert not (t & v), f"{subset} 内部 train/val 重叠"
        assert not (v & overlap), f"{subset} val 与全局 val 不一致"
        n_val = len(v)
        n_total = n_val + len(t)
        print(f"  {subset}: train={len(t)}, val={n_val}, 总计={n_total}")
        if n_val < MIN_VAL_PER_CLASS:
            raise SystemExit(
                f"[STOP] {subset} val 仅 {n_val} 张 (< {MIN_VAL_PER_CLASS})，"
                f"不要自行调整比例，请写入 QUESTIONS.md 等待确认。"
            )

    print(f"  全局: train={len(train)}, val={len(val)}")
    assert len(train) + len(val) == 200, "总原图数应为 200（dirty/missing 各 100）"
    print("[OK] 同一原图只出现在 train 或 val 之一，无切片级泄露。")


if __name__ == "__main__":
    split = build_split()
    verify(split)
    out = Path(__file__).resolve().parent / "split.json"
    out.write_text(
        json.dumps(split, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[OK] split.json 已落盘: {out}")
