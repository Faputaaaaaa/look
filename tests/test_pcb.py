"""PCB 基准线回归测试：profile 切换 / file_list 白名单 / per-image 尺寸裁剪。

无权重、纯数据层，与 PR1/PR2 测试同风格（fast, checkpoint-free）。
运行：python -m unittest discover -s tests -t . -v
"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from config import config
from dataset import DefectDataset


def _make_fixture(root: Path):
    """造一个 3 图 2 标签的 YOLO 数据集（图片 320×240）。

    类别 id 用 xinpaide 映射内的 3/4（missing/dirty），避免触发
    「未知类跳过」警告污染测试输出（该行为有单独测试）。
    """
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for name in ("a", "b", "c"):
        Image.new("RGB", (320, 240), (128, 128, 128)).save(
            root / "images" / f"{name}.jpg")
    # a: 两个类别的框（id 3 和 id 4）；c: 一个框；b 无标签
    (root / "labels" / "a.txt").write_text(
        "3 0.5 0.5 0.2 0.2\n4 0.8 0.8 0.1 0.1\n", encoding="utf-8")
    (root / "labels" / "c.txt").write_text("3 0.2 0.3 0.15 0.2\n", encoding="utf-8")
    return root


class ProfileTests(unittest.TestCase):
    """config.set_profile 切换的正确性与隔离性。"""

    _PROFILE_FIELDS = (
        "dataset_profile", "data_dir", "label_format", "class_names",
        "class_name_map", "yolo_id_to_name", "pcb_split_json", "max_objects",
        "max_dets_per_image", "img_size", "preprocess_method", "window_size",
        "window_stride", "original_img_width", "original_img_height",
        "max_neg_windows_per_img", "use_mosaic", "use_random_crop_aug",
        "train_img_dirs", "train_label_dirs", "val_img_dirs", "val_label_dirs",
        "test_img_dirs", "test_label_dirs",
        "train_samples_per_subset", "val_samples_per_subset",
    )

    def setUp(self):
        # config 是全局单例：保存现场，tearDown 恢复（避免污染其他测试）
        self._saved = {k: getattr(config, k) for k in self._PROFILE_FIELDS}

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(config, k, v)

    def test_pcb_profile_sets_expected_fields(self):
        config.set_profile("pcb")
        self.assertEqual(len(config.class_names), 6)
        self.assertEqual(config.class_name_map["short"], 3)
        self.assertEqual(config.yolo_id_to_name[3], "short")
        self.assertEqual(config.img_size, 1024)
        self.assertEqual(config.preprocess_method, "resize")
        self.assertEqual(config.max_objects, 16)
        self.assertEqual(config.max_dets_per_image, 20)
        self.assertIsNone(config.max_neg_windows_per_img)
        self.assertTrue(config.pcb_split_json.endswith("pcb_split.json"))

    def test_pcb_mapping_isolated_from_xinpaide(self):
        # PCB id 3 = short、id 4 = spur；xinpaide 映射里 id 3 = missing、id 4 = dirty。
        # 两个映射必须互不污染（否则 PCB 标签会被静默映射成 missing/dirty）。
        config.set_profile("pcb")
        pcb_map = dict(config.yolo_id_to_name)
        self.assertEqual(pcb_map[3], "short")
        self.assertEqual(pcb_map[4], "spur")
        self.assertNotIn("dirty", config.class_names)

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            config.set_profile("nope")


class FileListTests(unittest.TestCase):
    def test_file_list_whitelist(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_fixture(root)
            ds = DefectDataset(
                root / "images", root / "labels",
                mode="val", label_format="yolo", preprocess_method="resize",
                file_list=["a.jpg", "c.jpg"])
            names = sorted(Path(s["img_path"]).name for s in ds.samples)
            self.assertEqual(names, ["a.jpg", "c.jpg"])

    def test_unknown_class_counted_not_silent(self):
        """[PCB] 映射错配的框必须计数（init 时汇总警告），不能静默吞掉。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images").mkdir(parents=True)
            (root / "labels").mkdir(parents=True)
            Image.new("RGB", (320, 240), (128, 128, 128)).save(
                root / "images" / "x.jpg")
            # id 0 = ghosting（不在 xinpaide class_name_map 里）
            (root / "labels" / "x.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            ds = DefectDataset(
                root / "images", root / "labels",
                mode="val", label_format="yolo", preprocess_method="resize")
            self.assertEqual(ds._skipped_unknown, 1)


class PerImageSizeTests(unittest.TestCase):
    def test_window_clip_uses_actual_image_size(self):
        """[PCB] _get_bboxes_in_window 必须按每图实际尺寸换算（回归测试）。

        1000×700 的图 + 靠右边缘的框：旧实现用全局 4096×2736 会把 bbox
        中心算到 3973px，任何窗口都框不住 → 正样本窗口数为 0；
        新实现按 1000×700 换算 → 至少 1 个正样本窗口，且 clip 不越界。
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images").mkdir(parents=True)
            (root / "labels").mkdir(parents=True)
            Image.new("RGB", (1000, 700), (128, 128, 128)).save(
                root / "images" / "t.jpg")
            (root / "labels" / "t.txt").write_text(
                "4 0.97 0.5 0.1 0.2\n", encoding="utf-8")
            self.assertNotEqual(config.original_img_width, 1000)  # 确保测的是 per-image 路径
            ds = DefectDataset(
                root / "images", root / "labels",
                mode="val", label_format="yolo", preprocess_method="sliding_window")
            pos = [s for s in ds.samples if len(s["all_bboxes"]) > 0]
            self.assertGreaterEqual(len(pos), 1)
            for b in pos[0]["all_bboxes"]:
                self.assertLessEqual(b["w"], 1.0 + 1e-6)
                self.assertLessEqual(b["cx"] + b["w"] / 2, 1.0 + 1e-6)


class UniqueWindowTests(unittest.TestCase):
    def test_unique_windows_no_duplicate_samples(self):
        """[PCB] use_unique_windows=True 时训练窗口枚举与推理同源（无重复）。

        960×960 图 + 640/320 窗口：旧 _generate_windows 的边界窗口与网格
        重合（960-640=320），会产生重复窗口；unique_sahi_windows 去重后
        窗口数 = 4。样本窗口 (x, y) 不应有重复。
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images").mkdir(parents=True)
            (root / "labels").mkdir(parents=True)
            Image.new("RGB", (960, 960), (128, 128, 128)).save(
                root / "images" / "u.jpg")
            (root / "labels" / "u.txt").write_text(
                "4 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            saved = config.use_unique_windows
            config.use_unique_windows = True
            try:
                ds = DefectDataset(
                    root / "images", root / "labels",
                    mode="val", label_format="yolo",
                    preprocess_method="sliding_window")
                keys = [(s["window"]["x"], s["window"]["y"]) for s in ds.samples]
                self.assertEqual(len(keys), len(set(keys)),
                                 "训练窗口有重复（未走 unique 枚举）")
                # 640/320 在 960×960 上 unique 窗口恰为 4 个
                self.assertEqual(len(set(keys)), 4)
            finally:
                config.use_unique_windows = saved


if __name__ == "__main__":
    unittest.main()
