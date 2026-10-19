"""Fast, checkpoint-free regression tests for the multi-defect inference repair."""
import unittest

import torch
from inference_utils import class_aware_keep, unique_sahi_windows


class WindowTests(unittest.TestCase):
    def test_no_duplicate_large_image_windows(self):
        windows = unique_sahi_windows(4096, 2736, 640, 320)
        self.assertEqual(len(windows), 96)
        self.assertEqual(len(windows), len(set(windows)))
        self.assertIn((3456, 2096, 4096, 2736), windows)

    def test_small_image_gets_one_padded_window(self):
        self.assertEqual(unique_sahi_windows(300, 200, 640, 320), [(0, 0, 300, 200)])

    def test_coverage_and_edge(self):
        windows = unique_sahi_windows(999, 777, 640, 320)
        self.assertIn((359, 137, 999, 777), windows)
        for y in range(0, 777, 25):
            for x in range(0, 999, 25):
                self.assertTrue(any(x1 <= x < x2 and y1 <= y < y2
                                    for x1, y1, x2, y2 in windows))

    def test_reject_gapped_stride(self):
        with self.assertRaises(ValueError):
            unique_sahi_windows(1000, 1000, 640, 641)


class MultiClassNmsTests(unittest.TestCase):
    def test_overlapping_different_classes_survive(self):
        boxes = torch.tensor([[0., 0., 50., 50.], [1., 1., 49., 49.]])
        scores = torch.tensor([0.90, 0.88])
        labels = torch.tensor([1, 2])
        keep = class_aware_keep(boxes, scores, labels, 0.3, 10)
        self.assertEqual(set(keep.tolist()), {0, 1})

    def test_overlapping_same_class_suppressed(self):
        boxes = torch.tensor([[0., 0., 50., 50.], [1., 1., 49., 49.]])
        scores = torch.tensor([0.90, 0.88])
        labels = torch.tensor([1, 1])
        keep = class_aware_keep(boxes, scores, labels, 0.3, 10)
        self.assertEqual(keep.tolist(), [0])

    def test_topk_remains_explicit_and_configurable(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]])
        scores = torch.tensor([0.90, 0.88])
        labels = torch.tensor([1, 2])
        self.assertEqual(len(class_aware_keep(boxes, scores, labels, 0.3, 1)), 1)
        self.assertEqual(len(class_aware_keep(boxes, scores, labels, 0.3, 2)), 2)


if __name__ == '__main__':
    unittest.main()
