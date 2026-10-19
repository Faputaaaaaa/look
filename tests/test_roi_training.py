import unittest

import torch

from roi_training import append_gt_proposals, sample_gt_aware_rois


class GTProposalTests(unittest.TestCase):
    def test_append_and_original_count(self):
        rpn = torch.tensor([[0., 0., 4., 4.]])
        gt = torch.tensor([[10., 10., 20., 20.], [25., 25., 30., 30.]], requires_grad=True)
        out, n = append_gt_proposals(rpn, gt)
        self.assertEqual(n, 1)
        self.assertEqual(out.shape, (3, 4))
        self.assertFalse(out.requires_grad)
        self.assertTrue(torch.equal(out[1:], gt.detach()))

    def test_gt_only_if_rpn_returns_empty(self):
        out, n = append_gt_proposals(torch.empty(0, 4), torch.tensor([[0., 0., 2., 2.]]))
        self.assertEqual(n, 0)
        self.assertEqual(out.shape, (1, 4))

    def test_invalid_gt_rejected(self):
        with self.assertRaises(ValueError):
            append_gt_proposals(torch.empty(0, 4), torch.tensor([[0., 0., 0., 4.]]))
        with self.assertRaises(ValueError):
            append_gt_proposals(torch.empty(0, 4), torch.tensor([[0., float('nan'), 2., 4.]]))


class GTROISamplerTests(unittest.TestCase):
    def test_coverage_even_if_roi_align_reorders_gt(self):
        torch.manual_seed(42)
        # Original RPN indexes 0..5 and appended GT indexes 6,7,8.
        # ROI Align groups by FPN levels, not original proposal order.
        roi_ids = torch.tensor([7, 1, 5, 8, 3, 6, 0, 2, 4])
        pos = torch.tensor([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=torch.bool)
        neg = torch.tensor([0, 0, 1, 0, 0, 0, 1, 1, 0], dtype=torch.bool)
        p, n = sample_gt_aware_rois(roi_ids, pos, neg, original_rpn_count=6, gt_count=3, max_positive=4)
        self.assertEqual(set(roi_ids[p].tolist()) & {6, 7, 8}, {6, 7, 8})
        self.assertEqual(len(p), 4)
        self.assertTrue(bool(pos[p].all()))
        self.assertTrue(bool(neg[n].all()))
        self.assertEqual(len(n), 3)
        self.assertFalse(bool(torch.isin(n, p).any()))

    def test_all_gt_with_empty_rpn(self):
        ids = torch.tensor([1, 0])
        p, n = sample_gt_aware_rois(ids, torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), 0, 2)
        self.assertEqual(set(p.tolist()), {0, 1})
        self.assertEqual(n.numel(), 0)

    def test_no_gt_negatives_only(self):
        ids = torch.tensor([0, 1])
        p, n = sample_gt_aware_rois(ids, torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), 2, 0)
        self.assertEqual(p.numel(), 0)
        self.assertEqual(n.numel(), 0)

    def test_gt_over_budget(self):
        with self.assertRaisesRegex(ValueError, 'increase the limit'):
            sample_gt_aware_rois(torch.arange(3), torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool), 0, 3, max_positive=2)

    def test_missing_gt_feature_fails(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            sample_gt_aware_rois(torch.tensor([0, 1]), torch.tensor([True, True]), torch.tensor([False, False]), 1, 2)

    def test_duplicate_gt_feature_fails(self):
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            sample_gt_aware_rois(torch.tensor([0, 0]), torch.tensor([True, True]), torch.tensor([False, False]), 0, 2)

    def test_overlapping_pos_neg_masks_fail(self):
        with self.assertRaisesRegex(ValueError, 'both positive and negative'):
            sample_gt_aware_rois(torch.tensor([0]), torch.tensor([True]), torch.tensor([True]), 0, 1)

    def test_non_positive_gt_mask_fails(self):
        with self.assertRaisesRegex(ValueError, 'must be classified'):
            sample_gt_aware_rois(torch.tensor([0]), torch.tensor([False]), torch.tensor([True]), 0, 1)

    def test_positive_subsampling_respects_budget(self):
        torch.manual_seed(12)
        idx = torch.arange(100)
        pos = torch.ones(100, dtype=torch.bool)
        neg = torch.zeros(100, dtype=torch.bool)
        p, n = sample_gt_aware_rois(idx, pos, neg, original_rpn_count=98, gt_count=2)
        self.assertEqual(p.numel(), 32)
        self.assertEqual(set(idx[p].tolist()) & {98, 99}, {98, 99})
        self.assertEqual(n.numel(), 0)

    def test_only_genuine_negatives_sampled(self):
        idx = torch.tensor([0, 1, 2, 3, 4])
        p, n = sample_gt_aware_rois(idx, torch.tensor([0, 0, 0, 0, 1], dtype=torch.bool),
                                    torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool), 4, 1)
        self.assertEqual(p.tolist(), [4])
        self.assertEqual(set(n.tolist()), {0, 2})


if __name__ == '__main__':
    unittest.main()
