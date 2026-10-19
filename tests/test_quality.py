import unittest
from quality.manifest import validate_manifest
from quality.split import build_group_split, check_group_isolation
from quality.metrics import evaluate


def record(image_id, sku, group, defects):
    return {"image_id": image_id, "path": image_id + ".png", "sku_id": sku,
            "capture_group": group, "defects": [{"class_name": cls, "bbox": box} for cls, box in defects]}


class QualityTest(unittest.TestCase):
    def setUp(self):
        self.m = {"classes": ["missing", "dirty"], "images": [
            record("a", "skuA", "batch1", [("missing", [0, 0, 10, 10]), ("dirty", [20, 20, 30, 30])]),
            record("b", "skuA", "batch2", []),
            record("c", "skuB", "batch3", [("dirty", [5, 5, 15, 15])]),
            record("d", "skuC", "batch4", [("missing", [2, 2, 12, 12])])
        ]}

    def det(self, cls, box, score=0.9):
        return {"class_name": cls, "bbox": box, "confidence": score}

    def predictions(self):
        return [{"image_id": "a", "detections": [self.det("missing", [0, 0, 10, 10]), self.det("dirty", [20, 20, 30, 30])]},
                {"image_id": "b", "detections": []},
                {"image_id": "c", "detections": [self.det("dirty", [5, 5, 15, 15])]},
                {"image_id": "d", "detections": [self.det("missing", [2, 2, 12, 12])]}]

    def test_two_classes_on_one_image_and_ap(self):
        r = evaluate(self.m, self.predictions())
        self.assertEqual(r["micro_tp"], 4)
        self.assertEqual(r["micro_fp"], 0)
        self.assertAlmostEqual(r["mAP50"], 1)
        self.assertAlmostEqual(r["all_defects_found_rate"], 1)
        self.assertAlmostEqual(r["exact_defect_set_rate"], 1)

    def test_multi_instance_and_multi_class_counters(self):
        r = evaluate(self.m, self.predictions())
        self.assertEqual(r["n_multi_instance"], 1)
        self.assertEqual(r["n_multi_class"], 1)
        self.assertEqual(r["multi_class_all_found_rate"], 1)
        self.assertEqual(r["multi_class_exact_set_rate"], 1)
        p = self.predictions()
        p[0]["detections"] = p[0]["detections"][:1]
        r = evaluate(self.m, p)
        self.assertEqual(r["multi_class_all_found_rate"], 0)
        self.assertEqual(r["multi_class_exact_set_rate"], 0)
        self.assertEqual(r["all_defects_found_rate"], 2 / 3)

    def test_only_single_defect_images_have_null_multiclass_rate(self):
        m = {"classes": ["missing"], "images": [record("one", "A", "g1", [("missing", [0, 0, 10, 10])])]}
        p = [{"image_id": "one", "detections": [self.det("missing", [0, 0, 10, 10])]}]
        r = evaluate(m, p)
        self.assertEqual(r["n_multi_class"], 0)
        self.assertIsNone(r["multi_class_all_found_rate"])

    def test_duplicate_is_fp(self):
        p = self.predictions()
        p[0]["detections"].append(self.det("missing", [0, 0, 10, 10], 0.8))
        r = evaluate(self.m, p)
        self.assertEqual(r["micro_tp"], 4)
        self.assertEqual(r["micro_fp"], 1)
        self.assertEqual(r["exact_defect_set_rate"], 2/3)
        self.assertEqual(r["all_defects_found_rate"], 1)

    def test_wrong_class_not_tp(self):
        p = self.predictions()
        p[0]["detections"][0]["class_name"] = "dirty"
        r = evaluate(self.m, p)
        self.assertEqual(r["micro_tp"], 3)
        self.assertEqual(r["micro_fp"], 1)
        self.assertEqual(r["micro_fn"], 1)

    def test_operating_score_is_independent_of_ap(self):
        p = self.predictions()
        p[0]["detections"][0]["confidence"] = 0.1
        r = evaluate(self.m, p, operating_score=0.5)
        self.assertEqual(r["micro_fn"], 1)
        self.assertAlmostEqual(r["mAP50"], 1)

    def test_normal_false_alarm(self):
        p = self.predictions()
        p[1]["detections"].append(self.det("missing", [0, 0, 5, 5]))
        r = evaluate(self.m, p)
        self.assertEqual(r["normal_false_alarm_rate"], 1)
        self.assertEqual(r["micro_fp"], 1)

    def test_missing_prediction_is_error(self):
        with self.assertRaises(ValueError):
            evaluate(self.m, self.predictions()[:3])

    def test_group_holdout(self):
        split = build_group_split(self.m, heldout_skus=["skuC"])
        self.assertEqual(split["test_unseen_sku"], ["d"])
        self.assertTrue(check_group_isolation(self.m, split))
        self.assertEqual(set(split["train"] + split["val"]), {"a", "b", "c"})
        self.assertGreater(sum(split["class_instance_counts"]["val"].values()), 0)

    def test_validation_stratification_uses_defective_group(self):
        m = {"classes": ["dirty"], "images": [
            record("normal1", "A", "g1", []),
            record("normal2", "A", "g2", []),
            record("pos1", "A", "g3", [("dirty", [0, 0, 10, 10])]),
            record("pos2", "A", "g4", [("dirty", [0, 0, 10, 10])]),
        ]}
        sp = build_group_split(m, val_fraction=0.25, seed=42)
        self.assertEqual(sp["class_instance_counts"]["val"]["dirty"], 1)
        self.assertEqual(sp["class_instance_counts"]["train"]["dirty"], 1)
        self.assertEqual(sp["coverage_warnings"], [])

    def test_unavoidable_missing_class_warning(self):
        m = {"classes": ["dirty"], "images": [
            record("normal", "A", "g1", []),
            record("pos", "A", "g2", [("dirty", [0, 0, 10, 10])]),
        ]}
        sp = build_group_split(m, val_fraction=0.5, seed=42)
        self.assertTrue(sp["coverage_warnings"])

    def test_group_leak_detection(self):
        split = build_group_split(self.m, ["skuC"])
        split["val"].append(split["train"][0])
        with self.assertRaises(ValueError):
            check_group_isolation(self.m, split)

    def test_holdout_group_cannot_mix_with_seen_sku(self):
        self.m["images"][3]["capture_group"] = "batch1"
        with self.assertRaises(ValueError):
            build_group_split(self.m, ["skuC"])

    def test_invalid_heldout_sku_claim_is_rejected(self):
        split = build_group_split(self.m, ["skuC"])
        split["heldout_skus"] = []
        with self.assertRaises(ValueError):
            check_group_isolation(self.m, split)

    def test_absent_defects_is_error(self):
        del self.m["images"][1]["defects"]
        with self.assertRaises(ValueError):
            validate_manifest(self.m)

    def test_duplicate_image_id_is_error(self):
        self.m["images"][1]["image_id"] = "a"
        with self.assertRaises(ValueError):
            validate_manifest(self.m)


if __name__ == "__main__":
    unittest.main()
