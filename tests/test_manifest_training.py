import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from config import config
from manifest_training import ManifestWindowDataset, window_defects


class WindowTargetsTest(unittest.TestCase):
    def test_two_classes_survive_same_window(self):
        ds = [
            {"bbox": [10, 20, 30, 40], "class_id": 0},
            {"bbox": [50, 50, 70, 70], "class_id": 1},
        ]
        targets, ambiguous = window_defects(ds, (0, 0, 80, 80))
        self.assertFalse(ambiguous)
        self.assertEqual([x[-1] for x in targets], [0, 1])
        self.assertAlmostEqual(targets[0][0], 0.25)

    def test_partly_cut_objects_are_ignored_not_background(self):
        t, a = window_defects([{"bbox": [0, 0, 100, 100], "class_id": 0}], (95, 0, 195, 100))
        self.assertEqual(t, [])
        self.assertTrue(a)

    def test_minimum_visible_area(self):
        t, a = window_defects([{"bbox": [0, 0, 20, 20], "class_id": 0}], (10, 0, 90, 80))
        self.assertFalse(a)
        self.assertEqual(len(t), 1)
        self.assertAlmostEqual(t[0][2], 0.125)

    def test_tiny_partial_rejected(self):
        t, a = window_defects([{"bbox": [0, 0, 3, 3], "class_id": 0}], (0, 0, 80, 80))
        self.assertEqual(t, [])
        self.assertTrue(a)


class ManifestLoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = []
        self.orig = (config.window_size, config.window_stride, config.img_size, config.use_augmentation)
        config.window_size = 80
        config.window_stride = 40
        config.img_size = 80
        config.use_augmentation = False
        for id_, sku, group, defects in [
            ("one.png", "sku-A", "g1", [{"class_name":"missing","bbox":[10,10,20,20]}, {"class_name":"dirty","bbox":[30,30,40,40]}]),
            ("two.png", "sku-A", "g2", []),
            ("three.png", "sku-B", "g3", [{"class_name":"missing","bbox":[10,10,20,20]}]),
        ]:
            Image.new('RGB', (80, 80), 'gray').save(self.root / id_)
            self.rows.append({"image_id":id_,"path":id_,"sku_id":sku,"capture_group":group,"defects":defects})
        self.mpath = self.root / 'manifest.json'
        self.spath = self.root / 'split.json'
        self.mpath.write_text(json.dumps({"classes":["missing","dirty"],"images":self.rows}),encoding="utf-8")
        self.spath.write_text(json.dumps({"train":["one.png"],"val":["two.png"],"test_unseen_sku":["three.png"],"heldout_skus":["sku-B"]}),encoding="utf-8")

    def tearDown(self):
        config.window_size, config.window_stride, config.img_size, config.use_augmentation = self.orig
        self.tmp.cleanup()

    def test_multi_instance_tensor_and_cls(self):
        ds = ManifestWindowDataset(self.mpath,self.spath,self.root,"train")
        im, n, boxes, meta = ds[0]
        self.assertEqual(n.item(),2)
        self.assertEqual(set(boxes[:2,4].int().tolist()),{0,1})
        self.assertEqual(tuple(im.shape),(3,80,80))
        self.assertEqual(meta['image_id'],'one.png')

    def test_normal_val_and_unseen_never_in_train(self):
        tr = ManifestWindowDataset(self.mpath,self.spath,self.root,"train")
        val = ManifestWindowDataset(self.mpath,self.spath,self.root,"val")
        self.assertEqual({r[1] for r in tr.records},{'one.png'})
        self.assertEqual({r[1] for r in val.records},{'two.png'})
        self.assertEqual(val[0][1].item(),0)

    def test_cannot_load_unseen(self):
        with self.assertRaisesRegex(ValueError,'only train/val'):
            ManifestWindowDataset(self.mpath,self.spath,self.root,"test_unseen_sku")

    def test_mismatched_manifest_classes(self):
        m=json.loads(self.mpath.read_text())
        m['classes'].reverse()
        self.mpath.write_text(json.dumps(m),encoding="utf-8")
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            ManifestWindowDataset(self.mpath,self.spath,self.root,"train")

    def test_group_leakage_rejected(self):
        m=json.loads(self.mpath.read_text())
        m['images'][1]['capture_group']='g1'
        self.mpath.write_text(json.dumps(m),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,'capture_group leaks'):
            ManifestWindowDataset(self.mpath,self.spath,self.root,"train")

    def test_bbox_outside_image_fails(self):
        m=json.loads(self.mpath.read_text())
        m['images'][0]['defects'][0]['bbox']=[10,10,100,110]
        self.mpath.write_text(json.dumps(m),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,'exceeds actual'):
            ManifestWindowDataset(self.mpath,self.spath,self.root,"train")

    def test_gt_budget_fail_loud(self):
        orig = config.max_objects
        try:
            config.max_objects=1
            with self.assertRaisesRegex(ValueError,'exceeds max_objects'):
                ManifestWindowDataset(self.mpath,self.spath,self.root,"train")
        finally:
            config.max_objects=orig

    def test_small_image_padding_preserves_bbox(self):
        # exact 80x80 under normal config is no padding; reduce actual size in fixture
        Image.new('RGB',(60,60),'gray').save(self.root/'one.png')
        ds=ManifestWindowDataset(self.mpath,self.spath,self.root,"train")
        im, n, boxes, meta=ds[0]
        self.assertEqual(n.item(),2)
        self.assertAlmostEqual(boxes[0,0].item(),15/80)
        self.assertAlmostEqual(boxes[0,2].item(),10/80)

    def test_train_flip_targets(self):
        config.use_augmentation=True
        ds=ManifestWindowDataset(self.mpath,self.spath,self.root,"train")
        with patch('random.random',return_value=0.0), patch('random.uniform',return_value=1.0):
            _,n,boxes,_=ds[0]
        self.assertEqual(n.item(),2)
        self.assertAlmostEqual(boxes[0,0].item(),1-15/80)

if __name__ == '__main__': unittest.main()
