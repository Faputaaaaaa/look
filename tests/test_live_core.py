import unittest
import numpy as np
import cv2

from defect_platform.live_core import (
    FrameJob, LatestFrameMailbox, MotionBoxes, bounded_roi, global_detections,
)


class LiveCoreTest(unittest.TestCase):
    def test_roi_full(self):
        self.assertEqual(bounded_roi((1080, 1920, 3), None), (0,0,1920,1080))

    def test_roi_clip(self):
        self.assertEqual(bounded_roi((100,150,3), (-5,-2,400,500)), (0,0,150,100))

    def test_roi_minimum(self):
        with self.assertRaises(ValueError):
            bounded_roi((100,100,3),(1,1,10,10))

    def test_roi_invalid(self):
        with self.assertRaises(ValueError):
            bounded_roi((100,100,3),(0,0,float('nan'),100))

    def test_global_coordinates(self):
        dets=[{"bbox":(5,3,20,30),"class_name":"dirty","confidence":0.8}]
        converted=global_detections(dets,(50,30,150,130),(160,200,3))
        self.assertEqual(converted[0]["bbox"],(55.,33.,70.,60.))
        self.assertEqual(converted[0]["class_name"],"dirty")

    def test_global_clip(self):
        dets=[{"bbox":(-50,-20,10000,10000),"class_name":"missing"}]
        self.assertEqual(global_detections(dets,(50,30,150,130),(80,100,3))[0]["bbox"], (0.,10.,100.,80.))

    def test_global_degenerate_dropped(self):
        dets=[{"bbox":(10000,0,10001,1)}]
        self.assertEqual(global_detections(dets,(0,0,100,100),(100,100,3)),[])

    def test_latest_mailbox(self):
        m=LatestFrameMailbox()
        m.offer("older");m.offer("latest")
        self.assertEqual(m.take(0),"latest")
        self.assertIsNone(m.take(0.001))

    def test_mailbox_close(self):
        m=LatestFrameMailbox();m.offer("old");m.close()
        self.assertIsNone(m.take(0))
        self.assertFalse(m.offer("new"))

    @staticmethod
    def patterned():
        rng=np.random.default_rng(1002)
        img=np.zeros((240,320,3),np.uint8)
        img[65:128,75:155]=rng.integers(20,255,size=(63,80,3),dtype=np.uint8)
        return img

    def test_tracker_short_shift(self):
        base=self.patterned()
        shifted=cv2.warpAffine(base, np.float32([[1,0,4],[0,1,3]]), (320,240))
        t=MotionBoxes(max_age=3)
        ts=100.
        t.ingest(1,ts,base)
        t.ingest(2,ts+.06,shifted)
        self.assertTrue(t.install(1,ts,[{"bbox":(80.,72.,140.,120.),
            "class_name":"dirty","confidence":0.9}]))
        visible=t.visible(ts+.07)
        self.assertEqual(len(visible),1)
        self.assertAlmostEqual(visible[0]["bbox"][0],84,delta=1.8)
        self.assertAlmostEqual(visible[0]["bbox"][1],75,delta=1.8)

    def test_tracker_stale_result_rejected(self):
        img=self.patterned()
        t=MotionBoxes(max_age=2)
        t.ingest(1,10,img)
        t.ingest(2,20,img)
        self.assertFalse(t.install(1,10,[{"bbox":(80,80,140,120)}]))

    def test_tracker_untracked_blank_frame(self):
        t=MotionBoxes(max_age=3)
        img=np.zeros((160,220,3),np.uint8)
        t.ingest(1,10,img)
        t.install(1,10,[{"bbox":(30,30,70,70)}])
        self.assertEqual(t.visible(10), [])

    def test_expiration_hides_old_detections(self):
        img=self.patterned()
        t=MotionBoxes(max_age=1.5)
        t.ingest(1,10,img)
        t.install(1,10,[{"bbox":(80,70,130,125)}])
        self.assertEqual(t.visible(20),[])

    def test_res_change_resets(self):
        t=MotionBoxes(max_age=3)
        t.ingest(1,10,self.patterned())
        t.ingest(2,11,np.zeros((300,400,3),np.uint8))
        self.assertEqual(len(t.history),1)

    def test_empty_detections(self):
        t=MotionBoxes()
        t.ingest(1,10,self.patterned())
        self.assertTrue(t.install(1,10,[]))
        self.assertEqual(t.visible(10),[])


if __name__=="__main__":
    unittest.main()
