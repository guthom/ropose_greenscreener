import unittest
import os
from unittest import TestCase
from greenscreener.RoposeGreenscreener import RoposeGreenscreener
import numpy as np
import cv2

class GreenscreenerTests(TestCase):

    def setUp(self):
        #load test data
        dirName = os.path.dirname(os.path.abspath(__file__))
        dirName = os.path.join(dirName, "test_data")
        datasetDir = os.path.join(os.path.join(dirName, "ropose_test_dataset/"))

        self.imgNorm = cv2.imread(os.path.join(dirName, "samples_normal/ropose.png"))
        self.imgNorm = cv2.cvtColor(self.imgNorm, cv2.COLOR_BGR2RGB)

        self.result = cv2.imread(os.path.join(dirName, "result_dataset.png"))
        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)

        self.greensceener = RoposeGreenscreener(datasetDir)
        self.greensceenerPreloader = RoposeGreenscreener(datasetDir, usePreloader=True)


    def test_roposeGreenscreener(self):
        res, additionalYoloData = self.greensceener.AddForeground(image=self.imgNorm)
        self.assertTrue(np.array_equal(res, self.result))
        res, additionalYoloData = self.greensceenerPreloader.AddForeground(image=self.imgNorm)
        self.assertTrue(np.array_equal(res, self.result))

    def tearDown(self):
        self.greensceener.Shutdown()
        self.greensceenerPreloader.Shutdown()
