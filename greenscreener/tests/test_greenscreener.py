import unittest
import os
from unittest import TestCase
from greenscreener.Greenscreener import Greenscreener
import numpy as np
import cv2

class GreenscreenerTests(TestCase):

    def setUp(self):
        self.greensceener = Greenscreener()

        #load test data
        dirName = os.path.dirname(os.path.abspath(__file__))
        dirName = os.path.join(dirName, "test_data")

        self.imgGreen = cv2.imread(os.path.join(dirName, "ropose_greenscreened.png"))
        self.imgGreen = cv2.cvtColor(self.imgGreen, cv2.COLOR_BGR2RGB)

        self.imgGreenResized = cv2.imread(os.path.join(dirName, "ropose_greenscreened_resized.png"))
        self.imgGreenResized = cv2.cvtColor(self.imgGreenResized, cv2.COLOR_BGR2RGB)

        self.imgNorm = cv2.imread(os.path.join(dirName, "ropose.png"))
        self.imgNorm = cv2.cvtColor(self.imgNorm, cv2.COLOR_BGR2RGB)

        self.imgNormResized = cv2.imread(os.path.join(dirName, "ropose_resized.png"))
        self.imgNormResized = cv2.cvtColor(self.imgNormResized, cv2.COLOR_BGR2RGB)

        self.result = cv2.imread(os.path.join(dirName, "result_ropose.png"))
        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)

        self.resultResized = cv2.imread(os.path.join(dirName, "result_ropose_resized.png"))
        self.resultResized = cv2.cvtColor(self.resultResized, cv2.COLOR_BGR2RGB)


    def test_foregroundExchange(self):
        res = self.greensceener.AddForeground(image=self.imgNorm, foreground=self.imgGreen)
        self.assertTrue(np.array_equal(res, self.result))

    def test_backgroundExchange(self):
        res = self.greensceener.AddBackground(image=self.imgGreen, background=self.imgNorm)
        self.assertTrue(np.array_equal(res, self.result))

    def test_foregroundExchangeDiffSize(self):
        res = self.greensceener.AddForeground(image=self.imgNormResized, foreground=self.imgGreen)
        self.assertTrue(np.array_equal(res, self.resultResized))
