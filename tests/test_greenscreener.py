import unittest
import os
from unittest import TestCase
from greenscreener.ImageGreenscreener import ImageGreenscreener
import numpy as np
import cv2
import copy
from guthoms_helpers.input_provider.ImageDirectory import ImageDirectory

class GreenscreenerTests(TestCase):

    def setUp(self):

        #load test data
        dirName = os.path.dirname(os.path.abspath(__file__))
        dirName = os.path.join(dirName, "test_data")

        self.greensceenerPreloader = ImageGreenscreener(os.path.join(dirName, "backgrounds_normal"), usePreloader=True)
        self.greensceener = ImageGreenscreener(os.path.join(dirName, "backgrounds_normal"))

        self.imgGreen = cv2.imread(os.path.join(dirName, "backgrounds_normal/ropose_greenscreened.png"))
        self.imgGreen = cv2.cvtColor(self.imgGreen, cv2.COLOR_BGR2RGB)

        self.imgGreenResized = cv2.imread(os.path.join(dirName, "backgrounds_resized/ropose_greenscreened_resized.png"))
        self.imgGreenResized = cv2.cvtColor(self.imgGreenResized, cv2.COLOR_BGR2RGB)

        self.imgNorm = cv2.imread(os.path.join(dirName, "samples_normal/ropose.png"))
        self.imgNorm = cv2.cvtColor(self.imgNorm, cv2.COLOR_BGR2RGB)

        self.imgNormResized = cv2.imread(os.path.join(dirName, "samples_resized/ropose_resized.png"))
        self.imgNormResized = cv2.cvtColor(self.imgNormResized, cv2.COLOR_BGR2RGB)

        self.result = cv2.imread(os.path.join(dirName, "result_ropose.png"))
        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)

        self.resultResized = cv2.imread(os.path.join(dirName, "result_ropose_resized.png"))
        self.resultResized = cv2.cvtColor(self.resultResized, cv2.COLOR_BGR2RGB)


    def test_foregroundExchange(self):
        res = self.greensceenerPreloader.AddForeground(image=copy.copy(self.imgNorm))
        self.assertTrue(np.array_equal(res, self.result))
        res = self.greensceener.AddForeground(image=copy.copy(self.imgNorm))
        self.assertTrue(np.array_equal(res, self.result))

    def test_backgroundExchange(self):
        res = self.greensceenerPreloader.AddBackground(image=copy.copy(self.imgGreen), background=copy.copy(self.imgNorm))
        self.assertTrue(np.array_equal(res, self.result))
        res = self.greensceener.AddBackground(image=copy.copy(self.imgGreen), background=copy.copy(self.imgNorm))
        self.assertTrue(np.array_equal(res, self.result))

    def test_foregroundExchangeDiffSize(self):
        res = self.greensceenerPreloader.AddForeground(image=copy.copy(self.imgNormResized))
        self.assertTrue(np.array_equal(res, self.resultResized))
        res = self.greensceener.AddForeground(image=copy.copy(self.imgNormResized))
        self.assertTrue(np.array_equal(res, self.resultResized))


    def test_creation(self):
        #return
        def BGRToRGB(img):
            return img
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        #return
        myDir = "/home/thomas/Desktop/Raws"
        inputProvider = ImageDirectory(myDir, preprocessingFunction=BGRToRGB)

        counter = 0
        self.greensceener = ImageGreenscreener("/home/thomas/greenscreener/real/")
        while not inputProvider.finished():
            image = inputProvider.GetData()
            res = self.greensceener.AddBackground(image=image)

            fileName = myDir + "/res/res_" + str(counter) +".png"

            cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fileName, res)
            counter +=1

    def tearDown(self):
        self.greensceener.Shutdown()
        self.greensceenerPreloader.Shutdown()
