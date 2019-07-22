import cv2
from typing import List, Tuple

import greenscreener.config as config
import numpy as np

from guthoms_helpers.common_stuff.DataPreloader import DataPreloader
from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset
import ropose_dataset_tools.DataSetLoader as datasetLoader
import random


class RoposeGreenscreener(object):

    def __init__(self, datasetDir: str, imageScale: Tuple[int, int] = None, maxPreloadCount = 3000,
                 usePreloader: bool = False):
        self.backgrounds: List[np.array] = []
        self.originalFileNames: List[str] = []
        self.datasetDir: str = datasetDir
        self.imageScale: str = imageScale

        print("Loading data for RoposeGreenscreener!")
        self.datasets = datasetLoader.LoadDataSet(datasetDir)

        self.usePreloader = usePreloader

        self.preloader = None
        if self.usePreloader:
            self.preloader = DataPreloader(self.datasets, loadMethod=RoposeGreenscreener.LoadDataset,
                                       maxPreloadCount=maxPreloadCount, infinite=True, shuffleData=True,
                                       waitForBuffer=False)


    @staticmethod
    def LoadDataset(dataset: Dataset):
        image = cv2.imread(dataset.rgbFrame.filePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, dataset

    @staticmethod
    def FitImageSizes(targetSpec: np.array, image: np.array):
        factor = (1.0, 1.0)
        if image.shape != targetSpec.shape:
            factor = (targetSpec.shape[0]/image.shape[0], targetSpec.shape[1]/image.shape[1])
            image = cv2.resize(image, dsize=(targetSpec.shape[1], targetSpec.shape[0]))
        return image, factor

    def AddForeground(self, image: np.array):

        foregroundImg, dataset = self.GetRandomForeGround()

        foreground, factor = self.FitImageSizes(targetSpec=image, image=foregroundImg)

        for i in range(0, dataset.yoloData.boundingBoxes.__len__()):
            dataset.yoloData.boundingBoxes[i] = dataset.yoloData.boundingBoxes[i].ScaleBB(factor[1], factor[0])

        hsvImage = cv2.cvtColor(foreground, cv2.COLOR_RGB2HSV)

        bgMask = cv2.inRange(hsvImage, config.lowerTH, config.upperTH)
        fgMask = cv2.bitwise_not(bgMask)

        foreground = cv2.bitwise_or(foreground, foreground, mask=fgMask)
        background = cv2.bitwise_or(image, image, mask=bgMask)

        res = cv2.bitwise_or(background, foreground)
        return res, dataset.yoloData

    def GetRandomForeGround(self):

        if self.usePreloader:
            return self.preloader.Next()
        else:
            index = random.randint(0, self.datasets.__len__()-1)
            return self.LoadDataset(self.datasets[index])

    def Shutdown(self):
        if self.preloader is not None:
            self.preloader.shutdown()
