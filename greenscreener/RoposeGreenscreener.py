import cv2
import os

from typing import List, Tuple

import greenscreener.config as config

from guthoms_helpers.common_stuff.DataPreloader import DataPreloader
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
import ropose_dataset_tools.DataSetLoader as datasetLoader
from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset
import numpy as np
import random

class Greenscreener:

    def __init__(self, datasetDir: str, imageScale: Tuple[int, int] = None):
        self.backgrounds: List[np.array] = []
        self.originalFileNames: List[str] = []
        self.datasetDir: str = datasetDir
        self.imageScale: str = imageScale

        print("Loading data for RoposeGreenscreener!")
        self.datasets = datasetLoader.LoadDataSet(datasetDir)

        #shuffle list for better randomness
        random.shuffle(self.datasets)
        random.shuffle(self.datasets)

        self.preloader = DataPreloader(self.datasets, loadMethod=Greenscreener.LoadDataset, maxPreloadCount=300,
                                       infinite=True)

    @staticmethod
    def LoadDataset(dataset: Dataset):
        image = cv2.imread(dataset.rgbFrame.filePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, dataset

    @staticmethod
    def FitImageSizes(targetSpec: np.array, image: np.array):
        factor = (1.0, 1.0)
        if image.shape != targetSpec.shape:
            image = cv2.resize(image, dsize=(targetSpec.shape[1], targetSpec.shape[0]))
            factor = (image.shape[0]/targetSpec.shape[0], image.shape[1]/targetSpec.shape[1])
        return image, factor

    def AddForeground(self, image: np.array):

        foregroundImg, dataset = self.preloader.Next()

        foreground, factor = self.FitImageSizes(targetSpec=image, image=foregroundImg)

        for boundingBox in dataset.yoloData.boundingBoxes:
            dataset.yoloData.resizedBoundingBoxes.append(boundingBox.ScaleBB(factor[0], factor[1]))

        hsvImage = cv2.cvtColor(foreground, cv2.COLOR_RGB2HSV)

        bgMask = cv2.inRange(hsvImage, config.lowerTH, config.upperTH)
        fgMask = cv2.bitwise_not(bgMask)

        foreground = cv2.bitwise_or(foreground, foreground, mask=fgMask)
        background = cv2.bitwise_or(image, image, mask=bgMask)

        res = cv2.bitwise_or(background, foreground)
        return res, dataset.yoloData


