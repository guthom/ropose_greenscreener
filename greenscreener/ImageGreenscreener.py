import cv2
from typing import List, Tuple

import numpy as np
from threading import Lock
import greenscreener.config as config

from guthoms_helpers.common_stuff.DataPreloader import DataPreloader
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper

class ImageGreenscreener(object):
    def __init__(self, imageDir: str = config.imageDir, imageScale: Tuple[int, int] = None, maxPreloadCount = 3000):
        self.backgrounds: List[np.array] = []
        self.originalFileNames: List[str] = []
        self.imageDir: str = imageDir
        self.imageScale: str = imageScale

        self.fileList = DirectoryHelper.ListDirectoryFiles(dirPath=self.imageDir, fileEndings=[".jpg", ".png"])


        self.preloader = DataPreloader(self.fileList, loadMethod=self.LoadImage, maxPreloadCount=maxPreloadCount,
                                       infinite=True, shuffleData=True, waitForBuffer=False)


    @staticmethod
    def LoadImage(path: str):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def FitImageSizes(targetSpec: np.array, image: np.array):

        if image.shape != targetSpec.shape:
            image = cv2.resize(image, dsize=(targetSpec.shape[1], targetSpec.shape[0]))
        return image

    def AddBackground(self, image: np.array, background: np.array=None):

        if background is None:
            background = self.preloader.Next()

        background = self.FitImageSizes(targetSpec=image, image=background)

        hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        bgMask = cv2.inRange(hsvImage, config.lowerTH, config.upperTH)
        fgMask = cv2.bitwise_not(bgMask)

        foreground = cv2.bitwise_or(image, image, mask=fgMask)
        background = cv2.bitwise_or(background, background, mask=bgMask)

        res = cv2.bitwise_or(background, foreground)
        return res

    def AddForeground(self, image: np.array, foreground: np.array=None):

        if foreground is None:
            foreground = self.preloader.Next()

        foreground = self.FitImageSizes(targetSpec=image, image=foreground)

        hsvImage = cv2.cvtColor(foreground, cv2.COLOR_RGB2HSV)

        bgMask = cv2.inRange(hsvImage, config.lowerTH, config.upperTH)
        fgMask = cv2.bitwise_not(bgMask)

        foreground = cv2.bitwise_or(foreground, foreground, mask=fgMask)
        background = cv2.bitwise_or(image, image, mask=bgMask)

        res = cv2.bitwise_or(background, foreground)
        return res

    def Shutdown(self):
        self.preloader.shutdown()
