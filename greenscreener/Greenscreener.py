import cv2
import os

from typing import List, Tuple

import greenscreener.config as config
import numpy as np

class Greenscreener:

    def __init__(self, backgroundDir: str = config.backgroundDir,
                 backgroundScale: Tuple[int, int] = config.backgroundScale):
        self.backgrounds: List[np.array] = []
        self.originalFileNames: List[str] = []
        self.backgourndDir: str = backgroundDir
        self.backgroundScale = backgroundScale

        self.Initialize()

    def LoadBackgrounds(self):
        fileList = os.listdir(config.backgroundDir)
        for file in fileList:
            image = cv2.imread(config.backgroundDir + file)
            image = cv2.resize(src=image, dsize=self.backgroundScale)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.backgrounds.append(image)
        return

    def LoadFilenames(self):
        if os.path.isdir(self.backgourndDir):
            self.originalFileNames = os.listdir(self.backgourndDir)

    def ResizeBackgrounds(self):
        size = config.backgroundScale

        for i in range(0, self.backgrounds.__len__()):
            self.backgrounds[i] = cv2.resize(src=self.backgrounds[i], dsize=size)

        return

    def GetRandomBackground(self):
        randInt = np.random.randint(self.backgrounds.__len__(), size=1)
        return self.backgrounds[int(randInt)]

    def ReplaceBackground(self, image: np.array):

        hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        background = self.GetRandomBackground()

        bgMask = cv2.inRange(hsvImage, config.lowerTH, config.upperTH)
        fgMask = cv2.bitwise_not(bgMask)

        foreground = cv2.bitwise_or(image, image, mask=fgMask)
        background = cv2.bitwise_or(background, background, mask=bgMask)

        res = cv2.bitwise_or(background, foreground)

        return res

    def Initialize(self):
        self.LoadBackgrounds()
        self.LoadFilenames()
        self.ResizeBackgrounds()
