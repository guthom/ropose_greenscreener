import cv2
import os

import greenscreener.config as config
import numpy as np

backgrounds = []
originalFileNames = []

def LoadBackgrounds():
    fileList = os.listdir(config.backgroundDir)
    for file in fileList:
        image = cv2.imread(config.backgroundDir + file)
        image = cv2.resize(src=image, dsize=config.backgroundScale)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        backgrounds.append(image)
    return


def LoadFilenames():
    if os.path.isdir(config.imageDir):
        originalFileNames = os.listdir(config.imageDir)


def ResizeBackgrounds():

    size = config.backgroundScale

    for i in range(0, backgrounds.__len__()):
        backgrounds[i] = cv2.resize(src=backgrounds[i], dsize=size)

    return


def ProcessFiles():
    if not os.path.exists(config.resultDir):
        os.mkdir(config.resultDir)

    imageCount = originalFileNames.__len__()
    processingNr = 0
    for file in originalFileNames:
        if file.__contains__(".png"):
            image = cv2.imread(config.imageDir + file)
            replacedImage = ReplaceBackground(image)
            cv2.imwrite(config.resultDir + file, replacedImage)
            image = None
            replacedImage = None
        processingNr += 1
        ShowSimpleProgress(imageCount, processingNr)
    return


def GetRandomBackground():
    randInt = np.random.randint(backgrounds.__len__(), size=1)
    return backgrounds[int(randInt)]


def ReplaceBackground(image):

    hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    background = GetRandomBackground()

    bgMask = cv2.inRange(hsvImage, config.lowerTH, config.upperTH)
    fgMask = cv2.bitwise_not(bgMask)

    foreground = cv2.bitwise_or(image, image, mask=fgMask)
    background = cv2.bitwise_or(background, background, mask=bgMask)

    res = cv2.bitwise_or(background, foreground)

    return res


def ShowSimpleProgress(imageCount, processingNr):
    infoString = str(processingNr) + "/" + str(imageCount)
    progress = round(float(processingNr / imageCount) * 100, 2)
    print(infoString + "[" + str(progress) + "%] ")
    return


def Initialize():
    LoadBackgrounds()
    LoadFilenames()
    ResizeBackgrounds()
