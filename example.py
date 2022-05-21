from greenscreener.ImageGreenscreener import ImageGreenscreener
import cv2

screener = ImageGreenscreener(backgroundDir="path/to/background/directory",
                              backgroundScale=(1280, 720))

image = cv2.imread("path/to/image")
image = cv2.resize(src=image, dsize=(1280, 720))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = screener.ReplaceBackground(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imshow("Greenscreener Demo", image)

cv2.waitKey(0)
