import cv2
import numpy as np
from matplotlib import pyplot as plt

def binarization(source,threshold):
    plt.title("Image Histogram")
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.plot(cv2.calcHist([source], [0], None, [256], [0, 256]))
    plt.xlim([0, 256])
    plt.show()

    _, binaryImage = cv2.threshold(source, threshold, 255, cv2.THRESH_BINARY)

    return binaryImage

def erosionAndDilation():
    binaryImage = binarization(imageFetus,35)
    kernel = np.ones((3,3), np.uint8)
    erosionImage = cv2.erode(binaryImage, kernel)
    dilationImage = cv2.dilate(binaryImage,kernel)

    cv2.imshow("Binary Image with Erosion", erosionImage)
    cv2.imshow("Binary Image with Dilation", dilationImage)
    cv2.waitKey(0)

def openingAndClosing():
    binaryImage = binarization(imageFetus,20)
    kernel = np.ones((3,3), np.uint8)
    openingImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)
    closingImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE,kernel)

    cv2.imshow("Binary Image with Opening", openingImage)
    cv2.imshow("Binary Image with Closing",closingImage)
    cv2.waitKey(0)

#imageHead = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0359(a)(headCT_Vandy).tif")
#imageChest = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0107(a)(chest-xray-vandy).tif")
imageFetus = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0120(a)(ultrasound-fetus1).tif")
#imageBreast = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0304(a)(breast_digital_Xray).tif")

erosionAndDilation()
openingAndClosing()