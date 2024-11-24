import cv2
from matplotlib import pyplot as plt

def binarization(source,threshold):
    plt.title("Breast Image Histogram")
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.plot(cv2.calcHist([source], [0], None, [256], [0, 256]))
    plt.xlim([0, 256])
    plt.show()

    _, binaryImage = cv2.threshold(source, threshold, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("Original Image", source)
    cv2.imshow("Binary Image", binaryImage)
    cv2.waitKey(0)

#imageHead = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0359(a)(headCT_Vandy).tif")
#imageChest = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0107(a)(chest-xray-vandy).tif")
#imageFetus = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0120(a)(ultrasound-fetus1).tif")
imageBreast = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0304(a)(breast_digital_Xray).tif")

binarization(imageBreast,35)

