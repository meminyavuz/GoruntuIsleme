import cv2
from matplotlib import pyplot as plt

def show_hist(image):
    plt.title("Breast Image Histogram")
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
    plt.xlim([0, 256])
    plt.show()

imageHead = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0359(a)(headCT_Vandy).tif")
#imageChest = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0107(a)(chest-xray-vandy).tif")
#imageFetus = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0120(a)(ultrasound-fetus1).tif")
#imageBreast = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0304(a)(breast_digital_Xray).tif")

show_hist(imageHead)


