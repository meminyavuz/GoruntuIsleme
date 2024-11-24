import cv2
import numpy as np
from matplotlib import pyplot as plt

def multiThreshold(image, t1, t2, t3):
    newImage = np.zeros_like(image)
    newImage[image < t1] = 0           
    newImage[(image >= t1) & (image < t2)] = 120  
    newImage[(image >= t2) & (image < t3)] = 200  
    newImage[image >= t3] = 255         

    plt.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
    plt.axvline(t1, color='black', linestyle='-', label=f'Thresh 1 = {t1}')
    plt.axvline(t2, color='black', linestyle='-', label=f'Thresh 2 = {t2}')
    plt.axvline(t3, color='black', linestyle='-', label=f'Thresh 3 = {t3}')
    plt.title("Histogram ve Çoklu Eşikler")
    plt.xlabel("Piksel Değeri")
    plt.ylabel("Frekans")
    plt.show()

    return newImage

#imageHead = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0359(a)(headCT_Vandy).tif")
#imageChest = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0107(a)(chest-xray-vandy).tif")
#imageFetus = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0120(a)(ultrasound-fetus1).tif")
imageBreast = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0304(a)(breast_digital_Xray).tif")

cv2.imshow("Original Image", imageBreast)
cv2.imshow("Multi-Threshold Image", multiThreshold(imageBreast,35,70,140))
cv2.waitKey(0)
cv2.destroyAllWindows()
