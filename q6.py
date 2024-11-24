import cv2
import numpy as np
from matplotlib import pyplot as plt

imageHead = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0359(a)(headCT_Vandy).tif",cv2.IMREAD_GRAYSCALE)
#imageChest = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0107(a)(chest-xray-vandy).tif",cv2.IMREAD_GRAYSCALE)
#imageFetus = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0120(a)(ultrasound-fetus1).tif",cv2.IMREAD_GRAYSCALE)
#imageBreast = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0304(a)(breast_digital_Xray).tif",cv2.IMREAD_GRAYSCALE)

equalizeHist = cv2.equalizeHist(imageHead)

def print(image):
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

print(imageHead)
print(equalizeHist)

threshold = 20 
_, binaryImage = cv2.threshold(equalizeHist, threshold, 255, cv2.THRESH_BINARY)

thresholds = [20, 100, 230]
segmentedImage = np.zeros_like(equalizeHist, dtype=np.uint8)

segmentedImage[equalizeHist <= thresholds[0]] = 64   
segmentedImage[(equalizeHist > thresholds[0]) & (equalizeHist <= thresholds[1])] = 128  
segmentedImage[(equalizeHist > thresholds[1]) & (equalizeHist <= thresholds[2])] = 192  
segmentedImage[equalizeHist > thresholds[2]] = 255  

kernel = np.ones((3, 3), np.uint8)

erosion = cv2.erode(binaryImage, kernel, iterations=1)
dilation = cv2.dilate(binaryImage, kernel, iterations=1)
opening = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)

def regionGrowing(image, sd, threshold):
    h, w = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    seed_value = image[sd]
    stack = [sd]

    while stack:
        x, y = stack.pop()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and segmented[nx, ny] == 0:
                diff = abs(int(image[nx, ny]) - int(seed_value))
                if diff <= threshold:
                    segmented[nx, ny] = 255
                    stack.append((nx, ny))

    return segmented

seed_point = (100, 100)  
regionImage = regionGrowing(equalizeHist, seed_point, threshold=20)

cv2.imshow("Original Image", imageHead)
cv2.imshow("Equalized Image", equalizeHist)
cv2.imshow("Binary Image", segmentedImage)
cv2.imshow("Segmented Image (3 Thresholds)", segmentedImage)
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)
cv2.imshow("Region Growing Result", regionImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
