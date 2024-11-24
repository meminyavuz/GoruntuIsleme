import cv2
import numpy as np

def regionGrowing(image, sp, thresholds):

    t1, t2, t3 = thresholds
    h, w = image.shape
    segmented = np.zeros((h, w, 3), dtype=np.uint8)

    seed_value = image[sp]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
    stack = [sp]

    while stack:
        x, y = stack.pop()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and np.all(segmented[nx, ny] == 0):
                pixel_value = image[nx, ny]
                diff = abs(pixel_value - seed_value)
                if diff <= t1:
                    segmented[nx, ny] = colors[0]
                    stack.append((nx, ny))
                elif diff <= t2:
                    segmented[nx, ny] = colors[1]
                    stack.append((nx, ny))
                elif diff <= t3:
                    segmented[nx, ny] = colors[2]
                    stack.append((nx, ny))

    return segmented

imageHead = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0359(a)(headCT_Vandy).tif",cv2.IMREAD_GRAYSCALE)
#imageChest = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0107(a)(chest-xray-vandy).tif",cv2.IMREAD_GRAYSCALE)
#imageFetus = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0120(a)(ultrasound-fetus1).tif",cv2.IMREAD_GRAYSCALE)
#imageBreast = cv2.imread("/Users/meyavuz/Desktop/goruntu isleme/Fig0304(a)(breast_digital_Xray).tif",cv2.IMREAD_GRAYSCALE)

sp = (100, 100)  
thresholds = (50, 100, 220)  
segmented_result = regionGrowing(imageHead, sp, thresholds)

cv2.imshow("Original Image", imageHead)
cv2.imshow("Segmented Image", segmented_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
