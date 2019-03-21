#!/home/shannon/miniconda2/envs/cvbot/bin/python
#-*- coding: utf-8 -*-　　　←表示使用 utf-8 編碼
import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('10_10.png',cv2.IMREAD_GRAYSCALE)
print type(img)
print img.shape
print img
img.reshape

# 讓視窗可以自由縮放大小
cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
cv2.imshow("10_10",img)
cv2.waitKey(0)
cv2.destoryAllWindows()

#cv2.imwrite('output.jpg', img)
#cv2.imwrite('output.tiff', img)

# 設定 JPEG 圖片品質為 90（可用值為 0 ~ 100）
#cv2.imwrite('output.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])

# 設定 PNG 壓縮層級為 5（可用值為 0 ~ 9）
#cv2.imwrite('output.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 5])



# 使用 OpenCV 讀取圖檔
img_bgr = cv2.imread('10_9.png')
# 將 BGR 圖片轉為 RGB 圖片
img_rgb = img_bgr[:,:,::-1]
# 或是這樣亦可
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# 使用 Matplotlib 顯示圖片
plt.imshow(img_rgb)
plt.show()


# 使用 OpenCV 讀取灰階圖檔
img_gray = cv2.imread('10_8.png', cv2.IMREAD_GRAYSCALE)
# 使用 Matplotlib 顯示圖片
plt.imshow(img_gray, cmap = 'gray')
plt.show()
