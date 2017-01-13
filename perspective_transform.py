import numpy as np
import cv2
from thresholding import thresholding
import matplotlib.pyplot as plt

#src points x,y 280,720   1250, 720 590, 450  710, 450
src = np.float32([[595, 460], [715, 460], [280, 720], [1260, 720]])

# c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
dst = np.float32([[350 ,0] ,[950 ,0] ,[350 ,720] ,[950 ,720]])
# d) use cv2.getPerspectiveTransform() to get M, the transform matrix
M = cv2.getPerspectiveTransform(src, dst)
# e) use cv2.warpPerspective() to warp your image to a top-down view
img = cv2.imread('test_images/test2.jpg')

dst_img = thresholding(img)
img_size = (dst_img.shape[1], dst_img.shape[0])
warped = cv2.warpPerspective(dst_img, M, img_size, flags = cv2.INTER_LINEAR)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(dst_img, cmap = 'gray')
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(warped, cmap='gray')
ax2.set_title('warped', fontsize=40)
'''
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
plt.plot(histogram)
'''
