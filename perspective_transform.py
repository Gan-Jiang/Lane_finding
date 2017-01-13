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
img = cv2.imread('test_images/test6.jpg')

dst_img = thresholding(img)
img_size = (dst_img.shape[1], dst_img.shape[0])
warped = cv2.warpPerspective(dst_img, M, img_size, flags = cv2.INTER_LINEAR)

lefty, leftx, righty, rightx = [], [], [], []

window_height = 100
window_width_half = 45
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
#plt.plot(histogram)

peak_left = np.argmax(histogram[:warped.shape[1]//2])
peak_right = np.argmax(histogram[warped.shape[1]//2:]) + warped.shape[1]//2

for offset in range(warped.shape[0],-1, -window_height):
    if offset - window_height < 0:
        start = 0
        for x in range(peak_left - window_width_half, peak_left + window_width_half + 1):
            for y in range(offset):
                if warped[y,x] == 1:
                    leftx.append(x)
                    lefty.append(y)


        for x in range(peak_right - window_width_half, peak_right + window_width_half + 1):
            for y in range(offset):
                if warped[y,x] == 1:
                    rightx.append(x)
                    righty.append(y)
    else:
        start = offset - window_height
        for x in range(peak_left - window_width_half, peak_left + window_width_half + 1):
            for y in range(offset - window_height, offset):
                if warped[y, x] == 1:
                    leftx.append(x)
                    lefty.append(y)

        for x in range(peak_right - window_width_half, peak_right + window_width_half + 1):
            for y in range(offset - window_height, offset):
                if warped[y, x] == 1:
                    rightx.append(x)
                    righty.append(y)

    histogram_left = np.sum(warped[start:offset, peak_left - window_width_half:peak_left + window_width_half + 1], axis=0)
    if histogram_left.max() > 5:
        peak_left = np.argmax(histogram_left) + peak_left - window_width_half

    histogram_right = np.sum(warped[start:offset, peak_right - window_width_half:peak_right + window_width_half + 1], axis=0)
    if histogram_right.max() > 5:
        peak_right = np.argmax(histogram_right) + peak_right - window_width_half

result = np.zeros_like(warped)
result[lefty,leftx] = 1
result[righty, rightx] = 1

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()


ax1.imshow(dst_img, cmap = 'gray')
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(warped, cmap='gray')
ax2.set_title('warped', fontsize=40)

ax3.imshow(result, cmap='gray')
ax3.set_title('result', fontsize=40)
