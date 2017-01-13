import numpy as np
import matplotlib.pyplot as plt
import cv2
from perspective_transform import pt

def lane_finding(warped):
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
    return leftx,lefty,rightx,righty
    '''
    result = np.zeros_like(warped)
    result[lefty,leftx] = 1
    result[righty, rightx] = 1

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()


    ax1.imshow(warped, cmap = 'gray')
    ax1.set_title('warped', fontsize=40)

    ax2.imshow(result, cmap='gray')
    ax2.set_title('result', fontsize=40)
    '''


img = cv2.imread('test_images/test1.jpg')
warped = pt(img)
leftx, lefty, rightx, righty = lane_finding(warped)