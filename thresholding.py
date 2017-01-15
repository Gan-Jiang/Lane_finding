import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from perspective_transform import pt,compute_M_Minv
from lane_finding import lane_finding, get_polyfit
#Apply the distortion correction to the raw image.
#read the camera calibration result
with open('dist_pickle.p', mode='rb') as f:
    dist_pickle = pickle.load(f)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 1

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobelx/np.max(abs_sobelx))
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def thresholding(img):
    dst = cv2.undistort(img, dist_pickle["mtx"], dist_pickle["dist"], None, dist_pickle["mtx"])
    imshape = dst.shape
    vertices = np.array([[(100, imshape[0]), (0.475 * imshape[1], 400),
                          (0.525 * imshape[1], 400),
                          (imshape[1] - 100, imshape[0])]], dtype=np.int32)
    region = region_of_interest(dst, vertices)
    M, Minv = compute_M_Minv()
    region = pt(region, M)
    color = color_mask(region)

    grad_binary = abs_sobel_thresh(region, orient='x', thresh_min=10, thresh_max=50)

    combined = np.zeros_like(color)
    combined[(color == 255) | grad_binary == 1] = 1
    vertices2 = np.array([[(0.23*imshape[1], imshape[0]), (0.08 * imshape[1], 0),
                           (0.82 * imshape[1], 0),
                           (0.7035*imshape[1], imshape[0])]], dtype=np.int32)
    combined = region_of_interest(combined, vertices2)
    vertices2 = np.array([[(350, imshape[0]), (599, 0),
                           (600, 0),
                           (850, imshape[0])]], dtype=np.int32)
    combined2 = region_of_interest(combined, vertices2)
    combined = combined - combined2
    return combined


def color_mask(img):
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_hsv_low = np.array([0, 80, 0])
    yellow_hsv_high = np.array([80, 255, 255])

    white_hsv_low = np.array([20, 0, 180])
    white_hsv_high = np.array([255, 80, 255])
    mask_yellow = cv2.inRange(dst, yellow_hsv_low, yellow_hsv_high)

    mask_white = cv2.inRange(dst, white_hsv_low, white_hsv_high)
    return mask_yellow+mask_white

img = cv2.imread('test_images/test15.jpg')
img = imresize(img, (720, 1280, 3))
M, Minv = compute_M_Minv()
warped = thresholding(img)
img_size = (img.shape[1], img.shape[0])

img2 = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

leftx, lefty, rightx, righty = lane_finding(warped)
left_fitx, right_fitx, yvals, left_fit, right_fit = get_polyfit(leftx, lefty, rightx, righty, order=2)

warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
result = cv2.addWeighted(img2, 1, color_warp, 0.3, 0)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(warped, cmap='gray')
ax2.set_title('combined', fontsize=40)

ax3.imshow(result)
ax3.set_title('poly', fontsize=40)
