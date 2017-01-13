import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
#Apply the distortion correction to the raw image.
#read the camera calibration result
with open('dist_pickle.p', mode='rb') as f:
    dist_pickle = pickle.load(f)
'''
img = cv2.imread('test_images/test_image.png')
img = imresize(img, (720, 1280, 3))
dst = cv2.undistort(img, dist_pickle["mtx"], dist_pickle["dist"], None, dist_pickle["mtx"])

dst_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.imshow(dst_image)
'''
#Use color transforms, gradients, etc., to create a thresholded binary image.

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

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magnitude = (sobelx ** 2 + sobely ** 2) ** 0.5
    # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled = np.uint8(255 * magnitude / np.max(magnitude))
    # 6) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gray)
    binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    # 7) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(gray)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image

    return binary_output

def s_threshold(img, thresh = (170, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary

def s_threshold(img, thresh = (170, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary

def r_threshold(img, thresh = (170, 255)):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_channel = rgb[:, :, 0]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= thresh[0]) & (r_channel <= thresh[1])] = 1
    return r_binary

def thresholding(img):
    dst = cv2.undistort(img, dist_pickle["mtx"], dist_pickle["dist"], None, dist_pickle["mtx"])
    s_binary = s_threshold(dst, thresh=(170, 255))
    grad_binary = abs_sobel_thresh(dst, orient='x', thresh_min=20, thresh_max=100)
    combined = np.zeros_like(s_binary)
    combined[(s_binary == 1) | grad_binary == 1] = 1
    return combined

'''
# Run the function
r_binary = r_threshold(dst, thresh=(205, 255))
#plt.imshow(r_binary, cmap='gray')

# Run the function
s_binary = s_threshold(img, thresh = (170, 255))
#plt.imshow(s_binary, cmap='gray')

# Run the function
dir_binary = dir_threshold(dst, sobel_kernel=15, thresh=(0.7, 1.3))
#plt.imshow(dir_binary, cmap='gray')


# Run the function
mag_binary = mag_thresh(dst, sobel_kernel=5, mag_thresh=(20, 100))
#plt.imshow(mag_binary, cmap='gray')

# Run the function
grad_binary = abs_sobel_thresh(dst, orient='x', thresh_min=20, thresh_max=100)
#plt.imshow(grad_binary, cmap='gray')

dst_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 9))
f.tight_layout()

ax1.imshow(dst_image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('sobelX', fontsize=40)

ax3.imshow(mag_binary, cmap='gray')
ax3.set_title('mag_binary', fontsize=40)

ax4.imshow(dir_binary, cmap='gray')
ax4.set_title('dir_binary', fontsize=40)

ax5.imshow(s_binary, cmap='gray')
ax5.set_title('s_binary', fontsize=40)

ax6.imshow(r_binary, cmap='gray')
ax6.set_title('r_binary', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#combine threshold
combined = np.zeros_like(dir_binary)
combined[(s_binary == 1)  | grad_binary == 1] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(dst_image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(combined, cmap='gray')
ax2.set_title('combined', fontsize=40)
'''