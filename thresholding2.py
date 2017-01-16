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
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)

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

def h_threshold(img, thresh = (15, 100)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= thresh[0]) & (h_channel <= thresh[1])] = 1
    return h_binary

def r_threshold(img, thresh = (170, 255)):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_channel = rgb[:, :, 0]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= thresh[0]) & (r_channel <= thresh[1])] = 1
    return r_binary

def thresholding(img):
    dst = cv2.undistort(img, dist_pickle["mtx"], dist_pickle["dist"], None, dist_pickle["mtx"])
    imshape = dst.shape
    s_binary = s_threshold(dst, thresh=(170, 255))
    grad_binary = abs_sobel_thresh(dst, orient='x', thresh_min=20, thresh_max=100)
    combined = np.zeros_like(s_binary)
    combined[(s_binary == 1) | grad_binary == 1] = 1
    vertices = np.array([[(100, imshape[0]), (0.475 * imshape[1], 400),
                          (0.525 * imshape[1], 400),
                          (imshape[1] - 100, imshape[0])]], dtype=np.int32)
    combined = region_of_interest(combined, vertices)
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


img = cv2.imread('test_images/test21.jpg')
img = imresize(img, (720, 1280, 3))
dst = cv2.undistort(img, dist_pickle["mtx"], dist_pickle["dist"], None, dist_pickle["mtx"])

dst_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
img_size = (dst_image.shape[1], dst_image.shape[0])
warped = cv2.warpPerspective(dst_image, M, img_size, flags=cv2.INTER_LINEAR)
warped2 = cv2.warpPerspective(dst, M, img_size, flags=cv2.INTER_LINEAR)
img = warped2
dst = warped
combined = thresholding(img)
color = color_mask(img)

'''
color = color_mask(dst)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(color, cmap = 'gray')
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(combined, cmap='gray')
ax2.set_title('combined', fontsize=40)

'''
# Run the function
r_binary = r_threshold(dst, thresh=(205, 255))
#plt.imshow(r_binary, cmap='gray')

# Run the function
s_binary = s_threshold(img, thresh = (170, 255))
#plt.imshow(s_binary, cmap='gray')

h_binary = s_threshold(img, thresh = (15, 100))

# Run the function
dir_binary = dir_threshold(dst, sobel_kernel=15, thresh=(0.7, 1.3))
imshape = dir_binary.shape

'''
imshape = dir_binary.shape
vertices = np.array([[(100, imshape[0]), (0.475 * imshape[1], 400),
                      (0.525 * imshape[1], 400),
                      (imshape[1] - 100, imshape[0])]], dtype=np.int32)
dir_binary = region_of_interest(dir_binary, vertices)

#plt.imshow(dir_binary, cmap='gray')

histogram = np.sum(dir_binary[dir_binary.shape[0] // 2:, :], axis=0)
plt.plot(histogram)
'''
# Run the function
mag_binary = mag_thresh(dst, sobel_kernel=5, mag_thresh=(30, 50))
#plt.imshow(mag_binary, cmap='gray')

# Run the function
grad_binary = abs_sobel_thresh(dst, orient='x', thresh_min=10, thresh_max=50)
#plt.imshow(grad_binary, cmap='gray')


# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 30  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40  # minimum number of pixels making up a line
max_line_gap = 0  # maximum gap in pixels between connectable line segments

line_image = np.copy(dst_image) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(grad_binary, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)


def draw_lines(img, lines, imshape, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    left_line_slope=[]
    right_line_slope=[]
    left_line_x=[]
    left_line_y=[]
    right_line_x=[]
    right_line_y=[]
    left_mark=[]
    right_mark=[]

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2,y2), color, thickness)



def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

draw_lines(line_image, lines, imshape, color=[255, 0, 0], thickness=8)
# Create a "color" binary image to combine with line image
color_edges = np.dstack((mag_binary, mag_binary, mag_binary))

# Draw the lines on the edge image
lines_edges=weighted_img(warped, line_image, α=0.8, β=1., λ=0.)


f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 9))
f.tight_layout()

ax1.imshow(warped)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(lines_edges, cmap='gray')
ax2.set_title('lines', fontsize=40)

ax3.imshow(mag_binary, cmap='gray')
ax3.set_title('mag_binary', fontsize=40)

ax4.imshow(color, cmap='gray')
ax4.set_title('color', fontsize=40)

ax5.imshow(s_binary, cmap='gray')
ax5.set_title('s_binary', fontsize=40)

ax6.imshow(grad_binary, cmap='gray')
ax6.set_title('grad_binary', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#combine threshold
combined = np.zeros_like(dir_binary)
combined[(s_binary == 1)  | grad_binary == 1] = 1

'''
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(dst_image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(combined, cmap='gray')
ax2.set_title('combined', fontsize=40)
'''