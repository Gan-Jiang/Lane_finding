import pickle
import cv2
import numpy as np
from perspective_transform import pt,compute_M_Minv
import matplotlib.image as mpimg
#from lane_finding import lane_finding,draw_back, get_polyfit, get_curvature, vehicle_position
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


def abs_sobel_thresh(img, thresh_min=0, thresh_max=255):
    '''
    This is used to apply x gradient.
    :param img:
    :param thresh_min:
    :param thresh_max:
    :return:
    '''
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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


def color_mask(img):
    '''
    This is used to convert the color space to HSV, and pick yellow lane and white lane according to the color range.
    :param img:
    :return:
    '''

    dst = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    yellow_hsv_low = np.array([0, 80, 0])
    yellow_hsv_high = np.array([80, 255, 255])

    white_hsv_low = np.array([20, 0, 180])
    white_hsv_high = np.array([255, 80, 255])

    mask_yellow = cv2.inRange(dst, yellow_hsv_low, yellow_hsv_high)
    mask_white = cv2.inRange(dst, white_hsv_low, white_hsv_high)
    return mask_yellow+mask_white


def thresholding(img, M):
    '''
    This is used to combine X gradient and color selection.
    :param img:
    :return:
    '''
    #undistort
    dst = cv2.undistort(img, dist_pickle["mtx"], dist_pickle["dist"], None, dist_pickle["mtx"])

    #mpimg.imsave("output_images/undistort/test6.png", dst)

    #apply region mask
    imshape = dst.shape
    vertices = np.array([[(50, imshape[0]), (0.472 * imshape[1], 400),
                          (0.528 * imshape[1], 400),
                          (imshape[1] - 50, imshape[0])]], dtype=np.int32)
    region = region_of_interest(dst, vertices)

    #apply perspective transform
    region = pt(region, M)

    #apply color selection
    color = color_mask(region)

    #apply x gradient.
    grad_binary = abs_sobel_thresh(region, thresh_min=10, thresh_max=50)

    #use hough transform to clean the pixels found by grad_binary
    # Define the Hough transform parameters
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 0  # maximum gap in pixels between connectable line segments

    line_image = np.copy(color) * 0  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(grad_binary, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), 255, thickness=1)

    #combine x sobel and color selection
    combined = np.zeros_like(color)
    combined[(color == 255) | (line_image == 255)] = 1

    #mpimg.imsave("output_images/threshold/test6.png", combined, cmap='gray')


    return combined


def thresholding2(img, leftrange, rightrange, M):
    '''
    This is used to search lane line in the range found by the last frame.
    :param img:
    :param leftrange: the max and min of the left lane in last frame
    :param rightrange: the max and min of the right lane in last frame
    :param M:
    :return:
    '''
    dst = cv2.undistort(img, dist_pickle["mtx"], dist_pickle["dist"], None, dist_pickle["mtx"])
    region = pt(dst, M)
    #length: half width range
    length = 30

    color1 = color_mask(region[:, leftrange[0]-length:leftrange[1]+length, :])
    color2 = color_mask(region[:, rightrange[0]-length:rightrange[1]+length, :])

    grad_binary1 = abs_sobel_thresh(region[:, leftrange[0]-length:leftrange[1]+length, :],  thresh_min=10, thresh_max=50)
    grad_binary2 = abs_sobel_thresh(region[:, rightrange[0]-length:rightrange[1]+length, :], thresh_min=10, thresh_max=50)


    # Define the Hough transform parameters
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 0  # maximum gap in pixels between connectable line segments

    line_image1 = np.copy(grad_binary1) * 0  # creating a blank to draw lines on
    line_image2 = np.copy(grad_binary2) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    lines1 = cv2.HoughLinesP(grad_binary1, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    lines2= cv2.HoughLinesP(grad_binary2, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    try:
        for line in lines1:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image1, (x1, y1), (x2, y2), 255, thickness=1)
    except:
        pass
    try:
        for line in lines2:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image2, (x1, y1), (x2, y2), 255, thickness=1)
    except:
        pass

    combined = np.zeros([720, 1280])
    combined[:, leftrange[0]-length:leftrange[1]+length] = color1 + line_image1
    combined[:, rightrange[0]-length:rightrange[1]+length] = color2 + line_image2
    combined[(combined >= 255)] = 1
    return combined

'''
M, Minv = compute_M_Minv()
img = cv2.imread('test_images/test6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
warped = thresholding(img, M)
leftx, lefty, rightx, righty = lane_finding(warped)

yvals = np.linspace(0, 100, num=101) * 7.2

left_fitx, left_fit = get_polyfit(leftx, lefty, yvals)
right_fitx, right_fit = get_polyfit(rightx, righty,yvals)
left_curverad, right_curverad = get_curvature(left_fitx, right_fitx, yvals)
ve_position = vehicle_position(left_fitx, right_fitx)
ve_position = 'Lane deviation: ' + str(np.around(ve_position, 2)) + ' m.'

str_curv = 'Curvature: Left = ' + str(np.around(left_curverad, 2)) + ', Right = ' + str(np.around(right_curverad, 2))
font = cv2.FONT_HERSHEY_COMPLEX
result = draw_back(warped, img, left_fitx, right_fitx, yvals, Minv)
cv2.putText(result, str_curv, (30, 60), font, 1, (0, 255, 0), 2)
cv2.putText(result, ve_position, (30, 90), font, 1, (0, 255, 0), 2)
mpimg.imsave("output_images/final/test6.png", result)
'''