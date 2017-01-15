import numpy as np
import matplotlib.pyplot as plt
import cv2
from perspective_transform import compute_M_Minv
from thresholding import thresholding
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
        try:
            if histogram_left.max() > 5:
                    peak_left = np.argmax(histogram_left) + peak_left - window_width_half
        except:
            pass

        histogram_right = np.sum(warped[start:offset, peak_right - window_width_half:peak_right + window_width_half + 1], axis=0)
        try:
            if histogram_right.max() > 5:
                peak_right = np.argmax(histogram_right) + peak_right - window_width_half
        except:
            pass
    return np.array(leftx),np.array(lefty),np.array(rightx),np.array(righty)
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


def get_curvature(left_fitx, right_fitx, yvals):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    y_eval = 720
    left_fit_cr = np.polyfit(yvals * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals * ym_per_pix, right_fitx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad

def get_polyfit(leftx, lefty, rightx, righty, order = 2):
    left_fit = np.polyfit(lefty, leftx, order)
    yvals = np.linspace(0, 100, num=101) * 7.2
    left_fitx = left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2]

    right_fit = np.polyfit(righty, rightx, order)
    right_fitx = right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2]
    '''
    plt.plot(leftx, lefty, 'o', color='red')
    plt.plot(rightx, righty, 'o', color='blue')
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, yvals, color='green', linewidth=3)
    plt.plot(right_fitx, yvals, color='green', linewidth=3)
    plt.gca().invert_yaxis()  # to visualize as we do the images
    '''
    return left_fitx, right_fitx, yvals, left_fit, right_fit

def vehicle_position(left_fitx):
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    return (1280//2 - left_fitx[-1]) * xm_per_pix




def draw_back(warped, img, left_fitx, right_fitx, yvals, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result


'''
img = cv2.imread('test_images/test8.jpg')
img2 = process_image(img)
plt.imshow(img2)
'''