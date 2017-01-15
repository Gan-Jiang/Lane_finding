from moviepy.editor import VideoFileClip
from IPython.display import HTML
from lane_finding import get_curvature, vehicle_position, lane_finding, get_polyfit, draw_back
from perspective_transform import compute_M_Minv
from thresholding import thresholding
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2
from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        #polynomial coefficients for the most recent fit
        self.init = False
        self.recent_leftfit = deque(maxlen = 3)
        self.recent_rightfit = deque(maxlen = 3)
        self.last_fit = None

    def update_recent(self,left_fitx, right_fitx, yvals, left_fit, right_fit):
        self.init = True
        self.last_fit = [left_fitx, right_fitx, yvals]
        self.recent_leftfit.append(left_fit)
        self.recent_rightfit.append(right_fit)

    def judge(self, left_fitx, right_fitx, yvals, left_fit, right_fit):
        if self.judge_A_left(left_fit[0],left_fit[1], self.recent_leftfit) and self.judge_A_left(right_fit[0], right_fit[1], self.recent_rightfit):
            return True
        return False

    def judge_A_left(self, left_fit_A, left_fit_B, recent_leftfit):
        mean_A = 0
        mean_B = 0
        for i in recent_leftfit:
            mean_A += i[0]
            mean_B += i[1]

        mean_A /= recent_leftfit.maxlen
        mean_B /= recent_leftfit.maxlen

        if abs(left_fit_A - mean_A) > 0.0015 or abs(left_fit_B - mean_B) > 0.3:
            return False
        return True


line = Line()
count1 = 0
count2 = 0
count = 0
def process_image(img):
    global line, count1, count2, count
    count += 1
    M, Minv = compute_M_Minv()
    warped = thresholding(img)
    leftx, lefty, rightx, righty = lane_finding(warped)
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

    try:
        left_fitx, right_fitx, yvals, left_fit, right_fit = get_polyfit(leftx, lefty, rightx, righty, order = 2)
        if line.init:
            if line.judge(left_fitx, right_fitx, yvals, left_fit, right_fit):
                line.update_recent(left_fitx, right_fitx, yvals, left_fit, right_fit)
            else:
                count1 += 1
                #too large difference
                left_fitx, right_fitx, yvals = line.last_fit
        else:
            line.update_recent(left_fitx, right_fitx, yvals, left_fit, right_fit)
    except:
        if line.init:
            count2 += 1
            left_fitx, right_fitx, yvals = line.last_fit

    #left_curverad, right_curverad = get_curvature(left_fitx, right_fitx, yvals)
    #ve_position = vehicle_position(left_fitx)
    #if count >525:
    #    aaa=1
    return draw_back(warped, img, left_fitx, right_fitx, yvals, Minv)

img = cv2.imread('test_images/test15.jpg')
img2 = process_image(img)
plt.imshow(img2)

'''from PIL import Image
img2 = Image.fromarray(img, 'RGB')
img2.save('test2.jpg')
'''
'''
white_output = 'test8.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
'''
'''
white_output = 'test4.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

white_output = 'test5.mp4'
clip1 = VideoFileClip("harder_challenge_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
'''