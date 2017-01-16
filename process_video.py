from moviepy.editor import VideoFileClip
from IPython.display import HTML
from lane_finding import get_curvature, vehicle_position, lane_finding, get_polyfit, draw_back
from perspective_transform import compute_M_Minv
from thresholding import thresholding, thresholding2
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2
from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        #polynomial coefficients for the most recent fit
        self.detected = False
        self.recent_leftrange = [0, 0]
        self.recent_rightrange = [0, 0]
        self.left_fit = None
        self.right_fit = None
'''
class Line():
    def __init__(self):
        #polynomial coefficients for the most recent fit
        self.init = False
        self.recent_leftfit = deque(maxlen = 4)
        self.recent_rightfit = deque(maxlen = 4)
        self.last_fit = None

    def update_recent(self,left_fitx, right_fitx, yvals, left_fit, right_fit):
        self.init = True
        self.last_fit = [left_fitx, right_fitx, yvals]
        self.recent_leftfit.append(left_fit)
        self.recent_rightfit.append(right_fit)

    def judge(self, left_fitx, right_fitx, yvals, left_fit, right_fit):
        if self.judge_A_left(left_fit[0],left_fit[1],left_fit[2],self.recent_leftfit) and self.judge_A_left(right_fit[0], right_fit[1], right_fit[2], self.recent_rightfit):
            return True
        return False

    def judge_A_left(self, left_fit_A, left_fit_B, left_fit_C, recent_leftfit):
        mean_A = 0
        mean_B = 0
        mean_C = 0

        for i in recent_leftfit:
            mean_A += i[0]
            mean_B += i[1]
            mean_C += i[2]
        length = len(recent_leftfit)
        mean_A /= length
        mean_B /= length
        mean_C /= length

        if abs(left_fit_A - mean_A) > 0.0015 or abs(left_fit_B - mean_B) > 0.3 or abs(left_fit_C - mean_C) > 50:
            return False
        return True
'''

line = Line()
count1 = 0
count2 = 0
count = 0
M, Minv = compute_M_Minv()


def process_image(img):

    global line, count1, count2, count, M, Minv
  #  if count < 1105:
  #      count += 1
  #      return img
    count += 1
    if line.detected == True:
        #there is line detected in the last frame
        warped = thresholding2(img, line.recent_leftrange, line.recent_rightrange)
        leftx, lefty, rightx, righty = lane_finding(warped)
        yvals = np.linspace(0, 100, num=101) * 7.2
        left_fitx, left_fit = get_polyfit(leftx, lefty, yvals, line.left_fit)
        right_fitx, right_fit = get_polyfit(rightx, righty, yvals, line.right_fit)
        if 0.85 < (left_fitx[-1] - right_fitx[-1])/((left_fitx[1] - right_fitx[1])) < 1.15:
            pass
        else:
            warped = thresholding(img)
            leftx, lefty, rightx, righty = lane_finding(warped)
            yvals = np.linspace(0, 100, num=101) * 7.2
            left_fitx, left_fit = get_polyfit(leftx, lefty, yvals, line.left_fit)
            right_fitx, right_fit = get_polyfit(rightx, righty, yvals, line.right_fit)

    else:
        warped = thresholding(img)
        leftx, lefty, rightx, righty = lane_finding(warped)
        yvals = np.linspace(0, 100, num=101) * 7.2
        left_fitx, left_fit = get_polyfit(leftx, lefty, yvals)
        right_fitx, right_fit = get_polyfit(rightx, righty, yvals)
        line.detected = True

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


    line.recent_leftrange = [left_fitx.min(), left_fitx.max()]
    line.recent_rightrange = [right_fitx.min(), right_fitx.max()]
    line.left_fit = left_fit
    line.right_fit = right_fit

    left_curverad, right_curverad = get_curvature(left_fitx, right_fitx, yvals)
    ve_position = vehicle_position(left_fitx, right_fitx)
    ve_position = 'Lane deviation: ' + str(ve_position) + ' m.'

    str_curv = 'Curvature: Right = ' + str(left_curverad) + ', Left = ' + str(right_curverad)
    font = cv2.FONT_HERSHEY_COMPLEX
    result = draw_back(warped, img, left_fitx, right_fitx, yvals, Minv)
    cv2.putText(result, str_curv, (30, 60), font, 1, (0, 255, 0), 2)
    cv2.putText(result, ve_position, (30, 90), font, 1, (0, 255, 0), 2)
    #  if count >1105:
   #     aaa=1
    return result
'''
img = cv2.imread('test_images/test4.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = process_image(img)
plt.imshow(img2)
'''
'''from PIL import Image
img2 = Image.fromarray(img, 'RGB')
img2.save('test2.jpg')
'''

white_output = 'test15.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


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