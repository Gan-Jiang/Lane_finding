from moviepy.editor import VideoFileClip
from lane_finding import get_curvature, vehicle_position, lane_finding, get_polyfit, draw_back
from perspective_transform import compute_M_Minv
from thresholding import thresholding, thresholding2
import numpy as np
import cv2

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        #last frame detected
        self.detected = False

        #line range in last frame
        self.recent_leftrange = [0, 0]
        self.recent_rightrange = [0, 0]

        self.left_fit = None
        self.right_fit = None

        self.left_fit2 = None
        self.right_fit2 = None

    def update_fit(self, leftfit, rightfit):
        self.left_fit2 = self.left_fit
        self.right_fit2 = self.right_fit
        self.left_fit = leftfit
        self.right_fit = rightfit
line = Line()

M, Minv = compute_M_Minv()
yvals = np.linspace(0, 100, num=101) * 7.2


def process_image(img):

    global line, count1, count2, count, M, Minv, yvals

    if line.detected == True:
        #there is line detected in the last frame
        warped = thresholding2(img, line.recent_leftrange, line.recent_rightrange, M)
        leftx, lefty, rightx, righty = lane_finding(warped)
        left_fitx, left_fit = get_polyfit(leftx, lefty, yvals, line.left_fit, line.left_fit2)
        right_fitx, right_fit = get_polyfit(rightx, righty, yvals, line.right_fit, line.right_fit2)
        if 0.85 < (left_fitx[-1] - right_fitx[-1])/((left_fitx[1] - right_fitx[1])) < 1.15:
            pass
        else:
            warped = thresholding(img, M)
            leftx, lefty, rightx, righty = lane_finding(warped)
            left_fitx, left_fit = get_polyfit(leftx, lefty, yvals, line.left_fit, line.left_fit2)
            right_fitx, right_fit = get_polyfit(rightx, righty, yvals, line.right_fit, line.right_fit2)

    else:
        warped = thresholding(img, M)
        leftx, lefty, rightx, righty = lane_finding(warped)
        left_fitx, left_fit = get_polyfit(leftx, lefty, yvals, line.left_fit, line.left_fit2)
        right_fitx, right_fit = get_polyfit(rightx, righty, yvals, line.right_fit, line.right_fit2)
        line.detected = True

    line.recent_leftrange = [left_fitx.min(), left_fitx.max()]
    line.recent_rightrange = [right_fitx.min(), right_fitx.max()]

    line.update_fit(left_fit,right_fit)

    left_curverad, right_curverad = get_curvature(left_fitx, right_fitx, yvals)
    ve_position = vehicle_position(left_fitx, right_fitx)
    ve_position = 'Lane deviation: ' + str(np.around(ve_position, 2)) + ' m.'

    str_curv = 'Curvature: Left = ' + str(np.around(left_curverad, 2)) + ', Right = ' + str(np.around(right_curverad, 2))
    font = cv2.FONT_HERSHEY_COMPLEX
    result = draw_back(warped, img, left_fitx, right_fitx, yvals, Minv)
    cv2.putText(result, str_curv, (30, 60), font, 1, (0, 255, 0), 2)
    cv2.putText(result, ve_position, (30, 90), font, 1, (0, 255, 0), 2)
    return result

white_output = 'output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)