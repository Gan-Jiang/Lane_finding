import numpy as np
import cv2

def compute_M_Minv():
    '''
    This is used to compute M and Minv
    :return:
    '''
    # src points x,y 280,720   1250, 720 590, 450  710, 450
    src = np.float32([[595, 460], [715, 460], [280, 720], [1260, 720]])

    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[350, 0], [950, 0], [350, 720], [950, 720]])

    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def pt(img, M):
    '''
    This is used to apply perspective transform
    :param img:
    :param M:
    :return:
    '''
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)
    return warped
