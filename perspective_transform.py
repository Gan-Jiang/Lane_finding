import numpy as np
import cv2
from thresholding import thresholding
import matplotlib.pyplot as plt


def pt(img):
    #src points x,y 280,720   1250, 720 590, 450  710, 450
    src = np.float32([[595, 460], [715, 460], [280, 720], [1260, 720]])

    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[350 ,0] ,[950 ,0] ,[350 ,720] ,[950 ,720]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view

    dst_img = thresholding(img)
    img_size = (dst_img.shape[1], dst_img.shape[0])
    warped = cv2.warpPerspective(dst_img, M, img_size, flags = cv2.INTER_LINEAR)
    return warped