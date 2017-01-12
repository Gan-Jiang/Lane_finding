import pickle
import cv2
import matplotlib.pyplot as plt
#Apply the distortion correction to the raw image.
#read the camera calibration result
with open('dist_pickle.p', mode='rb') as f:
    dist_pickle = pickle.load(f)

img = cv2.imread('test_images/solidWhiteRight.jpg')
dst = cv2.undistort(img, dist_pickle["mtx"], dist_pickle["dist"], None, dist_pickle["mtx"])
#dst_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
#plt.imshow(dst_image)