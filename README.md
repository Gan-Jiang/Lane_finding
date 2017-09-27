## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!


# What I did: 
Find and draw the lanes on the roads.

There are the following .py files:
(1) calibration.py: Compute the camera matrix and distortion coefficients.
(2) thresholding.py: Apply the following: (a). region selection (b). color selection (c). X-sobel threshold.
    threshoulding2() is used to find the lane lines according to the line centers found in the last frame.
(3) perspective_transform.py: Apply perspective_transform.
(4) lane_finding.py: Apply histogram method and sliding window to find the lane line pixels. The curvature and vehicle
    position are also found in this file.
(5) process_video.py: The pipeline to apply all the above functions to a video.

First, run calibration.py to compute the camera matrix and distortion coefficients which are stored in dist_pickle.p.

Then, for test images, I have conducted the following. The details can be referred to the corresponding function codes.
(1) Apply thresholding(img, M) in thresholding.py for each test image. M is the camera matrix.
    (a). This function will first undistort the image. All the images after undistorting are stored in
    'output_images/undistort/'
    (b). Then, a trapezoid shape region selection is applied.
    (c). After region selection, the function conducts the perspective transformation by running pt() fucntion. I apply the
    gradient threshold after perspective transformation because after perspective transformation, the lane line will be
    roughly vertical.
    (d). Apply a color mask by running color_mask(). In that function, the images are first transformed to HSV color
    space. Then, two color ranges are selected representing yellow lines and white lines.
    (e). A X-sobel threshoulding is applied to each image.
    (f). Run HoughLinesP to eliminate many noise points found by X-sobel. By setting max_line_gap = 0, the function can
    eliminate the individual points.
    (g). Combine the pixels found by color_mask and X-sobel, and return it. All the images are stored in
     'output_images/threshold'

(2) Apply lane_finding(warped) to find the lane line pixels using histogram method and sliding window. The lane line are
draw back onto the original images. All images are stored in 'output_images/final/'. The curvature and vehicle position
are also put into the images.

All the above functions are applied to the video. The output video is stored as output.mp4. In addition, while
processing video, I use Line() class to store the line fit of the recent frame. When I am confident about the found
 lane lines, I will only search a window around the line center in the last frame(using thresholding2()). After that,
 I will check if the distance between the top pixel in the left lane line and right lane line, and the distance between
 the bottom pixel in the left lane line and right lane line. If their ratio is in a specific range. I consider it a
 valid lane line. If not, I will use thresholding() to refind the lane line center. Also, a moving average is also used
 to make the lane lines more smooth.

