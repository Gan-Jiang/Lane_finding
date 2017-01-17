# Lane_finding
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
