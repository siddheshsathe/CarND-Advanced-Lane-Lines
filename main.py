import cv2
import os
import sys
import matplotlib.pyplot as plt
from calibration import calibrateCamera, undistort
from binarization_utils import binarize
from warp_perspective import warpPerspective
from line import slidingWindowFits, drawOnRoad, Line, usePreviousFits
from moviepy.editor import VideoFileClip
import numpy as np
from globals import xm_per_pix, time_window


_processedFrameCount = 0                    # counter of frames processed (when processing video)
leftLine = Line(buffer=time_window)  # line on the left of the lane
rightLine = Line(buffer=time_window)  # line on the right of the lane


def finalOutputDisplay(image, offFromCenter):
    # add text (curvature and offset info) on the upper right of the blend
    avgCurvatureRadius = np.mean([leftLine.curvature_meter, rightLine.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Radius of Curvature: {:.4f}m'.format(avgCurvatureRadius), (22, 41), font, 0.9, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(image, 'Off from center: {:.4f}m'.format(offFromCenter), (704, 41), font, 0.9, (0,0,0), 2, cv2.LINE_AA)

    return image


def calculateOffFromCenter(leftLine, rightLine, frame_width):
    if leftLine.detected and rightLine.detected:
        leftLine_bottom = np.mean(leftLine.all_x[leftLine.all_y > 0.95 * leftLine.all_y.max()])
        rightLine_bottom = np.mean(rightLine.all_x[rightLine.all_y > 0.95 * rightLine.all_y.max()])
        lane_width = rightLine_bottom - leftLine_bottom
        midpoint = frame_width / 2
        offset_pix = (leftLine_bottom + lane_width / 2) - midpoint
        offFromCenter = xm_per_pix * offset_pix
    else:
        offFromCenter = -1

    return offFromCenter


def frameProcessPipeline(frame, keepState=True):
    
    global leftLine, rightLine, _processedFrameCount

    # undistort the image using coefficients found in calibration
    undistortedImage = undistort(frame, mtx, dist, verbose=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    binaryImage = binarize(undistortedImage, verbose=False)

    # compute perspective transform to obtain bird's eye view
    warpedImage, M, Minv = warpPerspective(binaryImage, verbose=False)

    # fit 2-degree polynomial curve onto lane lines found
    if _processedFrameCount > 0 and keepState and leftLine.detected and rightLine.detected:
        leftLine, rightLine, img_fit = usePreviousFits(warpedImage, leftLine, rightLine, verbose=False)
    else:
        leftLine, rightLine, img_fit = slidingWindowFits(warpedImage, leftLine, rightLine, numWindows=9, verbose=False)

    # compute offset in meter from center of the lane
    offFromCenter = calculateOffFromCenter(leftLine, rightLine, frame_width=frame.shape[1])

    # draw the surface enclosed by lane lines back onto the original frame
    image = drawOnRoad(undistortedImage, Minv, leftLine, rightLine, keepState)

    # stitch on the top of final output images from different steps of the pipeline
    blend_output = finalOutputDisplay(image, offFromCenter)

    _processedFrameCount += 1

    return blend_output


if __name__ == '__main__':

    # first things first: calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrateCamera(calibImageDir='camera_cal')

    mode = sys.argv[1]

    if mode == 'video':

        selector = 'project'
        clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(frameProcessPipeline)
        clip.write_videofile('out_{}_{}.mp4'.format(selector, time_window), audio=False)

    elif mode == 'images':

        test_img_dir = 'test_images'
        for test_img in os.listdir(test_img_dir):

            frame = cv2.imread(os.path.join(test_img_dir, test_img))

            blend = frameProcessPipeline(frame, keepState=False)

            cv2.imwrite('output_images/{}'.format(test_img), blend)

            plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
            plt.show()
            break
    else:
        raise Exception('Mode was given incorrect. Please choosed among "images" or "video"')
