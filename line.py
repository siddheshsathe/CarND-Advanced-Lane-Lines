import numpy as np
import cv2
import glob
import collections
import matplotlib.pyplot as plt
from calibration import calibrateCamera, undistort
from binarization_utils import binarize
from warp_perspective import warpPerspective
from globals import ym_per_pix, xm_per_pix


class Line(object):
    def __init__(self, buffer=10):

        # flag to mark if the line was detected the last iteration
        self.detected = False

        # polynomial coefficients fitted on the last iteration
        self.previousFitPixel = None
        self.previousFitMeter = None

        # list of polynomial coefficients of the last N iterations
        self.latestFitPixel = collections.deque(maxlen=buffer)
        self.latestFitMeter = collections.deque(maxlen=2 * buffer)

        self.radius_of_curvature = None

        # store all pixels coords (x, y) of line detected
        self.all_x = None
        self.all_y = None

    def updateLine(self, upatedFitPixel, upatedFitMeter, detected, clear_buffer=False):
        self.detected = detected

        if clear_buffer:
            self.latestFitPixel = []
            self.latestFitMeter = []

        self.previousFitPixel = upatedFitPixel
        self.previousFitMeter = upatedFitMeter

        self.latestFitPixel.append(self.previousFitPixel)
        self.latestFitMeter.append(self.previousFitMeter)

    def draw(self, mask, color=(255, 0, 0), lineWidth=50, average=False):
        h, w, c = mask.shape

        plotY = np.linspace(0, h - 1, h)
        coeffs = self.average_fit if average else self.previousFitPixel

        lineCenter = coeffs[0] * plotY ** 2 + coeffs[1] * plotY + coeffs[2]
        leftLine = lineCenter - lineWidth // 2
        rightLine = lineCenter + lineWidth // 2

        # Some magic here to recast the x and y points into usable format for cv2.fillPoly()
        leftPoints = np.array(list(zip(leftLine, plotY)))
        rightPoints = np.array(np.flipud(list(zip(rightLine, plotY))))
        pts = np.vstack([leftPoints, rightPoints])

        # Draw the lane onto the warped blank image
        return cv2.fillPoly(mask, [np.int32(pts)], color)

    @property
    # average of polynomial coefficients of the last N iterations
    def average_fit(self):
        return np.mean(self.latestFitPixel, axis=0)

    @property
    # radius of curvature of the line (averaged)
    def curvature(self):
        yEval = 0
        coeffs = self.average_fit
        return ((1 + (2 * coeffs[0] * yEval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @property
    # radius of curvature of the line (averaged)
    def curvature_meter(self):
        yEval = 0
        coeffs = np.mean(self.latestFitMeter, axis=0)
        return ((1 + (2 * coeffs[0] * yEval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])


def slidingWindowFits(binaryThresholded, lineLeft, lineRight, numWindows=9, verbose=False):
    height, width = binaryThresholded.shape

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binaryThresholded[height//2:-30, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binaryThresholded, binaryThresholded, binaryThresholded)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(height / numWindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binaryThresholded.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100  # width of the windows +/- margin
    minpix = 50   # minimum number of pixels found to recenter window

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    # We've already implemented this in tutorials. Copying it as it is.
    for window in range(numWindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
                          & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                           & (nonzero_x < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    lineLeft.all_x, lineLeft.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    lineRight.all_x, lineRight.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

    detected = True
    if not list(lineLeft.all_x) or not list(lineLeft.all_y):
        left_pixel = lineLeft.previousFitPixel
        left_meter = lineLeft.previousFitMeter
        detected = False
    else:
        left_pixel = np.polyfit(lineLeft.all_y, lineLeft.all_x, 2)
        left_meter = np.polyfit(lineLeft.all_y * ym_per_pix, lineLeft.all_x * xm_per_pix, 2)

    if not list(lineRight.all_x) or not list(lineRight.all_y):
        right_pixel = lineRight.previousFitPixel
        right_meter = lineRight.previousFitMeter
        detected = False
    else:
        right_pixel = np.polyfit(lineRight.all_y, lineRight.all_x, 2)
        right_meter = np.polyfit(lineRight.all_y * ym_per_pix, lineRight.all_x * xm_per_pix, 2)

    lineLeft.updateLine(left_pixel, left_meter, detected=detected)
    lineRight.updateLine(right_pixel, right_meter, detected=detected)

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    leftx = left_pixel[0] * ploty ** 2 + left_pixel[1] * ploty + left_pixel[2]
    rightx = right_pixel[0] * ploty ** 2 + right_pixel[1] * ploty + right_pixel[2]

    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    if verbose:
        f, ax = plt.subplots(1, 2)
        f.set_facecolor('white')
        ax[0].imshow(binaryThresholded, cmap='gray')
        ax[1].imshow(out_img)
        ax[1].plot(leftx, ploty, color='yellow')
        ax[1].plot(rightx, ploty, color='yellow')
        ax[1].set_xlim(0, 1280)
        ax[1].set_ylim(720, 0)

        plt.show()

    return lineLeft, lineRight, out_img


def usePreviousFits(binaryThresholded, lineLeft, lineRight, verbose=False):
    height, width = binaryThresholded.shape

    left_pixel = lineLeft.previousFitPixel
    right_pixel = lineRight.previousFitPixel

    nonzero = binaryThresholded.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    left_lane_inds = (
    (nonzero_x > (left_pixel[0] * (nonzero_y ** 2) + left_pixel[1] * nonzero_y + left_pixel[2] - margin)) & (
    nonzero_x < (left_pixel[0] * (nonzero_y ** 2) + left_pixel[1] * nonzero_y + left_pixel[2] + margin)))
    right_lane_inds = (
    (nonzero_x > (right_pixel[0] * (nonzero_y ** 2) + right_pixel[1] * nonzero_y + right_pixel[2] - margin)) & (
    nonzero_x < (right_pixel[0] * (nonzero_y ** 2) + right_pixel[1] * nonzero_y + right_pixel[2] + margin)))

    # Extract left and right line pixel positions
    lineLeft.all_x, lineLeft.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    lineRight.all_x, lineRight.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

    detected = True
    if not list(lineLeft.all_x) or not list(lineLeft.all_y):
        left_pixel = lineLeft.previousFitPixel
        left_meter = lineLeft.previousFitMeter
        detected = False
    else:
        left_pixel = np.polyfit(lineLeft.all_y, lineLeft.all_x, 2)
        left_meter = np.polyfit(lineLeft.all_y * ym_per_pix, lineLeft.all_x * xm_per_pix, 2)

    if not list(lineRight.all_x) or not list(lineRight.all_y):
        right_pixel = lineRight.previousFitPixel
        right_meter = lineRight.previousFitMeter
        detected = False
    else:
        right_pixel = np.polyfit(lineRight.all_y, lineRight.all_x, 2)
        right_meter = np.polyfit(lineRight.all_y * ym_per_pix, lineRight.all_x * xm_per_pix, 2)

    lineLeft.updateLine(left_pixel, left_meter, detected=detected)
    lineRight.updateLine(right_pixel, right_meter, detected=detected)

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    leftx = left_pixel[0] * ploty ** 2 + left_pixel[1] * ploty + left_pixel[2]
    rightx = right_pixel[0] * ploty ** 2 + right_pixel[1] * ploty + right_pixel[2]

    # Create an image to draw on and an image to show the selection window
    img_fit = np.dstack((binaryThresholded, binaryThresholded, binaryThresholded)) * 255
    window_img = np.zeros_like(img_fit)

    # Color in left and right line pixels
    img_fit[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    img_fit[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([leftx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([leftx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([rightx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([rightx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(img_fit, 1, window_img, 0.3, 0)

    if verbose:
        plt.imshow(result)
        plt.plot(leftx, ploty, color='yellow')
        plt.plot(rightx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()

    return lineLeft, lineRight, img_fit


def drawOnRoad(undistortedImage, Minv, lineLeft, lineRight, keepState):
    height, width, _ = undistortedImage.shape

    left = lineLeft.average_fit if keepState else lineLeft.previousFitPixel
    right = lineRight.average_fit if keepState else lineRight.previousFitPixel

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    leftx = left[0] * ploty ** 2 + left[1] * ploty + left[2]
    rightx = right[0] * ploty ** 2 + right[1] * ploty + right[2]

    # draw road as green polygon on original frame
    road_warp = np.zeros_like(undistortedImage, dtype=np.uint8)
    leftPoints = np.array([np.transpose(np.vstack([leftx, ploty]))])
    rightPoints = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((leftPoints, rightPoints))
    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
    roadUnwarped = cv2.warpPerspective(road_warp, Minv, (width, height))  # Warp back to original image space

    putOnTheRoad = cv2.addWeighted(undistortedImage, 1., roadUnwarped, 0.3, 0)

    # now separately draw solid lines to highlight them
    line_warp = np.zeros_like(undistortedImage)
    line_warp = lineLeft.draw(line_warp, color=(255, 0, 0), average=keepState)
    line_warp = lineRight.draw(line_warp, color=(0, 0, 255), average=keepState)
    line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))

    lines_mask = putOnTheRoad.copy()
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]

    putOnTheRoad = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=putOnTheRoad, beta=0.5, gamma=0.)

    return putOnTheRoad