import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


# selected threshold to highlight yellow lines
yellowLinesThMin = np.array([0, 70, 70])
yellowLinesThMax = np.array([50, 255, 255])


def applyThresholdOnHSV(frame, thMin, thMax, verbose=False):
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    min = np.all(HSV > thMin, axis=2)
    max = np.all(HSV < thMax, axis=2)

    out = np.logical_and(min, max)

    if verbose:
        plt.imshow(out, cmap='gray')
        plt.show()

    return out


def applySobel(frame, ksize):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = np.sqrt(sobx ** 2 + soby ** 2)
    magnitude = np.uint8(magnitude / np.max(magnitude) * 255)

    _, magnitude = cv2.threshold(magnitude, 50, 1, cv2.THRESH_BINARY)

    return magnitude.astype(bool)


def applyEqWhiteMask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eq = cv2.equalizeHist(gray)

    _, th = cv2.threshold(eq, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

    return th


def binarize(img, verbose=False):
    h, w = img.shape[:2]

    binary = np.zeros(shape=(h, w), dtype=np.uint8)

    # highlight yellow lines by threshold in HSV color space
    hsvYellow = applyThresholdOnHSV(img, yellowLinesThMin, yellowLinesThMax, verbose=False)
    binary = np.logical_or(binary, hsvYellow)

    # highlight white lines by thresholding the equalized frame
    eq_white_mask = applyEqWhiteMask(img)
    binary = np.logical_or(binary, eq_white_mask)

    # get Sobel binary mask (thresholded gradients)
    sobel_mask = applySobel(img, ksize=9)
    binary = np.logical_or(binary, sobel_mask)

    # apply a light morphology to "fill the gaps" in the binary image
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if verbose:
        f, ax = plt.subplots(2, 3)
        f.set_facecolor('white')
        ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('input_frame')
        ax[0, 0].set_axis_off()
        ax[0, 0].set_axis_bgcolor('red')
        ax[0, 1].imshow(eq_white_mask, cmap='gray')
        ax[0, 1].set_title('white mask')
        ax[0, 1].set_axis_off()

        ax[0, 2].imshow(hsvYellow, cmap='gray')
        ax[0, 2].set_title('yellow mask')
        ax[0, 2].set_axis_off()

        ax[1, 0].imshow(sobel_mask, cmap='gray')
        ax[1, 0].set_title('sobel mask')
        ax[1, 0].set_axis_off()

        ax[1, 1].imshow(binary, cmap='gray')
        ax[1, 1].set_title('before closure')
        ax[1, 1].set_axis_off()

        ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('after closure')
        ax[1, 2].set_axis_off()
        plt.show()

    return closing
