#!/usr/bin/env python
import functools
import time
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        out = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return out
    return newfunc


@timeit
def find_indexes(arr):
    """
    Returns index of highest and lowest element in a array.
    """

    min = np.argmin(arr)
    max = np.argmax(arr)
    return (min, max)

def nothing(x):
    pass

@timeit
def calculate_histogram(image):
    """
    Gets values of green in a image, calculates difference and returns
    pixels with highest and lowest values.
    """
    x = 5
    y = 4
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", image)
    h,s,v = 100,100,100
    #create window
    cv.namedWindow('result')
    #create trackbars
    cv.createTrackbar('h', 'result', 0, 179, nothing)
    cv.createTrackbar('s', 'result', 0, 255, nothing)
    cv.createTrackbar('v', 'result', 0, 255, nothing)
    while(1):
        h = cv.getTrackbarPos('h', 'result')
        s = cv.getTrackbarPos('s', 'result')
        v = cv.getTrackbarPos('v', 'result')
        lower_value = np.array([h,s,v])
        upper_value = np.array([180,255,255])
        mask = cv.inRange(image, lower_value, upper_value)
        cv.imshow('result', mask)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
    return (find_indexes(x), find_indexes(y))


@timeit
def cartoonify(image):
    """
    Makes a copy of input image, applies a median blur to cartoon-ify the
    image. It gets rid of noise and not wanted colors. Calculates histogram of
    green color and draws edges of a board.
    """

    out = image.copy()
    out = cv.medianBlur(image, 5)
    (x, y) = calculate_histogram(out)
    # draw edges of a board
    cv.circle(image, (x[1], y[0]), 5, (0, 0, 255), -1)
    cv.circle(image, (x[1], y[1]), 5, (0, 0, 255), -1)
    cv.circle(image, (x[0], y[0]), 5, (0, 0, 255), -1)
    cv.circle(image, (x[0], y[1]), 5, (0, 0, 255), -1)


def main(inputFile):
    im = cv.imread(inputFile, cv.CV_LOAD_IMAGE_COLOR)
    # cartoon-ify image
    cartoonify(im)
    # show image
    cv.imshow("project", im)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1])
