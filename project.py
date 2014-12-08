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


@timeit
def calculate_histogram(image):
    """
    Gets values of green in a image, calculates difference and returns
    pixels with highest and lowest values.

    """
    (width, height, _) = image.shape
    vlines = np.zeros(width)
    hlines = np.zeros(height)
    for x in range(width):
        for y in range(height):
            if 100 < image[x][y][2] < 256:
                hlines[y] += 1
                vlines[x] += 1

    y = np.diff(vlines)
    x = np.diff(hlines)
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
