#!/usr/bin/env python
import functools
import time
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import yaml

h, s, v, H, S, V = 0, 0, 0, 180, 255, 255


def load_config():
    global h, s, v, H, S, V
    with open('config.tsv') as f:
        out = f.read().strip().split('\t')
        print out, map(int, out)
        if len(out) == 6:
            h, s, v, H, S, V = map(int, out)


def save_config():
    global h, s, v, H, S, V
    with open('config.tsv', 'w') as f:
        f.write('\t'.join(map(str, [h, s, v, H, S, V])))


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
    save_config()


@timeit
def calculate_histogram(image):
    """
    Gets values of green in a image, calculates difference and returns
    pixels with highest and lowest values.
    """
    global h, s, v, H, S, V
    x = 5
    y = 4
    original_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # create window
    cv.namedWindow('result')

    # create trackbars
    cv.createTrackbar('h', 'result', h, 179, nothing)
    cv.createTrackbar('s', 'result', s, 255, nothing)
    cv.createTrackbar('v', 'result', v, 255, nothing)
    cv.createTrackbar('H', 'result', H, 179, nothing)
    cv.createTrackbar('S', 'result', S, 255, nothing)
    cv.createTrackbar('V', 'result', V, 255, nothing)
    while(1):
        h = cv.getTrackbarPos('h', 'result')
        s = cv.getTrackbarPos('s', 'result')
        v = cv.getTrackbarPos('v', 'result')
        H = cv.getTrackbarPos('H', 'result')
        S = cv.getTrackbarPos('S', 'result')
        V = cv.getTrackbarPos('V', 'result')

        lower_value = np.array([h, s, v])
        upper_value = np.array([H, S, V])
        mask = cv.inRange(original_image, lower_value, upper_value)
        cv.imshow('result', mask)
        (cnts, _) = cv.findContours(mask.copy(), cv.RETR_TREE,
                                    cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
        our_cnt = None
        for c in cnts:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                our_cnt = approx
                break

        hsv_img = original_image.copy()
        cv.drawContours(hsv_img, [our_cnt], -1, (0, 255, 0), 3)
        cv.imshow("hsv", hsv_img)
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
    load_config()
    # cartoon-ify image
    cartoonify(im)
    # show image
    cv.imshow("project", im)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1])
