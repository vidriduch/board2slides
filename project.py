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
    """
    Loads hsv values for image tresholding from config.
    """
    global h, s, v, H, S, V
    with open('config.tsv') as f:
        out = f.read().strip().split('\t')
        if len(out) == 6:
            h, s, v, H, S, V = map(int, out)


def save_config():
    """
    Saves trackbar tresholding hsv values to config file.
    """
    global h, s, v, H, S, V
    with open('config.tsv', 'w') as f:
        f.write('\t'.join(map(str, [h, s, v, H, S, V])))


def timeit(func):
    """
    Profiling function to measure time it takes to finish function.
    """
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        out = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return out
    return newfunc


def trackbar_callback(x):
    """
    Callback for trackbars. 
    Stores values of current state of trackbars.
    """
    save_config()

@timeit
def find_board(image):
    """
    Finds a board by calling openCV function to find contures in image.
    Than it sorts those contures and stores the biggest one. 
    In case there is more than one we go over all found contures and 
    keep only one with 4 points.
    """
    (cnts, _) = cv.findContours(image, cv.RETR_TREE,
                                    cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
    our_cnt = None
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            our_cnt = approx
            break
    
    return our_cnt




@timeit
def find_and_draw_edges(image, origin):
    """
    Transforms color space of original image from BGR to HSV.
    Creates a window with trackbars to adjust hsv values and to show mask image.
    Tresholds image to get only green parts of original image.
    Calls function find_board which returns conture of a board. 
    Draws conture of board.
    """
    global h, s, v, H, S, V

    original_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # create window
    cv.namedWindow('result')

    # create trackbars
    cv.createTrackbar('h', 'result', h, 179, trackbar_callback)
    cv.createTrackbar('s', 'result', s, 255, trackbar_callback)
    cv.createTrackbar('v', 'result', v, 255, trackbar_callback)
    cv.createTrackbar('H', 'result', H, 179, trackbar_callback)
    cv.createTrackbar('S', 'result', S, 255, trackbar_callback)
    cv.createTrackbar('V', 'result', V, 255, trackbar_callback)
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
        
        our_cnt = find_board(mask.copy())
        
        img = origin.copy()
        cv.drawContours(img, [our_cnt], -1, (0, 255, 0), 3)
        cv.imshow("img", img)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break


@timeit
def cartoonify(image):
    """
    Makes a copy of input image, applies a median blur to cartoon-ify the
    image. It gets rid of noise and not wanted colors. Calls function which will 
    find a board and draw edges of that board to original image.
    """
    out = image.copy()
    out = cv.medianBlur(image, 5)
    find_and_draw_edges(out, image)


def main(inputFile):
    im = cv.imread(inputFile, cv.CV_LOAD_IMAGE_COLOR)
    load_config()
    # cartoon-ify image
    cartoonify(im)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1])
