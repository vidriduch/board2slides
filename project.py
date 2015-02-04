#!/usr/bin/env python
import functools
import time
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import yaml


def timeit(func):
    """
    Profiling function to measure time it takes to finish function.
    """
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        msg = 'function [{}] finished in {} ms'
        print(msg.format(func.__name__, int(elapsed_time * 1000)))
        return out
    return newfunc


class BoardSearcher:

    h, s, v, H, S, V = 0, 0, 0, 180, 255, 255

    def __init__(self):
        self.load_config()

    def load_config(self):
        """
        Loads hsv values for image tresholding from config.
        """
        with open('config.tsv') as f:
            out = f.read().strip().split('\t')
            if len(out) == 6:
                self.h, self.s, self.v, self.H, self.S, self.V = map(int, out)

    def save_config(self):
        """
        Saves trackbar tresholding hsv values to config file.
        """
        with open('config.tsv', 'w') as f:
            f.write('\t'.join(map(str, [self.h, self.s, self.v,
                                        self.H, self.S, self.V])))

    def trackbar_callback(self, x):
        """
        Callback for trackbars.
        Stores values of current state of trackbars.
        """
        self.save_config()

    @timeit
    def find_board(self, image):
        """
        Finds a board by calling openCV function to find contures in image.
        Than it sorts those contures and stores the biggest one.
        In case there is more than one we go over all found contures and
        keep only one with 4 points.
        """
        (cnts, _) = cv.findContours(image,
                                    cv.RETR_TREE,
                                    cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
        our_cnt = None
        for c in cnts:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.1 * peri, True)
            if len(approx) == 4:
                our_cnt = approx
                break

        return our_cnt

    def get_mask(self, image):
        """
        Returns mask of image.
        """
        lower_value = np.array([self.h, self.s, self.v])
        upper_value = np.array([self.H, self.S, self.V])
        return cv.inRange(image, lower_value, upper_value)

    def create_trackbars(self):
        """
        Creates window for displaying trackbars and mask of image
        """

        # create window
        cv.namedWindow('result')

        cv.createTrackbar('h', 'result', self.h, 179, self.trackbar_callback)
        cv.createTrackbar('s', 'result', self.s, 255, self.trackbar_callback)
        cv.createTrackbar('v', 'result', self.v, 255, self.trackbar_callback)
        cv.createTrackbar('H', 'result', self.H, 179, self.trackbar_callback)
        cv.createTrackbar('S', 'result', self.S, 255, self.trackbar_callback)
        cv.createTrackbar('V', 'result', self.V, 255, self.trackbar_callback)

    @timeit
    def find_and_draw_edges(self, image, origin):
        """
        Transforms color space of original image from BGR to HSV.
        Creates a window with trackbars to adjust hsv values
        and to show mask image. Tresholds image to get only green
        parts of original image.
        Calls function find_board which returns conture of a board.
        Draws conture of board.
        """
        original_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        self.h = cv.getTrackbarPos('h', 'result')
        self.s = cv.getTrackbarPos('s', 'result')
        self.v = cv.getTrackbarPos('v', 'result')
        self.H = cv.getTrackbarPos('H', 'result')
        self.S = cv.getTrackbarPos('S', 'result')
        self.V = cv.getTrackbarPos('V', 'result')

        mask = self.get_mask(original_image)
        cv.imshow('result', mask)

        our_cnt = self.find_board(mask)

        img = origin.copy()
        cv.drawContours(img, [our_cnt], -1, (0, 255, 0), 3)
        cv.imshow("img", img)

    @timeit
    def get_board(self, image):
        """
        Makes a copy of input image, applies median blur to cartoon-ify
        image. It gets rid of noise and not wanted colors.
        Calls function which will find a board and
        draw edges of that board to original image.
        """
        out = cv.medianBlur(image, 5)
        self.find_and_draw_edges(out, image)

    @timeit
    def get_conture(self, image):
        """
        Returns conture of given image.
        """
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = self.get_mask(hsv_image)
        return self.find_board(mask)

    @timeit
    def image_search(self, image):
        """
        Main image search function.
        """
        im = cv.imread(image, cv.CV_LOAD_IMAGE_COLOR)
        self.create_trackbars()
        while(True):
            self.get_board(im)
            if cv.waitKey(1) & 0xFF == 27:
                break

    def video_search(self, video):
        """
        Main video search function.
        """
        vid = cv.VideoCapture(video)
        self.create_trackbars()
        while(vid.isOpened()):
            ret, frame = vid.read()
            self.get_board(frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
        vid.release()


image_extension_list = ["jpg", "gif", "png"]
video_extension_list = ["mkv", "wmv", "avi", "mp4"]


def main(inputFile):
    board = BoardSearcher()

    if(any(inputFile[-3:] == i for i in video_extension_list)):
        board.video_search(inputFile)
    elif (any(inputFile[-3:] == i for i in image_extension_list)):
        board.image_search(inputFile)
    else:
        print "Unrecognized file format"
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1])
