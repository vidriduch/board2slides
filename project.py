#!/usr/bin/env python
import functools
import time
import sys
import cv2 as cv
import numpy as np
from random import randint


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
    def __init__(self):
        self.width = None
        self.height = None
        self.saved_cnt = None
        self.seed = None
        self.numberOfSlides = None
        self.frameCounter = None
        self.saveInterval = None
        self.similarity = None
        self.load_config()

    def load_config(self):
        """
        Loads hsv values for image tresholding from config.
        """
        with open('config.tsv') as f:
            out = f.read().strip().split('\t')
            if len(out) == 2:
                self.lo, self.hi = map(int, out)

    def save_config(self):
        """
        Saves trackbar tresholding hsv values to config file.
        """
        with open('config.tsv', 'w') as f:
            f.write('\t'.join(map(str, [self.lo, self.hi])))

    def trackbar_callback(self, x):
        """
        Callback for trackbars.
        Stores values of current state of trackbars.
        """
        self.save_config()

    def find_board(self, image):
        """
        Finds a board by calling openCV function to find contures in image.
        Than it sorts those contures and stores the biggest one.
        In case there is more than one we go over all found contures and
        keep only one with 4 points.
        """
        im = image.copy()
        (cnts, _) = cv.findContours(im,
                                    cv.RETR_TREE,
                                    cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
        our_cnt = None
        for c in cnts:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.1 * peri, True)
            if len(approx) == 4:
                # The board needs to be at least 1/3 of the image size
                min_size = np.array([self.height * 1/3.0, self.height * 1/3.9])

                a = np.abs(approx[0] - approx[2])[0] > min_size
                b = np.abs(approx[1] - approx[3])[0] > min_size
                true = [True, True]
                if np.array_equal(a, true) or np.array_equal(b, true):
                    our_cnt = approx
                break

        return our_cnt

    def get_seed(self, image):
        if self.seed is not None:
            rgb = image[self.seed[1], self.seed[0]]
            if rgb[1] > rgb[0] and rgb[1] > rgb[2]:
                return self.seed

        h, w = image.shape[:2]
        self.seed = (randint(w/4, (w/4)*3), randint(h/4, (h/4)*3))
        rgb = image[self.seed[1], self.seed[0]]
        # three retries to find proper seed
        for i in xrange(1, 3):
            if rgb[1] > rgb[0] and rgb[1] > rgb[2]:
                break
        return self.seed

    def get_mask(self, image):
        """
        Returns mask of image. We use floodfill algorithm that
        compares 4 neighboring pixels and based on specified
        threashold fills mask image that is bigger by two pixels
        in every direction with white color and than we remove
        left noise by running dilation.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        connectivity = 4
        mask[:] = 0
        self.lo = cv.getTrackbarPos('lo', 'result')
        self.hi = cv.getTrackbarPos('hi', 'result')
        flags = connectivity
        flags |= cv.FLOODFILL_MASK_ONLY
        flags |= 255 << 8
        self.seed = self.get_seed(image)
        cv.floodFill(image, mask, self.seed, (255, 255, 255), (self.lo,)*3,
                     (self.hi,)*3, flags)
        kernel = np.ones((1, 1), np.uint8)
        mask = cv.dilate(mask, kernel, iterations=4)
        return mask

    def create_trackbars(self):
        """
        Creates window for displaying trackbars and mask of image
        """
        # create window
        cv.namedWindow('result')
        cv.createTrackbar('lo', 'result', self.lo, 255, self.trackbar_callback)
        cv.createTrackbar('hi', 'result', self.hi, 255, self.trackbar_callback)

    def check_change(self, image, mask):
        imageOld = cv.imread("slide"+repr(self.numberOfSlides)+".png",
                                cv.CV_LOAD_IMAGE_COLOR)
        maskOld = cv.imread("mask.png", cv.CV_LOAD_IMAGE_GRAYSCALE)
        if(maskOld is None):
            return True

        if(mask.shape != maskOld.shape):
            return False

        applyMask = cv.bitwise_and(maskOld, mask)
        imageOld = cv.bitwise_and(imageOld, imageOld, mask=applyMask)
        checkImage = cv.bitwise_and(image, image, mask=applyMask)

        orb = cv.ORB()
        kp1, ds1 = orb.detectAndCompute(imageOld, None)
        kp2, ds2 = orb.detectAndCompute(checkImage, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(ds1, ds2)

        val = float(len(matches))/len(kp1)
        if(val > self.similarity):
            print "Is the same"

        return False

    def get_sorted_rectangle(self, cnt, image):
        pts = cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def get_croped_image(self, rect, image, mask):
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]],
            dtype="float32")

        warpMat = cv.getPerspectiveTransform(rect, dst)
        warp = cv.warpPerspective(image, warpMat, (maxWidth, maxHeight))
        warpMask = cv.warpPerspective(mask, warpMat, (maxWidth, maxHeight))
        return (warp, warpMask)

    def write_image(self, cnt, image, mask):
        if(cnt is None):
            return
        rect = self.get_sorted_rectangle(cnt, image)
        warp, warpMask = self.get_croped_image(rect, image, mask)

        if(self.check_change(warp, warpMask)):
            cv.imwrite("slide" + repr(self.numberOfSlides) + ".png", warp)
            cv.imwrite("mask.png", warpMask)
            self.numberOfSlides += 1

    def find_and_draw_edges(self, image, origin):
        """
        Transforms color space of original image from BGR to HSV.
        Creates a window with trackbars to adjust hsv values
        and to show mask image. Tresholds image to get only green
        parts of original image.
        Calls function find_board which returns conture of a board.
        Draws conture of board.
        """
        mask = self.get_mask(image)
        cv.imshow('result', mask)
        our_cnt = self.find_board(mask)

        if our_cnt is None:
            our_cnt = self.saved_cnt

        if self.saved_cnt is None:
            self.saved_cnt = our_cnt

        img = origin.copy()
        cv.drawContours(img, [our_cnt], -1, (0, 255, 0), 3)
        if(self.frameCounter % self.saveInterval == 0):
            self.write_image(our_cnt, img, mask)

        self.frameCounter = (self.frameCounter + 1) % self.saveInterval
        cv.imshow("img", img)

    def preprocesing(self, image):
        im = image.copy()
        imGboost = cv.inRange(image, (0, 0, 0), (110, 255, 110))
        im[:, :, 0] = cv.bitwise_and(image[:, :, 0], imGboost)
        im[:, :, 1] = cv.bitwise_and(image[:, :, 1], imGboost)
        im[:, :, 2] = cv.bitwise_and(image[:, :, 2], imGboost)
        cv.imshow("preprocesing", im)
        return im

    def get_board(self, image):
        """
        Makes a copy of input image, applies median blur to cartoon-ify
        image. It gets rid of noise and not wanted colors.
        Calls function which will find a board and
        draw edges of that board to original image.
        """
        if self.width is None or self.height is None:
            self.height, self.width = image.shape[:2]

        out = cv.medianBlur(image, 5)
        out = self.preprocesing(out)
        self.find_and_draw_edges(out, image)

    def get_conture(self, image):
        """
        Returns conture of given image.
        """
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = self.get_mask(hsv_image)
        return self.find_board(mask)

    def image_search(self, image):
        """
        Main image search function.
        """
        im = cv.imread(image, cv.CV_LOAD_IMAGE_COLOR)
        if(im is None):
            print "Can not open file."
            return
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
    board.numberOfSlides = 0
    board.frameCounter = 0
    # how many matches we have to find to not save
    board.similarity = 0.70
    # how often run similarity check and save slide function in frames
    board.saveInterval = 30

    if(any(inputFile[-3:] == i for i in video_extension_list)):
        board.video_search(inputFile)
    elif (any(inputFile[-3:] == i for i in image_extension_list)):
        board.image_search(inputFile)
    else:
        print "Unrecognized file format"
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1])
