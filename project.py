#!/usr/bin/env python
from __future__ import print_function
import functools
import time
import sys
import cv2 as cv
import numpy as np
from random import randint
import imghdr
import argparse


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

    last_saved_slide = None
    last_saved_slide_mask = None

    def __init__(self, n_slides=0,
                 frame_counter=0,
                 save_interval=30,
                 similarity=0.70):
        self.width = None
        self.height = None
        self.saved_cnt = None
        self.seed = None
        self.number_of_slides = n_slides
        self.frame_counter = frame_counter
        # number of frames which have to pass to run again check if
        # current frame is similar to the last saved slide
        self.save_interval = save_interval
        # ratio of similarity between images based on which we decide if we
        # are going to save an image
        self.similarity = similarity
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
        keep only one with 4 points

        Args:
            image(numpy.ndarray): Image to find contures from

        Returns:
            Found conture on given image
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

    def seed_sufficient(self, image):
        """
        My ghetto green filter that only checks if we really hit the board.
        So it checks if the green value of seed pixel is the highest

        Args:
            image(numpy.ndarra): Image from which we get seed values

        Returns:
            bool: True if found the board. False otherwise
        """
        rgb = image[self.seed[1], self.seed[0]]
        if rgb[1] > rgb[0] and rgb[1] > rgb[2]:
            return True
        return False

    def get_seed(self, image):
        """
        Tries to find good seed for flood fill algorithm by randomly picking
        a point somewhere in the middle of a image and then checking if its
        value is correct

        Args:
            image(numpy.ndarray): Image to search in

        Returns:
            Coordinates of seed position
        """
        if self.seed is not None and self.seed_sufficient(image):
            return self.seed

        h, w = image.shape[:2]
        # three retries to find proper seed
        for i in xrange(1, 3):
            self.seed = (randint(w/4, (w/4)*3), randint(h/4, (h/4)*3))
            if self.seed_sufficient(image):
                break

        return self.seed

    def get_mask(self, image):
        """
        Returns mask of image. We use floodfill algorithm that
        compares 4 neighboring pixels and based on specified
        threashold fills mask image that is bigger by two pixels
        in every direction with white color and than we remove
        left noise by running dilation

        Args:
            image(numpy.ndarray): Preprocessed image

        Returns:
            Mask of given image
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
        """
        Loads last slide and last image mask and perfroms bitwise &
        between masks of current and last image and then applies
        this new mask to old image and current image and tries to
        find keypoints and match them between images. If we match
        more than set similarity then the pictures are almost the same and
        we don't have to save else we save new slide.

        Args:
            image(numpy.ndarray): Image to check
            mask(numpy.ndarray): Mask of given image

        Returns:
            bool: True if images are the same. False otherwise
        """
        if self.last_saved_slide is None or self.last_saved_slide_mask is None:
            last_slide_name = "slide{}.png".format(self.number_of_slides)
            self.last_saved_slide = cv.imread(last_slide_name,
                                              cv.CV_LOAD_IMAGE_COLOR)
            self.last_saved_slide_mask = cv.imread("mask.png",
                                                   cv.CV_LOAD_IMAGE_GRAYSCALE)

        if self.last_saved_slide_mask is None or self.last_saved_slide is None:
            return True

        if mask.shape != self.last_saved_slide_mask.shape:
            return False

        prepared_mask = cv.bitwise_and(self.last_saved_slide_mask, mask)
        self.last_saved_slide = cv.bitwise_and(self.last_saved_slide,
                                               self.last_saved_slide,
                                               mask=prepared_mask)
        check_image = cv.bitwise_and(image, image, mask=prepared_mask)

        orb = cv.ORB()
        kp1, ds1 = orb.detectAndCompute(self.last_saved_slide, None)
        kp2, ds2 = orb.detectAndCompute(check_image, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(ds1, ds2)

        val = float(len(matches))/len(kp1)
        if(val > self.similarity):
            return False

        return True

    def get_sorted_rectangle(self, cnt):
        """
        Tries to determine which corner is which based on
        given conture and then sorts them in correct order so
        we can use it latter to shift perspective of image

        Args:
            cnt(numpy.ndarray): Contures of a board

        Returns:
            Corectly sorted conture of a board
        """
        pts = cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def get_cropped_image(self, rect, image, mask):
        """
        Tries to crop the table from image and warps its perspective
        so we can get table image as if we are standing in front of it

        Args:
            rect(numpy.ndarray): Contures of a board
            image(numpy.ndarray): Image to shift and crop from
            mask(numpy.ndarray): Mask to shift

        Returns:
            Shifted perspective of croped table and its mask for
            further processing and checking.
        """
        (tl, tr, br, bl) = rect
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        max_width = max(int(width_a), int(width_b))
        max_height = max(int(height_a), int(height_b))
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]],
            dtype="float32")

        warp_mat = cv.getPerspectiveTransform(rect, dst)
        warp = cv.warpPerspective(image, warp_mat, (max_width, max_height))
        warp_mask = cv.warpPerspective(mask, warp_mat, (max_width, max_height))
        return (warp, warp_mask)

    def write_image(self, cnt, image, mask):
        """
        Handles writing images to the disk. Tries to extract only board
        from image based on given contures then it performs basic check
        how much is this image different from previous one. Based on
        results decides to write or not

        Args:
            cnt(numpy.ndarray): Contures of a board
            image(numpy.ndarray): Image to compare and write
            mask(numpy.ndarray): Mask of same areas of compared images
        """
        if cnt is None:
            return
        rect = self.get_sorted_rectangle(cnt)
        warp, warp_mask = self.get_cropped_image(rect, image, mask)

        if(self.check_change(warp, warp_mask)):
            cv.imwrite("slide{0}.png".format(self.number_of_slides), warp)
            cv.imwrite("mask.png", warp_mask)
            self.last_saved_slide = warp
            self.last_saved_slide_mask = warp_mask
            self.number_of_slides += 1

    def find_and_draw_edges(self, image, origin, debug=True):
        """
        Transforms color space of original image to HSV.
        Creates a window with trackbars to adjust hsv values
        and to show mask image. Thresholds image to get only green
        parts of original image.
        Calls function find_board which returns conture of a board.
        Draws conture of board

        Args:
            image(numpy.ndarray): Preprocessed image
            origin(numpy.ndarray): Original loaded image
        """
        mask = self.get_mask(image)
        if debug:
            cv.imshow('mask', mask)
        our_cnt = self.find_board(mask)

        if our_cnt is None:
            our_cnt = self.saved_cnt

        if self.saved_cnt is None:
            self.saved_cnt = our_cnt

        img = origin.copy()
        cv.drawContours(img, [our_cnt], -1, (0, 255, 0), 3)
        if(self.frame_counter == 0):
            self.write_image(our_cnt, img, mask)

        self.frame_counter = (self.frame_counter + 1) % self.save_interval
        if debug:
            cv.imshow("final image", img)

    def preprocesing(self, image, debug=True):
        """
        Makes a copy of input image then makes a threasholding operation
        so we can get mask of green areas. Then applies
        that mask to every channel of output image.

        Args:
            image(numpy.ndarray): Image to process
            debug(bool): If true display debuging window

        Returns:
            Image with boosted green channel
        """
        im = image.copy()
        g_boost = cv.inRange(image, (0, 0, 0), (110, 255, 110))
        im[:, :, 0] = cv.bitwise_and(image[:, :, 0], g_boost)
        im[:, :, 1] = cv.bitwise_and(image[:, :, 1], g_boost)
        im[:, :, 2] = cv.bitwise_and(image[:, :, 2], g_boost)

        if debug:
            cv.imshow("preprocesing", im)
        return im

    def get_board(self, image):
        """
        Makes a copy of input image, applies median blur to cartoon-ify
        image. It gets rid of noise and not wanted colors.
        Calls function which will find a board and
        draw edges of that board to original image

        Args:
            image(numpy.ndarray): Image to process
        """
        if self.width is None or self.height is None:
            self.height, self.width = image.shape[:2]

        out = cv.medianBlur(image, 5)
        out = self.preprocesing(out)
        self.find_and_draw_edges(out, image)

    def get_conture(self, image):
        """
        Function copies the image and transforms color space of
        this copied image to HSV. Then gets mask and calls main
        funtion for finding the board

        Args:
            image(numpy.ndarray): Loaded image

        Returns:
             Conture of given image
        """
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = self.get_mask(hsv_image)
        return self.find_board(mask)

    def image_search(self, image):
        """
        Main image search function

        Args:
            image(string): Path to image file to process
        """
        im = cv.imread(image, cv.CV_LOAD_IMAGE_COLOR)
        if(im is None):
            print("Can not open file.", file=sys.stderr)
            return
        self.create_trackbars()
        while(True):
            self.get_board(im)
            if cv.waitKey(1) & 0xFF == 27:
                break

    def video_search(self, video):
        """
        Main video search function

        Args:
            video(string): Path to video file to process
        """
        vid = cv.VideoCapture(video)
        self.create_trackbars()
        while(vid.isOpened()):
            ret, frame = vid.read()
            self.get_board(frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
        vid.release()

    def start_processing(self, input_file):
        """
        Main function to determine if input file is a video or image
        and start processing the file accordingly

        Args:
            input_file(string): Path to input file
        """
        try:
            if (input_file.endswith(video_extension_list)):
                self.video_search(input_file)
            elif (imghdr.what is not None):
                self.image_search(input_file)
            else:
                print("Unrecognized file format", file=sys.stderr)
            cv.destroyAllWindows()
        except IOError:
            print("Wrong file or path to file", file=sys.stderr)


video_extension_list = ("mkv", "wmv", "avi", "mp4")


def main(input_file, slide_number=0, start_frame=0, check_interval=30,
         sim=0.70):
    board = BoardSearcher(n_slides=slide_number, frame_counter=start_frame,
                          save_interval=check_interval, similarity=sim)
    for file_name in input_file:
        board.start_processing(file_name)

if __name__ == '__main__':
    desc = '''
            board2slides - Extracts notes as slides from educational
            (whiteboard/blackboard) videos.
            '''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filename', nargs='+', metavar='filename',
                        help='List of vidos or images to process.')

    parser.add_argument('-n' '--slide-number', nargs='?', default=0,
                        type=int, metavar='N', dest='slide_number',
                        help='''
                             On which slide number to start. Slide with this
                             number will also be loaded. (default: 0)
                             ''')

    parser.add_argument('-i' '--save-interval', nargs='?', default=30,
                        type=int, metavar='I', dest='save_interval',
                        help='''
                             How many frames have to pass to perform check if
                             board changed and possibly save slide from video.
                             (default: 30)
                             ''')

    parser.add_argument('-s' '--similarity', nargs='?', default=0.70,
                        type=float, metavar='S', dest='similarity',
                        help='''
                             On have many percent frames have to be similar to
                             skip saving of slide. (default: 0.70)
                             ''')

    parser.add_argument('-f' '--start-frame', nargs='?', default=0,
                        type=int, metavar='F', dest='start_frame',
                        help='''
                             On which frame to start processing the video.
                             (default: 0)
                             ''')

    args = parser.parse_args()
    main(args.filename, slide_number=args.slide_number,
         start_frame=args.start_frame, check_interval=args.save_interval,
         sim=args.similarity)
