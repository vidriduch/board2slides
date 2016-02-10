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
import csv


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
    last_saved_hash = None
    last_hash = None
    similarity = 0.60
    hash_function = None
    debug = True

    grid_size = (8, 8)
    stored_section = None
    last_saved_section = None
    section_threshold = 15
    reject_threshold = 10
    section_overlap = 10

    board_extraction_tp = []
    board_extraction_fp = []

    debug_image = None
    eval_file = None

    def __init__(self, n_slides=0,
                 frame_counter=0,
                 save_interval=30,
                 similarity=0.60,
                 compare_function='dhash',
                 section_thresh=15,
                 section_rjct=10,
                 section_overlap=10,
                 grid_size=(8, 8),
                 eval_filename=None,
                 debug=True):
        self.debug = debug
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
        self.__func_keyword_to_function__(compare_function)
        self.load_config()
        self.section_threshold = section_thresh
        self.reject_threshold = section_rjct
        self.section_overlap = section_overlap
        self.grid_size = grid_size
        self.eval_file = eval_filename
        self.stored_section = [[[None for x in range(2)]
                               for x in range(grid_size[0])]
                               for x in range(grid_size[1])]
        self.last_saved_section = [[[None for x in range(2)]
                                   for x in range(grid_size[0])]
                                   for x in range(grid_size[1])]

    def __func_keyword_to_function__(self, keyword):
        switcher = {
            'dhash': '__compute_dhash__',
            'phash': '__compute_phash__',
            'ahash': '__compute_ahash__',
            'orb': None
        }
        compare_function = switcher.get(keyword, '__compute_dhash__')
        self.hash_function = getattr(self, compare_function)

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

    def evaluate_board_extraction(self, image, cnts, eval_file):
        """
        Function for evaluating how good we extracted table from video or
        image. It computes bitwise and between real table and extracted table
        to get common areas and then computes difference between these
        tables to get areas that we say are table but are not.

        Args:
            image(numpy.ndarray): Full image of entire scene
            cnts(list): Contures of found table
            eval_file(string): File with boards metadata
        """
        with open(eval_file, "r") as meta_file:
            reader = csv.reader(meta_file)
            i = 0
            rect = None
            for row in reader:
                if i == self.frame_counter:
                    rect = row
                i += 1
            if rect is None:
                return
            orig_board = np.zeros(image.shape[:2], dtype="uint8")
            cnts_blank = np.zeros(image.shape[:2], dtype="uint8")
            poly = np.array([[rect[0], rect[2]], [rect[1], rect[2]],
                             [rect[1], rect[3]], [rect[0], rect[3]]],
                            np.int32)
            cv.fillPoly(orig_board, [poly], (255, 255, 255))
            (board_cnt, _) = cv.findContours(orig_board.copy(), cv.RETR_TREE,
                                             cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(cnts_blank, [cnts], 0, (255, 255, 255), -1)
            board_area = cv.contourArea(board_cnt[0])
            eval_board = cv.bitwise_and(orig_board, cnts_blank)
            (eval_cnt, _) = cv.findContours(eval_board, cv.RETR_TREE,
                                            cv.CHAIN_APPROX_SIMPLE)
            diff_board = cv.subtract(cnts_blank, orig_board)
            (diff_cnt, _) = cv.findContours(diff_board, cv.RETR_TREE,
                                            cv.CHAIN_APPROX_SIMPLE)
            diff_area = cv.contourArea(diff_cnt[0])
            eval_area = cv.contourArea(eval_cnt[0])
            tp = eval_area/board_area
            fp = diff_area/board_area
            self.board_extraction_tp.append(tp)
            self.board_extraction_fp.append(fp)

    def evaluate_compare_functions(self, eval_file):
        """
        Function for evaluating how many slides we made. How good
        are our compare functions. It checks meta data file if in
        range of our current frame we should create slide or not.

        Args:
            eval_file(string): File with media file metadata
        """
        with open(eval_file, "r") as meta_file:
            reader = csv.reader(meta_file)
            i = 0
            true_val = False
            for row in reader:
                if i >= self.frame_counter-120 and i <= self.frame_counter+120:
                    if row[4] is True:
                        true_val = True
                i += 1
            if true_val is False:
                self.slide_creating_fp += 1
            else:
                self.slide_creating_tp += 1

    def find_board(self, image):
        """
        Finds a board by calling openCV function to find contures in image.
        Than it sorts those contures and stores the biggest one.
        In case there is more than one we go over all found contures and
        keep only one with 4 points

        Args:
            image(numpy.ndarray): Image to find contures from

        Returns:
            Found conture in given image
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
        if self.debug is True:
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

    def __hamming_distance__(self, hash1, hash2):
        if(len(hash1) != len(hash2)):
            return 0
        return sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(hash1, hash2)))

    def __check_change_orb__(self, image, last_image):
        """
        Computes ORB fetures on last saved slide and given image.
        Tries to match features of these images and calculates ratio
        of matched features with all found features in last saved slide

        Args:
            image(numpy.ndarray): Image from which to get features
            last_image(numpy.ndarray): Image which we want to compare

        Returns:
            float: Similararity between last saved image and given image
        """
        orb = cv.ORB()
        kp1, ds1 = orb.detectAndCompute(self.last_image, None)
        kp2, ds2 = orb.detectAndCompute(image, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(ds1, ds2)
        return float(len(matches))/len(kp1)

    def __compute_ahash__(self, image):
        """
        Computes aHash. Implemantation based on
        http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

        Args:
            image(numpy.ndarray): Image from which to compute the hash

        Returns:
            numpy.ndarray: 2D binary array with computed aHash
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (8, 8), interpolation=cv.INTER_AREA)
        mean = cv.mean(resized)[0]
        ret = resized > mean
        return ret.flatten()

    def __compute_phash__(self, image):
        """
        Computes pHash based on discrete cosine transformation.
        Implemantation based on
        http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

        Args:
            image(numpy.ndarray): Image from which to compute the hash

        Returns:
            numpy.ndarray: 2D binary array with computed pHash
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (32, 32), interpolation=cv.INTER_AREA)
        dct = cv.dct(np.float32(resized))
        dct = np.uint8(dct)
        dct_low_freq = dct[:8, :8]
        med = np.median(dct_low_freq)
        ret = dct_low_freq > med
        return ret.flatten()

    def __compute_dhash__(self, image):
        """
        Computes dHash. Implemantation based on
        http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

        Args:
            image(numpy.ndarray): Image from which to compute the hash

        Returns:
            numpy.ndarray: 2D binary array with computed dHash
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (9, 8), interpolation=cv.INTER_AREA)
        ret = resized[:, 1:] > resized[:, :-1]
        return ret.flatten()

    def __compare_hashes__(self, image, compare_image=None):
        """
        Generate hashes of last saved slide and given image and
        computes hamming distance between hashes

        Args:
            image(numpy.ndarray): Image from which to compute the hash and
                                  compare to last saved slide hash

        Returns:
            float: Ratio between hamming distance of hash and length of
                   the hash
        """
        if compare_image is None:
            if self.last_saved_hash is None:
                tmp_hash = self.hash_function(self.last_saved_slide)
                self.last_saved_hash = tmp_hash
            self.last_hash = self.hash_function(image)
            new_hash = self.last_hash
            old_hash = self.last_saved_hash
        else:
            new_hash = self.hash_function(image)
            old_hash = self.hash_function(compare_image)

        hamming = self.__hamming_distance__(new_hash, old_hash)
        hash_len = len(new_hash)
        return float((hash_len - hamming))/hash_len

    def check_change(self, image, mask):
        """
        Loads last slide and last image mask and perfroms bitwise &
        between masks of current and last image and then applies
        this new mask to old and current image. Then calls similarity check
        function.If returned value is more than set similarity then the
        pictures are almost the same and we don't have to save else we
        save new slide.

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
        last_saved_masked_slide = cv.bitwise_and(self.last_saved_slide,
                                                 self.last_saved_slide,
                                                 mask=prepared_mask)
        check_image = cv.bitwise_and(image, image, mask=prepared_mask)

        if self.hash_function is None:
            val = self.__check_change_orb__(check_image,
                                            last_saved_masked_slide)
        else:
            val = self.__compare_hashes__(check_image,
                                          last_saved_masked_slide)
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

    def write_image(self, cnt, board, mask):
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
        self.stitch_board(board)
        if(self.check_change(board, mask)):
            cv.imwrite("slide{0}.png".format(self.number_of_slides), board)
            cv.imwrite("mask.png", mask)
            self.last_saved_slide = board
            self.last_saved_slide_mask = mask
            self.last_saved_hash = self.last_hash
            self.last_saved_section = self.stored_section
            self.number_of_slides += 1

        for x in range(0, self.grid_size[0]):
            for y in range(0, self.grid_size[1]):
                self.stored_section[x][y][0] = None
                self.stored_section[x][y][1] = None

    def find_and_draw_edges(self, image, origin):
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
        if self.debug:
            cv.imshow('mask', mask)
        our_cnt = self.find_board(mask)

        if our_cnt is None:
            our_cnt = self.saved_cnt

        if self.saved_cnt is None:
            self.saved_cnt = our_cnt
        img = origin.copy()
        cv.drawContours(img, [our_cnt], -1, (0, 0, 255), 3)
        if our_cnt is None:
            return
        if self.eval_file is not None:
            self.evaluate_board_extraction(image, our_cnt, self.eval_file)

        rect = self.get_sorted_rectangle(our_cnt)
        warp, warp_mask = self.get_cropped_image(rect, img, mask)
        self.split_board(warp, warp_mask)
        if(self.frame_counter % self.save_interval == 0):
            self.write_image(our_cnt, warp, warp_mask)

        self.frame_counter += 1
        if self.debug:
            cv.imshow("final image", img)

    def preprocesing(self, image):
        """
        Makes a copy of input image then makes a threasholding operation
        so we can get mask of green areas. Then applies
        that mask to every channel of output image.

        Args:
            image(numpy.ndarray): Image to process

        Returns:
            Image with boosted green channel
        """
        im = image.copy()
        g_boost = cv.inRange(image, (0, 0, 0), (110, 255, 110))
        im[:, :, 0] = cv.bitwise_and(image[:, :, 0], g_boost)
        im[:, :, 1] = cv.bitwise_and(image[:, :, 1], g_boost)
        im[:, :, 2] = cv.bitwise_and(image[:, :, 2], g_boost)

        if self.debug:
            cv.imshow("preprocesing", im)
        return im

    def __get_occlusion_mask__(self, board):
        """
        Function tries to extract mask of object in front of the table
        by diff-ing last saved slide against current image and then thresholds
        this to get mask of image. After this we dilate image to get rid of
        small blobs.

        Args:
            board(numpy.ndarray): Image of the croped board

        Returns:
           (numpy.ndarray): Mask of objects in foreground
        """
        if self.last_saved_slide is None:
            return np.zeros(board.shape[:2], dtype="uint8")

        height, width = board.shape[:2]
        size = height/3 * width/3
        gray_old = cv.cvtColor(self.last_saved_slide, cv.COLOR_BGR2GRAY)
        gray_new = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
        if gray_old.shape > gray_new.shape:
            gray_old = cv.resize(gray_old, (gray_new.shape[1],
                                            gray_new.shape[0]))
        elif gray_old.shape < gray_new.shape:
            gray_new = cv.resize(gray_new, (gray_old.shape[1],
                                            gray_old.shape[0]))
        frame_delta = cv.absdiff(gray_old, gray_new)
        thresh = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1]
        inter = cv.dilate(thresh, None, iterations=2)
        (cnts, _) = cv.findContours(inter,
                                    cv.RETR_TREE,
                                    cv.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv.contourArea(c) > size:
                cv.drawContours(inter, [c], 0, 255, -1)
            else:
                cv.drawContours(inter, [c], 0, 0, -1)
        return inter

    def __get_section_boundary__(self, x, y, section_size, width, height):
        """
        Funcitons returns based on given arguments boundary of individual
        section.

        Args:
            x(int): number of section on x axis
            y(int): number of section on y axis
            section_size(tuple): width and height of a section
            width(int): width of table
            height(int): height of table

        Returns:
             Coordiantes of cornes of a section
        """
        y_first = y*section_size[1]-self.section_overlap
        y_second = (y+1)*section_size[1]+self.section_overlap
        x_first = x*section_size[0]-self.section_overlap
        x_second = (x+1)*section_size[0]+self.section_overlap
        if y_first < 0:
            y_first = 0
        if y_second > height:
            y_second = height
        if x_first < 0:
            x_first = 0
        if x_second > width:
            x_second = width
        return (y_first, y_second, x_first, x_second)

    def split_board(self, board, mask):
        """
        Function splits the board into sections that are then
        individually processed. It runs through every section and
        checks if it touches any occluding object if it does "seen"
        counter is not incresed. If it doesn't sections is stored in
        a list and "seen" counter is incresed

        Args:
            board(numpy.ndarray): Image of crop and rotated board
            mask(numpy.ndarray): Mask of crop and rotated board

        Return:
            (numpy.ndarray): Final image of board without occluding objects
        """
        height, width = board.shape[:2]
        section_size = (width/self.grid_size[0], height/self.grid_size[1])
        occlusion_mask = self.__get_occlusion_mask__(board)

        if self.debug:
            self.debug_image = board.copy()
        for x in range(0, self.grid_size[0]):
            for y in range(0, self.grid_size[1]):
                boundaries = self.__get_section_boundary__(x, y, section_size,
                                                           width, height)
                tmpimg = board[boundaries[0]:boundaries[1],
                               boundaries[2]:boundaries[3]]
                tmp = np.zeros(board.shape[:2], dtype="uint8")
                cv.rectangle(tmp, (x*section_size[0], y*section_size[1]),
                                  ((x+1)*section_size[0],
                                   (y+1)*section_size[1]),
                             255, -1)
                if occlusion_mask.shape > tmp.shape:
                    occlusion_mask = cv.resize(occlusion_mask,
                                               (tmp.shape[1], tmp.shape[0]))
                elif occlusion_mask.shape < tmp.shape:
                    tmp = cv.resize(tmp, (occlusion_mask.shape[1],
                                          occlusion_mask.shape[0]))
                intersection = cv.bitwise_and(occlusion_mask, tmp)
                count = cv.countNonZero(intersection)
                if self.stored_section[x][y][0] is None:
                    self.stored_section[x][y][0] = tmpimg
                    self.stored_section[x][y][1] = 0
                if count <= 0:
                    self.stored_section[x][y][1] += 1
                    if self.debug:
                        color = (0, 255, 0)
                else:
                    if self.debug:
                        color = (0, 0, 255)

                if self.debug:
                    cv.rectangle(self.debug_image, (x*section_size[0],
                                                    y*section_size[1]),
                                                   ((x+1)*section_size[0],
                                                    (y+1)*section_size[1]),
                                 color, 1)

    def stitch_board(self, board):
        """
        Function takes sections of a board and based on given thresholding
        values tries to stitch them into one final image. If we seen
        section more than given section thresholding value we sow this
        section into final slide. If section is lower than section reject
        threshold we sow last good section into final image. If value is
        between these two thresholds we check how similar is this section
        to last good save one if they are similar last good one is sow if they
        are not we got some new information and new section is put into final
        slide

        Args:
            board(numpy.ndarray): Image of a board

        Returns:
            (numpy.ndarray): Final image of a board from stitched sections
        """
        height, width = board.shape[:2]
        section_size = (width/self.grid_size[0], height/self.grid_size[1])
        if self.last_saved_slide is None:
            blank = np.zeros((height, width, 3), dtype="uint8")
        else:
            blank = self.last_saved_slide.copy()
        if self.debug:
            font_offset = 35
        for x in range(0, self.grid_size[0]):
            for y in range(0, self.grid_size[1]):
                seen = self.stored_section[x][y][1]
                section = self.stored_section[x][y][0]
                boundaries = self.__get_section_boundary__(x, y, section_size,
                                                           width, height)
                if self.last_saved_section[x][y][0] is not None:
                    last_good_section = self.last_saved_section[x][y][0]
                else:
                    blank[boundaries[0]:boundaries[1],
                          boundaries[2]:boundaries[3]] = section
                    self.last_saved_section[x][y][0] = section
                    if self.debug:
                        cv.putText(self.debug_image, "{}".format(seen),
                                   (boundaries[2]+font_offset,
                                    boundaries[0]+font_offset),
                                   cv.FONT_HERSHEY_COMPLEX, 0.5,
                                   (255, 0, 0), 1)
                    continue
                if blank[boundaries[0]:boundaries[1],
                         boundaries[2]:boundaries[3]].shape != section.shape:
                    continue
                if seen >= self.section_threshold:
                    blank[boundaries[0]:boundaries[1],
                          boundaries[2]:boundaries[3]] = section
                    if self.debug:
                        cv.putText(self.debug_image, "{}".format(seen),
                                   (boundaries[2]+font_offset,
                                    boundaries[0]+font_offset),
                                   cv.FONT_HERSHEY_COMPLEX, 0.5,
                                   (0, 255, 0), 1)
                elif (seen < self.section_threshold and
                      seen > self.reject_threshold):
                    sim = self.__compare_hashes__(section, last_good_section)
                    if sim <= self.similarity:
                        blank[boundaries[0]:boundaries[1],
                              boundaries[2]:boundaries[3]] = section
                    if self.debug:
                        cv.putText(self.debug_image, "{}".format(seen),
                                   (boundaries[2]+font_offset,
                                    boundaries[0]+font_offset),
                                   cv.FONT_HERSHEY_COMPLEX, 0.5,
                                   (0, 0, 255), 1)

        if self.debug:
            cv.imshow("stitched image", blank)
            cv.imshow("board", self.debug_image)
        return blank

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
        if self.debug is True:
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
        if self.debug is True:
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
         sim=0.60, compare_func='dhash', section_threshold=15,
         section_reject=10, overlap=10, grid=(16, 16), eval_file=None,
         dbg=True):
    board = BoardSearcher(n_slides=slide_number, frame_counter=start_frame,
                          save_interval=check_interval, similarity=sim,
                          compare_function=compare_func,
                          section_thresh=section_threshold,
                          section_rjct=section_reject, section_overlap=overlap,
                          grid_size=grid, eval_filename=eval_file, debug=dbg)
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

    parser.add_argument('-n', '--slide-number', default=0,
                        type=int, metavar='N', dest='slide_number',
                        help='''
                             On which slide number to start. Slide with this
                             number will also be loaded. (default: 0)
                             ''')

    parser.add_argument('-i', '--save-interval', default=30,
                        type=int, metavar='I', dest='save_interval',
                        help='''
                             How many frames have to pass to perform check if
                             board changed and possibly save slide from video.
                             (default: 30)
                             ''')

    parser.add_argument('-s', '--similarity', default=0.60,
                        type=float, metavar='S', dest='similarity',
                        help='''
                             On have many percent frames have to be similar to
                             skip saving of slide. (default: 0.60)
                             ''')

    parser.add_argument('-f', '--start-frame', default=0,
                        type=int, metavar='F', dest='start_frame',
                        help='''
                             On which frame to start processing the video.
                             (default: 0)
                             ''')

    parser.add_argument('-c', '--compare-function', default='dhash',
                        dest='cfunc',
                        choices=['dhash', 'phash', 'ahash', 'orb'],
                        help='''
                             Specify a compare function which is going to
                             be used to perform similarity chceck between
                             last saved slide and currently proccesed one.
                             (default: dhash)
                             ''')

    parser.add_argument('-t', '--section-threshold', default=15,
                        type=float, metavar='T', dest='section_thresh',
                        help='''
                             Threshold value for how many times we have to
                             see individual section of a board to definetely
                             accept it. (default: 15)
                             ''')

    parser.add_argument('-r', '--section-reject', default=10,
                        type=float, metavar='T', dest='section_reject',
                        help='''
                             Threshold value for how many times we have to
                             see individual section of a board to definetely
                             reject it. (default: 10)
                             ''')

    parser.add_argument('--grid-width', default=16,
                        type=float, metavar='T', dest='grid_width',
                        help='''
                             How many sections we should split image of a board
                             along x axis. (default: 16)
                             ''')

    parser.add_argument('--section-overlap', default=10,
                        type=int, metavar='O', dest='sec_over',
                        help='''
                             How much such individual section overlap.
                             (default: 16)
                             ''')

    parser.add_argument('--grid-height', default=16,
                        type=float, metavar='T', dest='grid_height',
                        help='''
                             How many sections we should split image of a board
                             along y axis. (default: 16)
                             ''')

    parser.add_argument('-e', '--eval', type=str, metavar='filename',
                        dest='eval_file',
                        help='''
                             Specify csv file with meta data of given video
                             or image.
                             ''')

    parser.add_argument('-d', '--debug', action='store_false', dest='debug',
                        help='''
                             Turns off debuging features. (default: turned ON)
                             ''')

    args = parser.parse_args()
    main(args.filename, slide_number=args.slide_number,
         start_frame=args.start_frame, check_interval=args.save_interval,
         sim=args.similarity, compare_func=args.cfunc,
         section_threshold=args.section_thresh,
         section_reject=args.section_reject, overlap=args.sec_over,
         grid=(args.grid_width, args.grid_height), eval_file=args.eval_file,
         dbg=args.debug)
