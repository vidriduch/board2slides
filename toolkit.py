#!/usr/bin/env python
from __future__ import print_function
import sys
import cv2 as cv
import numpy as np
import ntpath
import functools
import time
import imghdr
import csv
import argparse
import phash
import os


def timeit(func):
    """
    Poor man's profiling function to
    measure time it takes to finish function
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


class BoardToolkit:

    video_extension_list = ("mkv", "wmv", "avi", "mp4")

    def __init__(self, lo_thresh=100, hi_thresh=255):
        """
        Args:
            lo_thresh(int): bottom threshold boundary
            hi_thresh(int): top threshold boundary
        """
        self.lo_thresh = lo_thresh
        self.hi_thresh = hi_thresh

    def __get_extremes(self, arr):
        """
        Function for returning extremes from array

        Args:
            arr(array): Array to search

        Returns:
            tuple: minimum and maximum from array
        """
        max = np.argmax(arr)
        min = np.argmin(arr)
        return (min, max)

    def get_board_boundaries(self, image):
        """
        Function gets rid of low intensity pixel values in
        color spectrum and sets them to 0. Then it calculates
        histogram of green color in x and y axis from image.
        Then we calculate 1-th order difference on this
        histogram

        Args:
            image(numpy.ndarray): Loaded image

        Retunrs:
            Returns extremes from calculated histogram
        """
        (width, height, _) = image.shape
        vlines = np.zeros(width)
        hlines = np.zeros(height)
        image[:, :, 1] = cv.inRange(image[:, :, 1],
                                    self.lo_thresh,
                                    self.hi_thresh)
        vlines = cv.reduce(image[:, :, 1], 1,
                           cv.cv.CV_REDUCE_SUM,
                           dtype=cv.CV_32S)
        hlines = cv.reduce(image[:, :, 1], 0,
                           cv.cv.CV_REDUCE_SUM,
                           dtype=cv.CV_32S)
        vlines = map(lambda *row: list(row), *vlines)
        y = np.diff(vlines[0])
        x = np.diff(hlines[0])
        return (self.__get_extremes(x), self.__get_extremes(y))

    def get_table_metadata(self, image):
        """
        Function distorts image with median blur to
        remove noise from image and unify colors a bit

        Args:
            image(numpy.ndarray): Loaded image

        Returns:
            Board's corner coordinates
        """
        process_image = cv.medianBlur(image, 5)
        return self.get_board_boundaries(process_image)

    def __get_file_name(self, path):
        """
        Args:
            path(string): Path to file

        Returns:
            string: File name extracted from path
        """
        head, tail = ntpath.split(path)
        name = tail or ntpath.basename(head)
        return name.split('.')[0]

    def check_change(self, frame):
        if cv.imread("toolkit_old.png", cv.CV_LOAD_IMAGE_COLOR) is None:
            cv.imwrite("toolkit_old.png", frame)
        cv.imwrite("toolkit_new.png", frame)
        digest1 = phash.image_digest('toolkit_old.png', 1.0, 1.0, 180)
        digest2 = phash.image_digest('toolkit_new.png', 1.0, 1.0, 180)
        cross_corr = phash.cross_correlation(digest1, digest2)
        if cross_corr < 0.40:
            cv.imwrite("toolkit_old.png", frame)
            return True
        else:
            return False

    def write_video_metadata(self, video, meta_file):
        """
        Sets up a video capture and loops through all of the
        frames. Pulls frame metadata from every frame and stores
        it into an array. After we are done with a video we flush these
        values to a file

        Args:
            video(string): Name of video file to load
            meta_file(string): Name of file where metadata will be stored
        """
        vid = cv.VideoCapture(video)
        if vid is None:
            print("Couldn't load video {}".format(video), file=sys.stderr)
            return
        frames = []
        i = 0
        while(vid.isOpened()):
            ret, frame = vid.read()
            if not ret:
                break
            make_slide = False
            if i % 120 == 0:
                if self.check_change(frame):
                    make_slide = True
            (x, y) = self.get_table_metadata(frame)
            frames.append((x, y, make_slide))
            i += 1
        with open(meta_file, "wb+") as csvfile:
            csv_writer = csv.writer(csvfile, lineterminator='\n')
            for i, frame in enumerate(frames):
                x, y, slide = frame
                csv_writer.writerow([x[0], x[1], y[0], y[1]], slide)
        vid.release()
        os.remove("toolkit_old.png")
        os.remove("toolkit_new.png")

    def write_image_metadata(self, image, meta_file):
        """
        Reads given image and gets his metadata which
        are written to a file

        Args:
            image(string): Name of image to load
            meta_file(string): Name of file where metadata will be stored
        """
        im = cv.imread(image, cv.CV_LOAD_IMAGE_COLOR)
        if im is None:
            print("Couldn't read image file {}".format(image), file=sys.stderr)
            return
        (x, y) = self.get_table_metadata(im)
        with open(meta_file, "wb+") as csvfile:
            csv_writer = csv.writer(csvfile, lineterminator='\n')
            csv_writer.writerow([x[0], x[1], y[0], y[1]])

    def write_metadata(self, input_file, output_file=None):
        """
        Main function for getting metadata from images or videos.
        It checks if format of a file is supported and then calls
        approperiate function

        Args:
            input_file(string): Path to file to process
            output_file(string): Path to output file where metadata
                                 will be stored
        """
        if(output_file is None):
            file_name = self.__get_file_name(input_file)
            output_file = "{}_meta.csv".format(file_name)

        try:
            if(input_file.endswith(self.video_extension_list)):
                self.write_video_metadata(input_file, output_file)
            elif (imghdr.what(input_file) is not None):
                self.write_image_metadata(input_file, output_file)
            else:
                print("Unrecognized file format", file=sys.stderr)
        except IOError:
            print("Wrong file or path to file", file=sys.stderr)


def main(input_file, output_file=None):
    toolkit = BoardToolkit()
    for file_name in input_file:
        toolkit.write_metadata(file_name, output_file)

if __name__ == '__main__':
    desc = "Command line toolkit for creating board2slides dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filename', metavar='filename', nargs='+',
                        help='list of videos or images to process')
    parser.add_argument('-o', '--output-file', nargs='?',
                        help='''
                             specify outputfile where metadata will be save
                             (default: basename from input file with _meta
                             sufix)
                             ''')
    args = parser.parse_args()
    main(args.filename, args.output_file)
