#!/usr/bin/env python
import sys
import cv2 as cv
import numpy as np
import ntpath
import functools
import time


def timeit(func):
    """
    Poors man profiling function to
    measure time it takes to finish function.
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
    def __init__(self, lo_thresh=100, hi_thresh=255):
        self.lo_thresh = lo_thresh
        self.hi_thresh = hi_thresh

    def __get_index(self, arr):
        """
        Function for returning min, max tuples from
        our arrays.
        """
        max = np.argmax(arr)
        min = np.argmin(arr)
        return (min, max)

    def calc_hist(self, image):
        """
        Function gets rid of low intensity pixels values in
        color spectrum and sets them to 0. Then it calculates
        histogram of green color in x and y axis from image.
        Then we calculate 1-th order difference on this
        histogram and return min and max value indexes.
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
        y = np.diff(map(lambda *row: list(row), *vlines)[0])
        x = np.diff(hlines[0])
        return (self.__get_index(x), self.__get_index(y))

    def get_table_metadata(self, image):
        """
        Function for returning coordinates of
        a boards corners. It distorts image with
        median blur to remove noise from image and
        unify colors a bit.
        """
        process_image = cv.medianBlur(image, 5)
        return self.calc_hist(process_image)

    def write_video_metadata(self, video, fd):
        """
        Sets up a video capture and loops through
        all of the frames. Pulls frame metadata
        from every frame and stores it into an array.
        After we are done with a video we flush these
        values to a file with frame index.
        """
        vid = cv.VideoCapture(video)
        if vid is None:
            print "Couldn't load video {}".format(video)
            return
        frames = []
        while(vid.isOpened()):
            ret, frame = vid.read()
            if not ret:
                break
            (x, y) = self.get_table_metadata(frame)
            frames.append((x, y))
        for i, frame in enumerate(frames):
            fd.write("{}: {} {}\n".format(i, frame[0], frame[1]))
        vid.release()

    def __get_file_name(self, path):
        """
        Returns only a file name from path.
        """
        head, tail = ntpath.split(path)
        name = tail or ntpath.basename(head)
        return name.split('.')[0]

    def write_image_metadata(self, image, fd):
        """
        Reads given image and gets his metadata which
        are written to a file.
        """
        im = cv.imread(image, cv.CV_LOAD_IMAGE_COLOR)
        if im is None:
            print "Couldn't read image file {}".format(image)
            return
        (x, y) = self.get_table_metadata(im)
        fd.write("0: {} {}".format(x, y))

    image_extension_list = ["jpg", "gif", "png"]
    video_extension_list = ["mkv", "wmv", "avi", "mp4"]

    @timeit
    def write_metadata(self, input_file):
        """
        Main function for getting metadata from images or videos.
        It checks if format of a file is supported and then calls
        approperiate function.
        """
        fd = open("{}_matadata".format(self.__get_file_name(input_file)), "w+")
        if(any(input_file[-3:] == i for i in self.video_extension_list)):
            self.write_video_metadata(input_file, fd)
        elif (any(input_file[-3:] == i for i in self.image_extension_list)):
            self.write_image_metadata(input_file, fd)
        else:
            print "Unrecognized file format"
        fd.close()


def main(input_file):
    toolkit = BoardToolkit()
    toolkit.write_metadata(input_file)

if __name__ == '__main__':
    main(sys.argv[1])
