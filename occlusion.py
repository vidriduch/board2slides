#!/usr/bin/env python
import sys
import cv2 as cv
import numpy as np


drawing = False
ix, iy = -1, -1
img = None
color = None
flood = False
rec_size = 5


def floodify(img, color, x, y):
    flags = 4
    flags | 255 << 8
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(img, mask, (x, y), color, 0, 255, flags)


# mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing, color, flood, rec_size
    if event == cv.EVENT_LBUTTONDOWN:
        color = (255, 255, 255)
        if flood is True:
            floodify(img, color, x, y)

    if event == cv.EVENT_RBUTTONDOWN:
        color = (0, 0, 0)
        if flood is True:
            floodify(img, color, x, y)

    if event == cv.EVENT_MOUSEMOVE and color is not None:
        cv.rectangle(img, (x-rec_size, y-rec_size),
                          (x+rec_size, y+rec_size), color, -1)

    if event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        flood = False
        color = None


def main(image1, image2, prep_mask=None, thresh=None):
    im1 = cv.imread(image1, cv.CV_LOAD_IMAGE_COLOR)
    im2 = cv.imread(image2, cv.CV_LOAD_IMAGE_COLOR)
    im11 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im22 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    if im11.shape > im22.shape:
        im2 = cv.resize(im2, (im1.shape[1], im1.shape[0]))
        im22 = cv.resize(im22, (im11.shape[1], im11.shape[0]))
    elif im11.shape < im22.shape:
        im1 = cv.resize(im11, (im2.shape[1], im2.shape[0]))
        im11 = cv.resize(im11, (im22.shape[1], im22.shape[0]))
    if prep_mask is None:
        frame_delta = cv.absdiff(im11, im22)
        frame_delta = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1]
        frame_delta = cv.erode(frame_delta, None, iterations=2)

        (cnts, _) = cv.findContours(frame_delta, cv.RETR_TREE,
                                    cv.CHAIN_APPROX_SIMPLE)

        height, width = frame_delta.shape[:2]
        for c in cnts:
            blank = np.zeros(frame_delta.shape[:2], dtype='uint8')
            cv.drawContours(blank, [c], 0, 255, -1)
            mean1 = map(lambda x: float(x),
                        cv.mean(im1, mask=blank))
            mean2 = map(lambda x: float(x),
                        cv.mean(im2, mask=blank))
            distance = cv.norm(np.array(mean1), np.array(mean2), cv.NORM_L2)
            if distance <= thresh:
                cv.drawContours(frame_delta, [c], 0, 0, -1)
            else:
                cv.drawContours(frame_delta, [c], 0, 255, -1)
    else:
        frame_delta = cv.imread(prep_mask, cv.CV_LOAD_IMAGE_COLOR)
        frame_delta = cv.cvtColor(frame_delta, cv.COLOR_BGR2GRAY)

    cv.namedWindow('image')
    cv.setMouseCallback('image', draw)
    global img, flood, rec_size
    img = frame_delta
    while(1):
        cv.imshow("image", frame_delta)
        cnt = cv.bitwise_and(im2, im2, mask=frame_delta)
        cv.imshow("pre mareka", cnt)
        k = cv.waitKey(1) & 0xFF
        if k == ord('f'):
            flood = True
        if k == ord('+'):
            rec_size += 1
            print(rec_size)
        if k == ord('-'):
            rec_size -= 1
            print(rec_size)
        if k == 27:
            break
    cv.destroyAllWindows()
    cv.imwrite("ground_truth_{}".format(sys.argv[2]), frame_delta)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        thresh = 70
    else:
        thresh = int(sys.argv[3])
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[4], thresh)
    else:
        main(sys.argv[1], sys.argv[2], thresh=thresh)
