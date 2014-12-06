#!/usr/bin/env python
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Returns index of highest and lowest
# element in a array
def getMinMaxIndex(arr):
	max = arr[0]
	min = arr[0]
	maxi = 0
	mini = 0
	for i in range(arr.shape[0]):
		if max < arr[i]:
			max = arr[i]
			maxi = i
		if min > arr[i]:
			min= arr[i]
			mini = i
	return (maxi, mini)

# Gets values of green in a image, calculates difference and
# returns pixels with highest and lowest values
def calculateHistogram(image):
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
	return (getMinMaxIndex(x), getMinMaxIndex(y))

# Makes a copy of input image, applies a median blur
# to cartoon-ify the image. It gets rid of noise and
# not wanted colors. Calculates histogram of green color 
# and draw edges of a board
def cartoonify(image):
	out = image.copy()
	out = cv.medianBlur(image, 5)
	(x, y) = calculateHistogram(out)
	#draw edges of a board
	cv.circle(image, (x[1], y[0]), 5, (0, 0 , 255), -1)
	cv.circle(image, (x[1], y[1]), 5, (0, 0 , 255), -1)
	cv.circle(image, (x[0], y[0]), 5, (0, 0 , 255), -1)
	cv.circle(image, (x[0], y[1]), 5, (0, 0 , 255), -1)

def main(inputFile):
	im = cv.imread(inputFile, cv.CV_LOAD_IMAGE_COLOR)
	#cartoon-ify image
	cartoonify(im)
	#show image
	cv.imshow("project", im)
	cv.waitKey(0)
	cv.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv[1])
