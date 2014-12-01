#!/usr/bin/env python
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

bins = np.arange(256).reshape(256,1)

#def calculateHistogram(image, nameOfWindow="Histogram"):
#	test = np.zeros((300,256,3))
#	color = [(255,0,0), (0,255,0), (0,0,255)]
#	for ch, col in enumerate(color):
#		hist = cv.calcHist([image], [ch], None, [256], [0, 256])
#		cv.normalize(hist, hist, 0, 256, cv.NORM_MINMAX)
#		h = np.int32(np.around(hist))
#		p = np.int32(np.column_stack((bins, h)))
#		cv.polylines(test, [p],False, col)
#	histogram = np.flipud(test)
#	cv.imshow(nameOfWindow, histogram)


def calculateHistogram(image, nameOfWindow="Histogram"):
	(width, height, _) = image.shape
	vlines = [0 for _ in range(width)]
	hlines = [0 for _ in range(height)]

	for x in range(width):
		for y in range(height):
			if 100 < image[x][y][2] < 256:
				hlines[y] += 1
				vlines[x] += 1	 
	
	plt.figure()
	plt.plot(hlines)
	plt.figure()
	plt.plot(vlines)
	plt.show()
########test = np.zeros((300,256,3))
########color = [(0,255,0)]
########hist = cv.calcHist([image], [0], None, [256], [0, 256])
########cv.normalize(hist, hist, 0, 256, cv.NORM_MINMAX)
########h = np.int32(np.around(hist))
########p = np.int32(np.column_stack((bins, h)))
########cv.polylines(test, [p], False, (0, 255, 0))
########histogram = np.flipud(test)
########print histogram.shape
########cv.imshow(nameOfWindow, histogram)

def cartoonify(image):
	out = image.copy()
	out = cv.medianBlur(image, 5)
	calculateHistogram(out, "Project Histogram")
	return out

def main(inputFile):
	im = cv.imread(inputFile, cv.CV_LOAD_IMAGE_COLOR)
	#create output image
	out = im.copy()
	#cartoon-ify image
	out = cartoonify(im)
	#show image
	cv.imshow("project", out)
	cv.waitKey(0)
	cv.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv[1])
