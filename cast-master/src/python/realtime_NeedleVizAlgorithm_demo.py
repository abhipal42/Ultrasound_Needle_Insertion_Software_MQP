#!/usr/bin/env python
'''
publish US image stream using OpenCV
'''
import pycast
import argparse
import numpy as np
import cv2
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import Qt, Slot, Signal


import numpy as np


global processed
processed = None

# called when a new processed image is streamed
# @param image the scan-converted image data
# @param width width of the image in pixels
# @param height height of the image in pixels
# @param bpp bits per pixel
# @param micronsPerPixel microns per pixel
# @param timestamp the image timestamp in nanoseconds


def newProcessedImage(image, width, height, bpp, micronsPerPixel, timestamp, imu):
	global processed
	print("new image (sc): {0}, {1}x{2} @ {3} bpp, {4:.2f} um/px, imu: {5} pts".
				format(timestamp, width, height, bpp, micronsPerPixel, len(imu)))
	img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
	# img.save('processed.jpg')
	ptr = img.constBits()
	processed = np.array(ptr).reshape(height, width, 4)  # Copies the data
	return


# called when a new raw image is streamed
# @param image the raw pre scan-converted image data, uncompressed 8-bit or jpeg compressed
# @param lines number of lines in the data
# @param samples number of samples in the data
# @param bps bits per sample
# @param axial microns per sample
# @param lateral microns per line
# @param timestamp the image timestamp in nanoseconds
# @param jpg jpeg compression size if the data is in jpeg format
def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg):
	return


# called when a new spectrum image is streamed
# @param image the spectral image
# @param lines number of lines in the spectrum
# @param samples number of samples per line
# @param bps bits per sample
# @param period line repetition period of spectrum
# @param micronsPerSample microns per sample for an m spectrum
# @param velocityPerSample velocity per sample for a pw spectrum
# @param pw flag that is true for a pw spectrum, false for an m spectrum
def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
	return


# called when freeze state changes
# @param frozen the freeze state
def freezeFn(frozen):
	if frozen:
			print("imaging frozen")
	else:
			print("imaging running")
	return


# called when a button is pressed
# @param button the button that was pressed
# @param clicks number of clicks performed
def buttonsFn(button, clicks):
	print("button pressed: {0}, clicks: {1}".format(button, clicks))
	return


class NeedleVisualization:

  def __init__(self):
    self.frameWidth = 440
    self.frameHeight = 440

    # ROI parameters
    self.rstart = 140  # previously 94
    self.rend = 348
    self. cstart = 195  # previously 166
    self. cend = 235  # previously 275

  def insert_frame(self, frame):
    self.frame = frame

    # Initial Preprocessing
    self.resized_frame = cv2.resize(
        self.frame, (self.frameWidth, self.frameHeight))
    self.resized_frame = cv2.cvtColor(
        self.resized_frame, cv2.COLOR_RGB2GRAY)

  def line_creation(self, source_image, overlay):
    lines = cv2.HoughLinesP(source_image, rho=6, theta=np.pi / 2,
                            threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=4)

    overlay_image = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    houghline = overlay_image.copy()

    if lines is not None:
        for line in lines:

            # represents the top left corner of image
            start_point = (line[0][0], line[0][1])
            # represents the bottom right corner of image
            end_point = (line[0][2], line[0][3])
            color = (0, 255, 0)  # Green color in BGR
            thickness = 2  # Line thickness

            cv2.line(houghline, start_point, end_point, color, thickness)

    return houghline

  def ROI_creation(self, source_image, row_start, row_end, col_start, col_end):
    # old one was [94:348, 166:275]
    ROI_frame = source_image[row_start:row_end, col_start:col_end]
    ROI_image = np.zeros_like(source_image)
    x = row_start
    y = col_start
    for i in range(0, row_end-row_start):
        for j in range(0, col_end-col_start):
            if ROI_frame[i][j] != 0:
                ROI_image[x + i, y + j] = ROI_frame[i, j]
    return ROI_image

  def detect_needle_line(self):
    # Achieving desired region of interest within Raw Frame
    ##############################################################
    ROI_image = self.ROI_creation(
        self.resized_frame, self.rstart, self.rend, self.cstart, self.cend)
    ##############################################################

    # Applying Paper Algorithm Filters
    #############################################################
    # gabor_filter = cv2.getGaborKernel((6,6), sigma=0.5, theta=0, lambd=0.5, gamma=0.8, psi=0, ktype=cv2.CV_32F)
    gabor_filter = cv2.getGaborKernel((3, 3), sigma=0.95, theta=0, lambd=5, gamma=0.8, psi=0, ktype=cv2.CV_32F)
    # gabor_filter = cv2.getGaborKernel((3,3), sigma=0.5, theta=0, lambd=30, gamma=0.8, psi=0, ktype=cv2.CV_32F)

    gabor_output = cv2.filter2D(ROI_image, -1, gabor_filter)

    # Binarized image is divided into grids for needle axis localization.
    # - Median filter
    median_filter = cv2.medianBlur(gabor_output, 7)
    # - automatic thresholding
    threshold = cv2.threshold(
        median_filter, 250, 255, cv2.THRESH_BINARY)[1]
    # - morphological operations
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(threshold, element)
    dilated = cv2.dilate(eroded, element)
    #############################################################

    # Hough Line Transforms
    #############################################################
    houghline = self.line_creation(dilated, self.resized_frame)
    #############################################################
    return houghline

  def show_needle_viz(self, houghline):
    resized_frame = cv2.cvtColor(self.resized_frame, cv2.COLOR_GRAY2BGR)
    overlay = cv2.cvtColor(houghline, cv2.COLOR_RGB2BGR)
    stack = np.hstack((resized_frame, overlay))
    cv2.imshow("stacked", stack)

# main function
def main():
	global processed
	parser = argparse.ArgumentParser()
	parser.add_argument('--address', '-a', dest='ip',
											help='ip address of probe.', required=True)
	parser.add_argument('--port', '-p', dest='port', type=int,
											help='port of the probe', required=True)
	parser.add_argument('--width', '-w', dest='width',
											type=int, help='image output width in pixels')
	parser.add_argument('--height', '-ht', dest='height',
											type=int, help='image output height in pixels')
	parser.set_defaults(ip=None)
	parser.set_defaults(port=None)
	parser.set_defaults(width=640)
	parser.set_defaults(height=480)
	args = parser.parse_args()

	if not args.ip or not args.port or args.port < 0:
		print("one or more arguments are invalid")
		parser.print_usage()
		return

	path = './'

	# initialize
	cast = pycast.Caster(newProcessedImage, newRawImage,
												newSpectrumImage, freezeFn, buttonsFn)
	ret = cast.init(path, args.width, args.height)
	if ret:
		print("initialization succeeded")
		ret = cast.connect(args.ip, args.port)
		if ret:
			print("connected to {0} on port {1}".format(args.ip, args.port))
		else:
			print("connection failed")
			cast.destroy()
			return
	else:
		print("initialization failed")
		return

	needle_viz = NeedleVisualization()

	# loop
	cv2.namedWindow('clarius',cv2.WINDOW_AUTOSIZE)
	isVisualize = True
	try:
		while True:
			if processed is not None:
				if isVisualize:
					# cv2.imshow('clarius', processed)
					needle_viz.insert_frame(processed)
					houghline = needle_viz.detect_needle_line()
					needle_viz.show_needle_viz(houghline)
                    
			key = cv2.waitKey(10)
			if key == ord('q'):
				break
			elif key == ord('s'):
				if processed is not None:
					print('save image')
					cv2.imwrite('./BScan.jpg', processed)
				else:
					print('empty image')
	except KeyboardInterrupt:
		print('terminated by user')

	cv2.destroyAllWindows()
	cast.destroy()

if __name__ == '__main__':
	main()
