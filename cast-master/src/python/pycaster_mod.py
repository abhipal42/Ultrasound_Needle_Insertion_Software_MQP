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

	# loop
	cv2.namedWindow('clarius',cv2.WINDOW_AUTOSIZE)
	isVisualize = True
	try:
		while True:
			if processed is not None:
				if isVisualize:
					cv2.imshow('clarius', processed)
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
