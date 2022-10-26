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
# from cv2 import cv2
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import load_model
# import time



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



class AFCMapPublisher:

  def __init__(self, isVis=True, conf_thr=0.1, freq=15) -> None:
    # =========== params ===========
    self.IMG_WIDTH = 128            # image width for inference
    self.IMG_HEIGHT = 128           # image height for inference
    self.IMG_DISP_WIDTH = 640       # image width for display
    self.IMG_DISP_HEIGHT = 480      # image height for display
    self.conf_thr = conf_thr        # threshold to binarize the mask
    self.isVis = isVis              # turn on/off rt visualization
    # ==============================

    # ========== pre-allocation ==========
    self.bmode_raw = np.zeros((self.IMG_DISP_HEIGHT, self.IMG_DISP_WIDTH), dtype=np.uint8)
    self.bmode = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH), dtype=np.uint8)
    self.AFC_bright = np.zeros((self.IMG_DISP_HEIGHT, self.IMG_DISP_WIDTH), np.float32)
    self.AFC_dark = np.zeros((self.IMG_DISP_HEIGHT, self.IMG_DISP_WIDTH), np.float32)
    # ====================================

    # ========== initialization ==========
    print("tensorflow version:", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # load models
    bright_model_path = 'C:/Users/abhip/Desktop/School/Robotic LUS/model_creation/bright_features_v2'
    dark_model_path = 'C:/Users/abhip/Desktop/School/Robotic LUS/model_creation/dark_features_v2'
    model_eval = {'bce_dice_loss': self.bce_dice_loss, 'dice_coeff': self.dice_coeff}
    self.dark_model = load_model(dark_model_path, custom_objects=model_eval)
    self.bright_model = load_model(bright_model_path, custom_objects=model_eval)
    # print(dark_model.summary(), '\n', bright_model.summary())

    # ========== Tensorflow Lite initialization ==========
    dark_tflite_model_path = "C:/Users/abhip/Desktop/School/Robotic LUS/darkFeaturesV2.tflite"
    bright_tflite_model_path = "C:/Users/abhip/Desktop/School/Robotic LUS/brightFeaturesV2.tflite"

    #load tflite model and allocate tensors
    self.interpreter_dark = tf.lite.Interpreter(model_path=dark_tflite_model_path)
    self.interpreter_dark.allocate_tensors()

    self.interpreter_bright = tf.lite.Interpreter(model_path=bright_tflite_model_path)
    self.interpreter_bright.allocate_tensors()

    #get input and output tensors
    self.input_details_dark = self.interpreter_dark.get_input_details()
    self.output_details_dark = self.interpreter_dark.get_output_details()

    self.input_details_bright = self.interpreter_bright.get_input_details()
    self.output_details_bright = self.interpreter_bright.get_output_details()

    print('initialized')
    # ====================================

#   def timer(func):
#     def wrapper(*args, **kwargs):
#       t_start = time.perf_counter()
#       ret = func(*args, **kwargs)
#       t_stop = time.perf_counter()
#       print(f'total time elapsed: {t_stop-t_start}')
#       return ret
    
#     return wrapper

  def clarius_us_cb(self, image) -> None:
    bmode_msg = image
    self.bmode_raw = cv2.cvtColor(bmode_msg, cv2.COLOR_BGR2RGB) #converting raw to model input image in terms of color channels
    # self.bmode_raw = cv2.cvtColor(bmode_msg, cv2.COLOR_BGR2GRAY) #converting raw to model input image in terms of color channels
    img_rgb_resize = cv2.resize(self.bmode_raw, (self.IMG_WIDTH, self.IMG_HEIGHT))
    self.bmode = np.expand_dims(img_rgb_resize, axis=0)

  def dice_coeff(self, y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / \
        (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

  def dice_loss(self, y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

  def bce_dice_loss(self, y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
  
  
  def get_score2(self, map, bbox):
    # calculate accumulative confidence value inside bounding box
    # bbox = [[x1,y1,w1,h1],[x2,y2,w2,h2], ...]
    
    n = len(bbox)
    score = [0 for i in range(n)]
    pixel = [0 for i in range(n)]
    final_score = [0 for i in range(n)]

    # print(n)
    for i in range(len(bbox)):
      # for n in range(np.shape(bbox[i])[0]):
      # print(bbox[i][:])
      for r in range(bbox[i][1], (bbox[i][1]+bbox[i][3])):
        for c in range(bbox[i][0], (bbox[i][0]+bbox[i][2])):
          if map[r][c] > 0.01:
            # print(map[r][c])
            score[i] += map[r][c]
            pixel[i] += 1
        if score[i] > 0:
          final_score[i] = score[i]/pixel[i]

    return final_score


  def get_bounding_box(self, msk, map, imgOverlay=None):
    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areaMin = 10
    bbox = []
    for cnt, _ in enumerate(contours):
      area = cv2.contourArea(contours[cnt])
      if area > areaMin:
        peri = cv2.arcLength(contours[cnt], True)
        approx = cv2.approxPolyDP(contours[cnt], 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if x != 0 and y != 0 and w != 0 and h != 0:
          bbox.append([x, y, w, h])
          if imgOverlay is not None:
            # cv2.drawContours(imgOverlay, contours[cnt], -1, (255, 0, 255), thickness=1)
            cv2.rectangle(imgOverlay, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
            
            #Printing Confidence Eval for each feature
            # cv2.putText(imgOverlay, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 0), 2)
            # print(self.get_score2(map, bbox)[0])
            confidence_eval_score = self.get_score2(map, bbox)[0] #converting single list to int
            # print(100*confidence_eval_score)
            cv2.putText(imgOverlay, "Eval: " + str(round(100*confidence_eval_score,2)) + "%", (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 0), 2) 
            
    return np.array(bbox), imgOverlay

  def pub_AFC_dark(self, imgOverlay=None):
    # AFC_dark_temp = self.dark_model.predict(self.bmode, verbose=0)

    #tensorflow lite prediction steps
    input_data = self.bmode.astype(np.float32)
    self.interpreter_dark.set_tensor(self.input_details_dark[0]['index'], input_data)
    self.interpreter_dark.invoke()
    AFC_dark_temp = self.interpreter_dark.get_tensor(self.output_details_dark[0]['index'])

    AFC_dark_temp = np.squeeze(AFC_dark_temp, axis=0)
    self.AFC_dark = cv2.resize(AFC_dark_temp, (self.IMG_DISP_WIDTH, self.IMG_DISP_HEIGHT))
    # self.AFC_dark_pub.publish(CvBridge().cv2_to_imgmsg(self.AFC_dark, encoding="passthrough"))
    self.msk_dark = (self.AFC_dark > self.conf_thr).astype(np.uint8)
    if imgOverlay is not None:
      overlay = cv2.cvtColor(255*self.AFC_dark, cv2.COLOR_GRAY2RGB).astype(np.uint8)
      overlay_new = cv2.applyColorMap(overlay, cv2.COLORMAP_OCEAN) #apply color overlay
      imgOverlay = cv2.addWeighted(imgOverlay, 0.6, overlay_new, 0.4, 1.0)
      return imgOverlay


  #@timer #Timer broke the pub_AFC_bright function for some reason

  def pub_AFC_bright(self, imgOverlay=None):
    # ========== manipulate input dim ==========
    bmode_resize = cv2.resize(self.bmode_raw, (1024, 768))  # restore to Clarius export dim
    bmode_resize = bmode_resize[:300, 238:880, :]
    bmode_resize = cv2.resize(bmode_resize, (self.IMG_WIDTH, self.IMG_HEIGHT))
    bmode_resize = np.expand_dims(bmode_resize, axis=0)
    # AFC_bright_out = self.bright_model.predict(bmode_resize, verbose=0)

    #tensorflow lite prediction steps
    input_data = bmode_resize.astype(np.float32) #need to convert input data into float before passing into lite model
    self.interpreter_bright.set_tensor(self.input_details_bright[0]['index'], input_data)
    self.interpreter_bright.invoke()
    AFC_bright_out = self.interpreter_bright.get_tensor(self.output_details_bright[0]['index'])

    # ========== manipulate output dim ==========
    AFC_bright_out = np.squeeze(AFC_bright_out, axis=0)
    AFC_bright_out = cv2.resize(AFC_bright_out, (880-238, 300))
    AFC_bright_temp = np.zeros((768, 1024), dtype=np.float32)
    AFC_bright_temp[:300, 238:880] = AFC_bright_out
    self.AFC_bright = cv2.resize(AFC_bright_temp, (self.IMG_DISP_WIDTH, self.IMG_DISP_HEIGHT))
    self.AFC_bright -= self.AFC_dark  # substract rib shadow to reduce false positive
    self.AFC_bright[self.AFC_bright < 0] = 0
    # ===== publish to topic
    # self.AFC_bright_pub.publish(CvBridge().cv2_to_imgmsg(self.AFC_bright, encoding="passthrough"))
    self.msk_bright = (self.AFC_bright > self.conf_thr).astype(np.uint8)
    if imgOverlay is not None:
      overlay = cv2.cvtColor(255*self.AFC_bright, cv2.COLOR_GRAY2RGB).astype(np.uint8)
      overlay_new = cv2.applyColorMap(overlay, cv2.COLORMAP_HOT) #apply color overlay
      imgOverlay = cv2.addWeighted(imgOverlay, 0.4, overlay_new, 0.6, 1.0)
      return imgOverlay


  def vis(self, left=None, right=None) -> None:

    self.bmode_raw = cv2.resize(self.bmode_raw, (self.IMG_DISP_WIDTH, self.IMG_DISP_HEIGHT))

    if left is not None and right is not None:
      stack = np.hstack((left, right))
    else:
      top = np.hstack((self.bmode_raw, self.bmode_raw))
      
      dark_segmentation = cv2.cvtColor(255*self.AFC_dark, cv2.COLOR_GRAY2RGB).astype(np.uint8) 
      bright_segmentation = cv2.cvtColor(255*self.AFC_bright, cv2.COLOR_GRAY2RGB).astype(np.uint8)

      #applying color overlay     
      new_bright_segmentation = cv2.applyColorMap(bright_segmentation, cv2.COLORMAP_HOT)
      new_dark_segmentation = cv2.applyColorMap(dark_segmentation, cv2.COLORMAP_OCEAN)

      bottom = np.hstack((new_bright_segmentation,new_dark_segmentation))
      stack = cv2.addWeighted(top, 0.4, bottom, 0.6, 1.0)
       
    cv2.imshow("stacked display", stack)



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

	AFC_map = AFCMapPublisher()

	# loop
	cv2.namedWindow('clarius',cv2.WINDOW_AUTOSIZE)
	isVisualize = True
	try:
		while True:
			if processed is not None:
				if isVisualize:
					# cv2.imshow('clarius', processed)
					
					# #Segmentations With Bounding Box + Confidence Eval
					AFC_map.clarius_us_cb(processed)
					# AFC_map.bmode_raw = cv2.cvtColor(AFC_map.bmode_raw, cv2.COLOR_BGR2GRAY)
					AFC_map.bmode_raw = cv2.resize(AFC_map.bmode_raw, (AFC_map.IMG_DISP_WIDTH, AFC_map.IMG_DISP_HEIGHT))
					overlay_bright = AFC_map.pub_AFC_bright(imgOverlay=AFC_map.bmode_raw.copy())
					overlay_dark = AFC_map.pub_AFC_dark(imgOverlay=AFC_map.bmode_raw.copy())
					bbox_dark, overlay_bbox_dark = AFC_map.get_bounding_box(AFC_map.msk_dark, AFC_map.AFC_dark, imgOverlay=overlay_dark)
					bbox_bright, overlay_bbox_bright = AFC_map.get_bounding_box(AFC_map.msk_bright, AFC_map.AFC_bright, imgOverlay=overlay_bright)
					AFC_map.vis(overlay_bbox_bright, overlay_bbox_dark)

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
