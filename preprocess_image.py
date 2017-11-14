import numpy as np
import math
import cv2

def get_max_shape(images):

	max_height = 0
	max_width = 0
	print("Getting max shape")

	for image in images:

		if image.shape[0] > max_height:
			max_height = image.shape[0]

		if image.shape[1] > max_width:
			max_width = image.shape[1]


	return [max_height, max_width]


def normalize_images(images):

	images = images.astype(np.float32)
	images = np.multiply(images, 1.0 / 255.0)

	return images

def down_sample(images, factor): 
	target_h = int(math.floor(float(images[0].shape[0]) * factor))
	target_w = int(math.floor(float(images[0].shape[1]) * factor))
	num_images = len(images)
	down_sampled_images = np.ones((num_images, target_h, target_w)) * 255

	for idx, image in enumerate(images):
		
		im = image

		#Downsample
		im = cv2.resize(im, (0, 0), fx = factor, fy=factor, interpolation = cv2.INTER_AREA) #cv2.INTER_LINEAR


		down_sampled_images[idx, :, :] = im

	return down_sampled_images


def down_sample_flexible(images, factor): 
	new_images = []

	for image in images:
		new_image = cv2.resize(image, (0, 0), fx = factor, fy=factor, interpolation = cv2.INTER_AREA) #cv2.INTER_LINEAR
		new_images.append(new_image)

	
	return new_images


def pad_images(images, target_shape = None):


	if (target_shape == None):
		max_height, max_width = get_max_shape(images)

	num_images = len(images)

	padded_images = np.ones((num_images, max_height, max_width)) * 255

	for idx, image in enumerate(images):

		h = image.shape[0]
		w = image.shape[1]

		padded_images[idx, :h, :w] = image


	return padded_images