import glob
import os
import cv2
import preprocess_image as pre
import numpy as np


def get_vocabulary_size():
	return len(get_vocabulary())

def get_vocabulary():
	vocab = [line for line in open('data/small_vocab.txt')]
	vocab = [x.strip('\n') for x in vocab]
	return vocab

def load_raw_data(load_all = False, num_sample = 400, max_token_length = 30, max_image_size = (60, 200)):

	image_folder = 'data/small/'
	token_sequences = []
	images = []

	included_counter = 0
	examples_counter = 0
	with open ("data/small.formulas.norm.txt", "r") as myfile:
		
		for idx, token_sequence in enumerate(myfile):
			examples_counter += 1
	    	#Check token size:
			token_sequence = token_sequence.rstrip('\n')
			tokens = token_sequence.split()

			file_name = str(idx) + '.png'
			image = cv2.imread(image_folder + file_name)

	    	#print(tokens)
			#if len(tokens) < max_token_length and image.shape[0] < max_image_size[0] and image.shape[1] < max_image_size[1]:
			token_sequences.append('**start** ' + token_sequence + ' **end**')



			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Grey scale
			images.append(image)

			#included_counter += 1

			if (load_all == False and idx == num_sample):
				break

			

	#print(str(included_counter) + "out of " + str(examples_counter) + " image/token-list pairs loaded")
        
	return images, token_sequences




from matplotlib import pyplot as plt

def view_image(image, normalized = False):
	if normalized:
		image = image * 255
		#print(image)
		image = image.astype(np.uint8)

	#print(image.shape)
	image = np.squeeze(image)
	#print(image)
	plt.imshow(image, cmap='gray')
	plt.show()



def preprocess_images(images):

	factor = 0.7
	encoder_input = pre.down_sample_flexible(images, factor)
	encoder_input = pre.pad_images(encoder_input)

	#print("Encoder_input shape:", encoder_input.shape)

	encoder_input = pre.normalize_images(encoder_input)

	# Add dimension for TensorFlow to work properly. (corresponding to the rgb channels even thoug we only do black & white)
	encoder_input = encoder_input.reshape(encoder_input.shape[0], encoder_input.shape[1], encoder_input.shape[2], 1)
	#print(max_shape)
	return encoder_input


def load_data():
	images, token_sequences = load_raw_data(load_all = True, 
									num_sample = 200, 
									max_token_length = 30,
									max_image_size = (50, 200))

	#num_sample = 15
	#images = images[:num_sample]
	#token_sequences = token_sequences[:num_sample]

	images = preprocess_images(images)

	return images, token_sequences

