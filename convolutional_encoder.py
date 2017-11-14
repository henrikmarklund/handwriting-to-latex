
from keras.layers import Dense

from keras.layers import Convolution2D, MaxPooling2D, Reshape, Input

from keras.backend import shape

from tensor2tensor.layers.common_attention import add_timing_signal_nd


reduce_factor = 2

def time_signal():

	def func(x):
		print("Function")
		return add_timing_signal_nd(x)

	return Lambda(func)

def get_encoded(image_h, image_w):

	encoder_inputs = Input(shape=(image_h, image_w,1), name="encoder_input_image")

	encoded = Convolution2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(encoder_inputs)
	encoded = MaxPooling2D(pool_size=2, padding='same')(encoded)

	encoded = Convolution2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(encoded)
	encoded = MaxPooling2D(pool_size=2, padding='same')(encoded)

	encoded = Convolution2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')(encoded)
	encoded = Convolution2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')(encoded)

	encoded = Convolution2D(filters=512, kernel_size=(3,3), padding='same', activation='relu')(encoded)

	encoded = Convolution2D(filters=512, kernel_size=(3,3), padding='valid', activation='relu')(encoded)


	#encoded = time_signal()(encoded)

	encoded_shape = encoded.get_shape().as_list()
	_, h, w, c = encoded_shape



	encoded = Reshape((w*h, c), name="unroll_encoding")(encoded)
	


	return encoder_inputs, encoded
