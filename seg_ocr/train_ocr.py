



import load_data
from keras.datasets import mnist
import matplotlib.pyplot as plt


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import numpy as np
## Model 1

X, Y, num_unique = load_data.load_data_step_2()
#num_classes = 275



np.random.seed(100)

num_samples = X.shape[0]

p = np.random.permutation(num_samples)

X = X[p,:,:]
Y = Y[p]


X_train = X[:60000]
Y_train = Y[:60000]

Y_train = np_utils.to_categorical(Y_train, num_classes = num_unique)

X_test = X[60000:]
Y_test = Y[60000:]
Y_test = np_utils.to_categorical(Y_test, num_classes = num_unique)


h = X_train.shape[1]
w = X_train.shape[2]

cropped_width = 15
cropped_height = 20

h_start = int(h / 2 - cropped_height / 2)
w_start = int(w / 2 - cropped_width / 2)


num_subset = 80000

X_train = X_train[:,h_start:h_start+cropped_height,w_start:w_start+cropped_width]
X_test = X_test[:,h_start:h_start+cropped_height,w_start:w_start+cropped_width]

print("X_train shape: ", X_train.shape)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

num_pixels = X_train.shape[1]
print("Num pixels: ", num_pixels)


print("num classes: ", Y_test.shape[1])
num_classes = Y_test.shape[1]


# fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
# load data
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
# flatten 28*28 images to a 784 vector for each image
#num_pixels = X_train.shape[1] * X_train.shape[2]
#X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

#X_train = X_train / 255
#X_test = X_test / 255

# one hot encode outputs
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = y_test.shape[1]

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model




# build the model

model = baseline_model()
# Fit the model
model.fit(X_train[:num_subset,:], Y_train[:num_subset], validation_split=0.10, epochs=100, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test[:num_subset,:], Y_test[:num_subset], verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))