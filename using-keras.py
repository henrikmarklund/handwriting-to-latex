#Import Numpy, Tensorflow and Keras
import numpy as np
import tensorflow as tf
import math
import cv2

from keras.layers import Input, LSTM, Dense, Lambda, GlobalAveragePooling1D
from keras.layers import Convolution2D, MaxPooling2D, Reshape, Flatten, BatchNormalization, Embedding
from keras import layers, backend ## IS THIS BEING USED?
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard, Callback
import keras.backend as K
from keras import optimizers, metrics

from keras.utils import plot_model
from IPython.display import Image

from matplotlib import pyplot as plt
import h5py
# Import our own helper functions
from prepare_data import get_decoder_data, get_decoder_data_int_sequences

## This cell contains a bunch of functions for loading and preprocessing the data

def get_max_shape(images):

    max_height = 0
    max_width = 0

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
    print("downsampling images")
    new_images = []

    for image in images:
        new_image = cv2.resize(image, (0, 0), fx = factor, fy=factor, interpolation = cv2.INTER_AREA) #cv2.INTER_LINEAR
        new_images.append(new_image)

    return new_images


def pad_images(images, target_shape = None):
    print(images[0])

    if (target_shape == None):
        max_height, max_width = get_max_shape(images)

    num_images = len(images)

    padded_images = np.ones((num_images, max_height, max_width)) * 255

    for idx, image in enumerate(images):

        h = image.shape[0]
        w = image.shape[1]

        padded_images[idx, :h, :w] = image


    return padded_images


def load_raw_data(dataset, max_token_length = 400, max_image_size = (60, 200), max_num_samples = 5000):

    
    token_vocabulary = []
    token_sequences = []
    images = []
    
    if dataset == "small":
        image_folder = 'data/tin/tiny/'
        formula_file_path = "data/tin/tiny.formulas.norm.txt"
    elif dataset == "test":
        image_folder = 'data/images_test/'
        formula_file_path = "data/test.formulas.norm.txt"
    elif dataset == "train":
        image_folder = '../data/images_train/'
        formula_file_path = "../data/train.formulas.norm.txt"


        
    included_counter = 0
    examples_counter = 0
    with open (formula_file_path, "r") as myfile:

        for idx, token_sequence in enumerate(myfile):
            examples_counter += 1
            #Check token size:
            token_sequence = token_sequence.rstrip('\n')
            tokens = token_sequence.split()

            file_name = str(idx) + '.png'
            image = cv2.imread(image_folder + file_name, 0)
            
            if image is None:
                print("Not loading image with id:", idx)
                continue
            
            #print(tokens)
            if len(tokens) <= max_token_length and image.shape[0] <= max_image_size[0] and image.shape[1] <= max_image_size[1]:
                token_sequences.append('**start** ' + token_sequence + ' **end**')
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Grey scale
                #print(image)
                
                images.append(image)
                for token in tokens:
                    if token not in token_vocabulary:
                        token_vocabulary.append(token)

                included_counter += 1
                if included_counter == max_num_samples:
                    break

        
    token_vocabulary.append("**start**")
    token_vocabulary.append("**end**")
    
    return images, token_sequences, token_vocabulary


def preprocess_images(images):

    down_sample_factor = 0.7
    encoder_input = down_sample_flexible(images, 0.7)
    encoder_input = pad_images(encoder_input)
    encoder_input = normalize_images(encoder_input)

    # Add dimension for TensorFlow Conv Layers to work properly as it needs (None, Height, Width, 1)
    encoder_input = encoder_input.reshape(encoder_input.shape[0], encoder_input.shape[1], encoder_input.shape[2], 1)

    return encoder_input

def create_metric_output_files():

    file = open("/output/metrics.txt","w") 

    file.write("Train loss" + "\t" + "Val loss" + "\n")

    file.close()

def load_data(dataset, max_token_length, max_image_size, max_num_samples):

    if (dataset == "small"):
        images, token_sequences, token_vocabulary = load_raw_data(dataset="small", max_num_samples=max_num_samples)
        images = preprocess_images(images)
    elif (dataset == "test"):
        images, token_sequences, token_vocabulary = load_raw_data(dataset="test",  max_token_length = max_token_length, max_image_size = max_image_size, max_num_samples=max_num_samples)
        images = preprocess_images(images)
    elif (dataset == "train"):
        images, token_sequences, token_vocabulary = load_raw_data(dataset="train",  max_token_length = max_token_length, max_image_size = max_image_size, max_num_samples=max_num_samples)
        images = preprocess_images(images)
    return images, token_sequences, token_vocabulary



## Load and process data (takes a up to 10 minutes)

max_token_length = 70
max_image_size = (60, 270)
max_num_samples = 100000
encoder_input_data, target_texts, token_vocabulary = load_data(dataset="train", 
                                                               max_token_length=max_token_length,
                                                               max_image_size=max_image_size,
                                                               max_num_samples=max_num_samples)

target_tokens = token_vocabulary


## Note: Approx 15 images are missing and will not be loaded.



num_decoder_tokens = len(target_tokens)

max_decoder_seq_length = max([len(txt.split()) for txt in target_texts])

target_token_index = dict(
    [(token, i) for i, token in enumerate(target_tokens)])

reverse_target_token_index = dict(
    (i, char) for char, i in target_token_index.items()) ## Will be used in the inference model

print("Maximum output sequence lenght: " + str(max_decoder_seq_length) + "\n")
print("Examples of sequences: ")
print("Ex. 1: " + str(target_texts[0]) + "\n")
print("Ex. 1: " + str(target_texts[1]) + "\n \n")

print("Number of examples: " + str(len(encoder_input_data)))


print("Number of tokens in our vocabulary: " + str(num_decoder_tokens))
print("5 example of tokens: " + str(target_tokens[0:5]) + "\n")

print("\n Example pairs (token, index) in dictionary: ")

for i, key in enumerate(target_token_index):
    print(key, target_token_index[key])
    if i > 10:
        break

_, image_h, image_w, _  = encoder_input_data.shape


# For forced teaching, we need decoder_input data and decoder target data. (takes a few minutes)
# Decoder target data is just decoder_input_data offset by one time step.

decoder_input_data, decoder_target_data = get_decoder_data(target_texts,
                                                            target_tokens,
                                                             num_decoder_tokens,
                                                              max_decoder_seq_length,
                                                               target_token_index)

print("Each row is a one-hot encoded token in the sequence.")
print("We have 10 columns because there are 10 tokens in our vocabulary")
print("We have 9 rows, because maximum output length is 9")
print("")

print("Decoder INPUT sequence example 1")
print(decoder_input_data[0]) #Each row is a one-hot encoded token in the sequence.
print("")
print("Decoder TARGET sequence example 1 (the same as above offset by one time step)")
print(decoder_target_data[0]) #Each row is a one-hot encoded token in the sequence.


## Time to build our model: Image -> ConvNet Encoder -> LSTM Decoder --> Latex




## Encoder step I: Encoding image into vectors (e1, e2, ..., en)
## Convnet design from Guillaume Genthial https://github.com/guillaumegenthial/im2latex/blob/master/model/encoder.py

def get_encoded(image_h, image_w):

    encoder_inputs = Input(shape=(image_h, image_w,1), name="encoder_input_image", dtype='float32')

    # Conv + max_pool / 2
    encoded = Convolution2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(encoder_inputs)
    encoded = MaxPooling2D(pool_size=2, padding='same')(encoded)

    encoded = BatchNormalization()(encoded)
    # Conv + max_pool /2
    encoded = Convolution2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(encoded)
    encoded = MaxPooling2D(pool_size=2, padding='same')(encoded)

    encoded = BatchNormalization()(encoded)
    
    # 2 Conv
    encoded = Convolution2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')(encoded)
    encoded = Convolution2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')(encoded)
    
    # Pooling + Convnet + Pooling (Note pool_size)
    encoded = MaxPooling2D(pool_size=(2,1))(encoded)
    encoded = Convolution2D(filters=512, kernel_size=(3,3), padding='same', activation='relu')(encoded)
    encoded = MaxPooling2D(pool_size=(1,2))(encoded)
    
    # BatchNormalization, Convolution
    encoded = BatchNormalization()(encoded)
    encoded = Convolution2D(filters=512, kernel_size=(3,3), padding='valid', activation='relu')(encoded)

    #encoded = time_signal()(encoded)

    encoded_shape = encoded.get_shape().as_list()
    _, h, w, c = encoded_shape

    #Unroll the encoding to a series of vectors (e1, e2, e3..... en)
    encoded = Reshape((w*h, c), name="unroll_encoding")(encoded)
    
    return encoder_inputs, encoded



## Encoder step II: transforming (e1, e2... en) to h0 and c0 
# h0, and c0  will be the initial state of the decoder

# Call convolutional encoder
encoder_inputs, encoded = get_encoded(image_h, image_w)

encoded_shape = encoded.get_shape().as_list()

#Compute the average e from encoding.

e_average = GlobalAveragePooling1D(name='average_e')(encoded)

e_average = BatchNormalization()(e_average)

#Compute h0 and c0, from e_average, following Genthial's suggestion
h0 = Dense(512, activation='tanh', name="h0")(e_average)
c0 = Dense(512, activation='tanh', name="c0")(e_average)

h0 = BatchNormalization()(h0)
c0 = BatchNormalization()(c0)

create_metric_output_files()

## Decoder. LSTM + Softmax layer

decoder_lstm_dim = 512

# Training decoder
# Set up the decoder, using `encoder_states` as initial state.
#decoder_inputs = Input(shape=(max_decoder_seq_length, num_decoder_tokens), name='decoder_input_sequence')


# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.

    
decoder_inputs = Input(shape=(max_decoder_seq_length, num_decoder_tokens), name='decoder_input_sequence')

decoder_lstm = LSTM(decoder_lstm_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=[h0, c0])

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

## Putting the training model together

model = Model(inputs=[encoder_inputs, decoder_inputs],outputs=decoder_outputs)

## Visualize the training model

#plot_model(model, to_file='output/model_visualizations/training_model.png', show_shapes=True)

#Image(filename='training_model.png') 

# Callback to get losses for each batch (and not each epoch)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

    def on_epoch_end(self, epoch, logs={}):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        file = open("/output/" + "metrics.txt","a") 
        
        file.write(str(train_loss) + "\t" + str(val_loss) + "\n")

        file.close()
        
loss_history = LossHistory()


# Optimizer

learning_rate = 0.003 # OBS: Learning rate is set with a callback instead (see next cell)
beta_1 = 0.9 # Keras default
beta_2 = 0.999 # Keras default
epsilon=1e-08 # Keras default
decay=0.0004 # OBS: Decaying with a callback instead (see next cell)
clipvalue = 5 # 

adam_optimizer = optimizers.Adam(lr=learning_rate,
                                       beta_1=beta_1,
                                       beta_2=beta_2, 
                                       epsilon=epsilon,
                                       decay=decay,
                                        clipvalue=clipvalue)



## Compile and train the model

# checkpoint
#filepath="/output/checkpoints/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, period=2)

callbacks_list = [loss_history]
#print("loading model")
#model.load_weights("my_model.h5")
model.compile(adam_optimizer, loss='categorical_crossentropy')



epochs = 24
batch_size = 32

model_history = model.fit([encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.05,
         callbacks=callbacks_list)




#print("Losses: ", loss_history.losses)


#print("Saving model")
model.save("/output/my_model.h5")




# Step 1. Set up the encoder as a separate model:
encoder_model = Model(encoder_inputs, [h0, c0]) #encoded and e_average are included for debugging purposes

#print("Save encoder model")
encoder_model.save("/output/encoder.h5")

#encoder_model = load_model("encoder.h5")

# Step 2. Set up the decoder as a separate model.

# The decoder takes three inputs: the input_state_h, input_state_c and a vector (last prediction)

latent_dim = 512
decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Will be a one-hot encoded vector
decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
print("Save decoder model")
decoder_model.save("/output/decoder.h5")

## Decode sequence using our two models

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    
    h0_ = states_value[0]
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    #Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["**start**"]] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    
    decoded_sentence = ''
    while not stop_condition:
        #print(target_seq)
        output_tokens, h, c = decoder_model.predict(
        [target_seq] + states_value)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token_prob = np.max(output_tokens[0, -1, :])
        sampled_char = reverse_target_token_index[sampled_token_index]
        
        # Exit condition: either hit max length
        # or find stop token.
        if (sampled_char == '**end**' or
            len(decoded_sentence.split()) > max_decoder_seq_length):
            stop_condition = True
        else: 
            decoded_sentence = decoded_sentence + ' ' + sampled_char
        

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    #Return h0_ for debugging
    return decoded_sentence, h0_


num_test = 40 # We're currently predicting on the train set

encoded_images = []

print("Predicting on training data")

for seq_index in range(encoder_input_data[:10].shape[0]):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]

    #plt.imshow(np.squeeze(input_seq), cmap='gray')
    #plt.show()
    
    decoded_sentence, h0_ = decode_sequence(input_seq)
    
    #plt.text(0.1, 0.5, r"$%s$" % decoded_sentence, fontsize = 10)                                  

    #hide axes                                                                      
    #fig = plt.gcf()
    #fig.set_size_inches(5, 1)

    #fig.axes.get_xaxis().set_visible(False)                                         
    #fig.axes.get_yaxis().set_visible(False)  

    #plt.draw() #or savefig                                                          
    #plt.show()
   
    
    
    #encoded_images.append(h0_)
    
    
    print('-')
    print('Target sentence: ' + str(target_texts[seq_index]))
    print('Decoded sentence: ' + str(decoded_sentence))



print("predicting on validation data")

val_target_texts = target_texts[-60:]
val_encoder_input_data = encoder_input_data[-60:]

for seq_index in range(val_encoder_input_data.shape[0]-3):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = val_encoder_input_data[seq_index: seq_index + 1]

    #plt.imshow(np.squeeze(input_seq), cmap='gray')
    #plt.show()
    
    decoded_sentence, h0_ = decode_sequence(input_seq)
    
    #plt.text(0.1, 0.5, r"$%s$" % decoded_sentence, fontsize = 10)                                  

    #hide axes                                                                      
    #fig = plt.gcf()
    #fig.set_size_inches(5, 1)

    #fig.axes.get_xaxis().set_visible(False)                                         
    #fig.axes.get_yaxis().set_visible(False)  

    #plt.draw() #or savefig                                                          
    #plt.show()
   
    
    
    #encoded_images.append(h0_)
    
    
    print('-')
    print('Target sentence: ' + str(val_target_texts[seq_index]))
    print('Decoded sentence: ' + str(decoded_sentence))
