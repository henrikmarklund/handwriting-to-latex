import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

def down_sample(images, factor): 
    target_h = int(math.floor(float(images[0].shape[0]) * factor)) + 1
    target_w = int(math.floor(float(images[0].shape[1]) * factor))
    num_images = images.shape[0]
    down_sampled_images = np.ones((num_images, target_h, target_w)) * 255

    for idx, image in enumerate(images):

        im = image

        #Downsample
        im = cv2.resize(im, (0, 0), fx = factor, fy=factor, interpolation = cv2.INTER_AREA) #cv2.INTER_LINEAR

        down_sampled_images[idx, :, :] = im

    return down_sampled_images


def crop_images(images, cropped_width, cropped_height):
    h = images.shape[1]
    w = images.shape[2]
    h_start = int(h / 2 - cropped_height / 2)
    w_start = int(w / 2 - cropped_width / 2)
    
    
    cropped_images = images[:,h_start:h_start+cropped_height,w_start:w_start+cropped_width]
    
    return cropped_images



def normalize_and_invert(images):
    return (255.0 - images) / 255

def flatten(images):
    return np.reshape(images,(images.shape[0], images.shape[1] * images.shape[2]))

def add_dimension(images):
    return np.reshape(images,(images.shape[0], images.shape[1], images.shape[2], 1))


def shuffle_data(X,Y, seed=None):
    if seed is not None:
        np.random.seed(seed)
    num_samples = X.shape[0]
    p = np.random.permutation(num_samples)
    if len(X.shape) == 3:
        X = X[p,:,:]
    elif len(X.shape) == 4:
        X = X[p,:,:,:]
    Y = Y[p]
    
    return X,Y


## Visualize training history
## CODE TAKEN FROM https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
def visualize_training_history(fit_history):
    # list all data in history
    print(fit_history.history.keys())
    # summarize history for accuracy
    plt.plot(fit_history.history['acc'])
    plt.plot(fit_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()





def pad_image(image, target_size):
    
    im = np.asarray(image)
    padded_image = np.ones(target_size)*255
    
    
    old_h = im.shape[0]
    old_w = im.shape[1]
    new_h = target_size[0]
    new_w = target_size[1]
    

    if old_h <= new_h and old_w <= new_w:
        offset_h = int((new_h - old_h) / 2)
        offset_w = int((new_w - old_w) / 2)
        padded_image[offset_h:offset_h+old_h, offset_w:offset_w+old_w] = im
    else:
        if old_h > new_h and old_w > new_w:
            offset_h = int((old_h - new_h) / 2)
            offset_w = int((old_w - new_w) / 2)
            padded_image = im[offset_h:offset_h+new_h, offset_w:offset_w+new_w]
        elif old_h <= new_h and old_w > new_w:
            offset_h = int((new_h - old_h) / 2)
            offset_w = int((old_w - new_w) / 2)
            padded_image[offset_h:offset_h+old_h,:] = im[:, offset_w:offset_w+new_w]
        elif old_h > new_h and old_w <= new_w:
            offset_w = int((new_w - old_w) / 2)
            offset_h = int((old_h - new_h) / 2)
            padded_image[:, offset_w:offset_w+old_w] = im[offset_h:offset_h+new_h, :]
    
    return padded_image


def pad_images(images, target_size):
    padded_images = np.ones((len(images), target_size[0], target_size[1])) * 255
    for idx, image in enumerate(images):
        padded_images[idx,:,:] = pad_image(image, target_size)

    return padded_images
def down_sample_single(image, factor): 

    #Downsample
    im = cv2.resize(image, (0, 0), fx = factor, fy=factor, interpolation = cv2.INTER_AREA) #cv2.INTER_LINEAR
    

    return im