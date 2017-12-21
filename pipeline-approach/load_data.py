#https://stackoverflow.com/questions/27577435/read-txt-file-line-by-line-in-python

import numpy as np
from matplotlib import pyplot as plt

from PIL import Image

ON_FLOYDHUB = False

if ON_FLOYDHUB:
    symbols_file_path = "/../data_seg_ocr/InftyCDB-3/CharInfoDB-3-B.txt"
    equations_folder_path = "/../data_seg_ocr/database/img/"
    ocr_code_list = "/../data_seg_ocr/InftyCDB-3/OcrCodeList.txt"
else:
    symbols_file_path = "../../data_seg_ocr/InftyCDB-3/CharInfoDB-3-B.txt"
    equations_folder_path = "../../data_seg_ocr/database/img/"
    ocr_code_list = "../../data_seg_ocr/InftyCDB-3/OcrCodeList.txt"



def load_data_step_1():

    widths = []
    heights = []

    num_examples = 0

    with open (symbols_file_path, "r") as myfile:
        
        for idx, line in enumerate(myfile):
            num_examples = num_examples + 1

            current_line = line.split(",")

            height = current_line[3]
            width = current_line[4]
            widths.append(width)
            heights.append(height)
  
        
    widths = np.array(widths, dtype=np.int32)
    heights = np.array(heights,dtype=np.int32)

    max_width = np.max(widths)
    max_height = np.max(heights)

    print("Max width: ", max_width)
    print("Min width: ", np.min(widths))
    print("Mean width: ", np.mean(widths))


    print("Max height: ", max_height)
    print("Min heigh: ", np.min(heights))
    print("Mean height: ", np.mean(heights))


    print("Number of examples: ", num_examples)


    X = np.zeros((num_examples, max_height, max_width))
    Y = []
    with open (symbols_file_path, "r") as myfile:
        
        for idx, line in enumerate(myfile):
            
            current_line = line.rstrip()
            current_line = current_line.split(',')

            height = int(current_line[3])
            width = int(current_line[4])

            img_bytes_1d = np.array(current_line[15:-1], dtype=np.uint8)

            img_bytes_2d = np.reshape(img_bytes_1d, (height , -1))

            img_bits_2d = np.unpackbits(img_bytes_2d, axis=1)

            img = img_bits_2d[:,:width]

            


            # Center the image
            y_offset = int((max_height - height) / 2)
            x_offset = int((max_width - width) / 2)

            X[idx, y_offset:y_offset+height, x_offset:x_offset+width] = img
            
            label = current_line[2]

            Y.append(label)
            

    return X, Y



def load_math_symbols():
    X, Y = load_data_step_1()
    target_tokens = set(Y)

    target_token_index = dict(
    [(token, i) for i, token in enumerate(target_tokens)])


    X = 255 - 255 * X


    new_Y = np.zeros(len(Y))
    for idx, y in enumerate(Y):
        new_Y[idx] = target_token_index[y]


    num_unique = len(set(target_tokens))

    print("Math symbols loaded")

    return X, new_Y, num_unique, target_token_index


def load_equations(max_num_samples):

    num_equations = 4400
    pad_size = 50
    images_folder = equations_folder_path

    equation_images = []

    for i in range(num_equations):
        
        img_number = "{0:0>4}".format(i+1)
        
        file_name = "img" + img_number + ".gif"
        file_path = images_folder + file_name
        im = Image.open(file_path)
        im = np.asarray(im)
        
        padded_im = np.ones((im.shape[0] + pad_size, im.shape[1] + pad_size))*255
        
        padded_im[int(pad_size/2):int(pad_size/2)+im.shape[0],int(pad_size/2):int(pad_size/2)+im.shape[1]] = im                 
        

        equation_images.append(padded_im)
    
        if i == max_num_samples:
            break
    return equation_images



def get_hex_to_token_dict():
    hex_to_token_dict = {}


    with open (ocr_code_list, "r") as myfile:
         
        for idx, line in enumerate(myfile):
            
            current_line = line.rstrip()
            current_line = current_line.split(',')
            
            hex_to_token_dict[current_line[0][2:]] = current_line[2]

    return hex_to_token_dict

