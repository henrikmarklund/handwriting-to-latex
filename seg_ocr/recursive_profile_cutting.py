import numpy as np

vertical_threshold = 0
horizontal_threshold = 0

def project(image, direction):

    #invert image
    image = 255 - image


    if direction == "VERTICAL":

        v = np.sum(image, 0)
        return v
    elif direction == "HORIZONTAL":

        v = np.sum(image, 1)
        return v


def cut(image_with_position, direction):

    image = image_with_position['image']
    
    position = image_with_position['position']

    v = project(image, direction)
    #print("v: ", v)
    
    new_images_with_positions = []

    
    
    if direction == "VERTICAL":
        threshold = vertical_threshold
    elif direction == "HORIZONTAL":
        threshold = horizontal_threshold
    

    
    reached_first = False
    
    white_counter = 0
    cut_positions = []
    
    
    last_non_white_index = None
    
    
    values_enough_for_split = [0]
    
    for index, value in enumerate(v):
        #print("Index: ", index)
        #print("value: ", value )
        if value in values_enough_for_split:
            white_counter = white_counter + 1
            
        else:
            last_non_white_index = index
            if reached_first:
                if white_counter > threshold:
                    cut_position = int(index - white_counter / 2)
                    cut_positions.append(cut_position)
            else:
                if index != 0:
                    cut_positions.append(index - 1)
                else:
                    cut_positions.append(index)     
            
            white_counter = 0        
            reached_first = True

    #if last_non_white_index == None:
        
    if last_non_white_index == len(v):
        cut_positions.append(last_non_white_index)
    else:
        cut_positions.append(last_non_white_index + 1)
        
       # print white_counter
        #print(" ")
    #Make the cuts
    
    #print("Cut positions: ", cut_positions)
    if len(cut_positions) == 0:
        new_images_with_positions.append(image_with_position)
        return new_images_with_positions
    else:

        for idx in range(len(cut_positions)):
            if idx == len(cut_positions) - 1:
                #print('last stop')
                break
                
            if direction == "VERTICAL":
                #print("Vertical")
                new_image = image[:,cut_positions[idx]:cut_positions[idx+1]]
                new_x_position = position[1] + cut_positions[idx]
                new_position = (position[0], new_x_position)
            elif direction == "HORIZONTAL":
                #print("Horizontal")
                new_image = image[cut_positions[idx]:cut_positions[idx+1],:]
                new_y_position = position[0] + cut_positions[idx]
                new_position = (new_y_position, position[1])
                                    
                
            new_image_with_position = {'image': new_image, 'position': new_position}

            new_images_with_positions.append(new_image_with_position)
            

        
        
    return new_images_with_positions


def recursive_cutter(image_with_position, direction):
    
    cut_images = cut(image_with_position, direction)
    
    if direction == "VERTICAL":
        next_direction = "HORIZONTAL"
    elif direction == "HORIZONTAL":
        next_direction = "VERTICAL"
        
    
    #print("Length of cut images: ", len(cut_images))
    #print(cut_images)
    
    if len(cut_images) == 1:
        i = 0
        
        return cut_images
    else:
        new_cut_images = []
        
        for cut_image in cut_images:
            new_cut_images.extend(recursive_cutter(cut_image, next_direction))
            
        
        return new_cut_images



def segment_equation(equation_image):

    my_equation_image = {'image': equation_image, 'position': (0,0)}

    images_with_positions = recursive_cutter(my_equation_image, "VERTICAL")
    return images_with_positions