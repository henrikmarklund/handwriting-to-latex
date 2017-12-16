def get_just_images(images_with_positions):
    smaller_images = []
    for image in images_with_positions:
        smaller_images.append(image['image'])
        
    return smaller_images


def get_latex(token):
    lower_case_letters = ('a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z')
    upper_case_letters = ('A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z')

    if token in lower_case_letters or token in upper_case_letters:
        latex = token
    elif token == 'zero':
        latex = '0'
    elif token == 'one':
        latex = '1'    
    elif token == 'two':
        latex = '2' 
    elif token == 'three':
        latex = '3'
    elif token == 'four':
        latex = '4'    
    elif token == 'five':
        latex = '5'
    elif token == 'six':
        latex = '6'
    elif token == 'seven':
        latex = '7'
    elif token == 'eight':
        latex = '8' 
    elif token == 'nine':
        latex = '9'
    elif token == 'equal':
        latex = "="
    elif token == 'greater':
        latex = ">"
    elif token == 'less':
        latex = "<"
    elif token == 'LeftPar':
        latex = "("
    elif token == 'RightPar':
        latex = ")"
    elif token == 'minus':
        latex = "-"
    elif token == 'plus':
        latex = "+"
    elif token == 'comma':
        latex = ","
    else:
        latex = "\\" + token
    return latex



def get_center_from_image(image_with_position):
    size = image_with_position['image'].shape
    position = image_with_position['position']
    
    
    centerx = int(position[1] + size[1] / 2)
    centery = int(position[0] + size[0] / 2)
    
    return (centery, centerx)


def split_using_contours(image_with_position):


    image = image_with_position['image']
    
    im2, contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    # Copied from this github: https://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d

    symbols = []

    for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            pad = 1 #padding

            cv2.rectangle(im, (x-pad, y-pad), (x+w+pad, y+h+pad), (0, 255, 0), 2)
            #print(x, w, y, h)
            symbol = im_orig[(y-pad):(y+h+pad), (x-pad):(x+w+pad)]
            symbols.append({ 'symbol': symbol, 'x_cord': x })
            #cv2.imshow(str(x), symbol)
            #cv2.waitKey(0)

