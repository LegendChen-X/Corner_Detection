import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

### This part code is from my A1. ###
################################# Begin #################################
def Gaussian_Model(sigma,point):
    x, y = point
    ratio = 1/(2*math.pi*sigma**2)
# Slightly different for the expression from the lecture. #
    e_part = math.exp(-(x**2+y**2)/(2*sigma**2))
    return ratio * e_part
    
def Gaussian_Blur(sigma,kernel_size):
# This can use array to boost access speed. #
    half_kernel = (kernel_size-1) // 2
    Gaussian_matrix = np.zeros((kernel_size,kernel_size),dtype=float)
    for i in range(kernel_size):
        for j in range(kernel_size):
            Gaussian_matrix[i][j] = Gaussian_Model(sigma,(j-half_kernel,half_kernel-i))
    return Gaussian_matrix

def Sobel_Operation(src):
# This function is only for grey scale image. #
# Please use getGreyImge first, if input is a RGB image. #
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    x_length, y_length = src.shape
    res = np.empty((x_length,y_length), dtype=float)
    conv_x = cv.filter2D(src,-1,sobel_x)
    conv_y = cv.filter2D(src,-1,sobel_y)
    conv_x = conv_x * conv_x
    conv_y = conv_y * conv_y
    for i in range(x_length):
        for j in range(y_length):
            res[i][j] = math.sqrt(conv_x[i][j]+conv_y[i][j])
    return res

def getGreyImge(img):
# Change the rgb value to grey. #
    rgb_weights = np.array([0.2989, 0.5870, 0.1140])
    return np.dot(img[...,:3], rgb_weights)
################################# End #################################
    
def cutColoum(src):
    i_length, j_length = src.shape
    # Array for saving the minimum energy from bottom to up (Dynamic Programming).
    backtrack = np.zeros((i_length, j_length), dtype=float)
    # Recod the direction. We want to use the direction to trace the minimum energy path.
    directions = np.zeros((i_length, j_length), dtype=int)
    # Get the gradients(energy map).
    gradients = Sobel_Operation(src)
    # Initilize the last row of backtrack. It should be the same as the last row of our energy map.
    for j in range(j_length): backtrack[i_length-1][j] = gradients[i_length-1][j]
    # From bottom to top.
    for i in reversed(range(i_length-1)):
        for j in range(j_length):
            # case 1: j=0, we only have two choice up and up-right.
            if not j:
                # Get the index for the minimum value.
                index = np.argmin(backtrack[i+1, j:j + 2])
                # Get the minmium path from this point to the bottom.
                backtrack[i][j] = gradients[i][j] + backtrack[i+1][j+index]
                # Trace the direction.
                directions[i][j] = index
            # case 2: j=length-1, we only have two choices up-left and up.
            elif j == j_length - 1:
                # Get the index for the minimum value.
                index = np.argmin(backtrack[i+1, j-1:j+1])
                # Get the minmium path from this point to the bottom.
                backtrack[i][j] = gradients[i][j] + backtrack[i+1][j-1+index]
                # Trace the direction.
                directions[i][j] = index - 1
            # case 3: normal j, we have three choices up-left, up and up-right.
            else:
                # Get the index for the minimum value.
                index = np.argmin(backtrack[i+1, j-1:j + 2])
                # Get the minmium path from this point to the bottom.
                backtrack[i][j] = gradients[i][j] + backtrack[i+1][j-1+index]
                # Trace the direction.
                directions[i][j] = index - 1
    # Boolean matrix to record which one will be removed or kept.
    binary_matrix = np.ones((i_length, j_length), dtype=int)
    # Index for the minimum value of the first row.
    min_j = np.argmin(backtrack[0])
    # Trace the path of minimum value and set position of boolean matrix to 0.
    for i in range(i_length):
        binary_matrix[i][min_j] = 0
        min_j = min_j + directions[i][min_j]
    return binary_matrix
    
def cutRow(src):
    i_length, j_length = src.shape
    rotate = np.zeros((j_length, i_length), dtype=float)
    # Rotate the matrix
    for i in range(i_length):
        for j in range(j_length):
            rotate[j_length-j-1][i] = src[i][j]
    # Rotate and cut one coloum.
    return cutColoum(rotate)

def mainCol(src,new_col):
# This funciton take src of RGB image.
    # Get r, c and dimension.
    row, col, _ = src.shape
# Make a copy, since we need to loop n times on new image.
    new_image = src.copy()
# If the parameter is invalid, return src.
    if(new_col>=col): return src
# Cut coloums col-new_col times.
    for k in range(col-new_col):
        binary_matrix = cutColoum(getGreyImge(new_image))
        i_length,j_length = binary_matrix.shape
        buff = new_image.copy()
        new_image = np.zeros((i_length,j_length-1,_),dtype=np.uint8)
        # Set new value to new_image based on binary matrix.
        for i in range(i_length):
            new_j = 0
            for j in range(j_length):
                if binary_matrix[i][j]:
                    new_image[i][new_j][0] = buff[i][j][0]
                    new_image[i][new_j][1] = buff[i][j][1]
                    new_image[i][new_j][2] = buff[i][j][2]
                    new_j += 1
    return new_image
    
def mainRow(src,new_row):
# This funciton take src of RGB image.
    # Get r, c and dimension.
    row, col, _ = src.shape
    # Make a copy, since we need to loop n times on new image.
    new_image = src.copy()
    # If the parameter is invalid, return src.
    if(new_row>=row): return src
    # Cut coloums row-new_row times.
    for k in range(row-new_row):
        binary_matrix = cutRow(getGreyImge(new_image))
        j_length,i_length = binary_matrix.shape
        rotate_binary_matrix = np.zeros((i_length, j_length), dtype=int)
        # Rotate the binary matrix.
        for i in range(i_length):
            for j in range(j_length):
                rotate_binary_matrix[i][j] = binary_matrix[j_length-j-1][i]
        buff = new_image.copy()
        new_image = np.zeros((i_length-1,j_length,_),dtype=np.uint8)
        # Set new value to new_image based on binary matrix.
        for j in range(j_length):
            new_i = 0
            for i in range(i_length):
                if rotate_binary_matrix[i][j]:
                    new_image[new_i][j][0] = buff[i][j][0]
                    new_image[new_i][j][1] = buff[i][j][1]
                    new_image[new_i][j][2] = buff[i][j][2]
                    new_i += 1
    return new_image
        
def main(src,new_row,new_col):
    # Make a copy.
    new_image = src.copy()
    # Cut rows.
    new_image = mainRow(new_image,new_row)
    # Cut cols.
    new_image = mainCol(new_image,new_col)
    return new_image
    
def process(path, new_row, new_col):
# resize the image.
    img = cv.imread(path)
    resized = cv.resize(img, (new_col, new_row), interpolation = cv.INTER_AREA)
    cv.imshow("Resized image", resized)
# Press 0 to continue.
    cv.waitKey(0)
# Crop image,
    img = cv.imread(path)
    crop_img = img[0:new_row, 0:new_col]
    cv.imshow("cropped", crop_img)
# Press 0 to continue.
    cv.waitKey(0)
# Seam image(Very slow).
    src_1 = plt.imread(path)
    src_buff = cv.filter2D(src_1,-1,Gaussian_Blur(0.6,3))
    img_1 = Image.fromarray(main(src_buff, new_row, new_col),'RGB')
    img_1.show()
        
if __name__ == '__main__':
    process("ex1.jpg",968,957)
    process("ex2.jpg",861,1200)
    process("ex3.jpg",870,1200)
