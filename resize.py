import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

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
    
def cutColoum(src):
    i_length, j_length = src.shape
    backtrack = np.zeros((i_length, j_length), dtype=float)
    directions = np.zeros((i_length, j_length), dtype=int)
    gradients = Sobel_Operation(src)
    for j in range(j_length): backtrack[i_length-1][j] = src[i_length-1][j]
    for i in reversed(range(i_length-1)):
        for j in range(j_length):
            if not j:
                index = np.argmin(backtrack[i+1, j:j + 2])
                backtrack[i][j] = gradients[i][j] + backtrack[i+1][j+index]
                directions[i][j] = index
            elif j == j_length - 1:
                index = np.argmin(backtrack[i+1, j-1:j+1])
                backtrack[i][j] = gradients[i][j] + backtrack[i+1][j-1+index]
                directions[i][j] = index - 1
            else:
                index = np.argmin(backtrack[i+1, j-1:j + 2])
                backtrack[i][j] = gradients[i][j] + backtrack[i+1][j-1+index]
                directions[i][j] = index - 1
    binary_matrix = np.ones((i_length, j_length), dtype=int)
    min_j = np.argmin(backtrack[0])
    for i in range(i_length):
        binary_matrix[i][min_j] = 0
        min_j = min_j + directions[i][min_j]
    return binary_matrix
    
def cutRow(src):
    i_length, j_length = src.shape
    rotate = np.zeros((j_length, i_length), dtype=float)
    for i in range(i_length):
        for j in range(j_length):
            rotate[j_length-j-1][i] = src[i][j]
    return cutColoum(rotate)

def mainCol(src,new_col):
    row, col, _ = src.shape
    new_image = src.copy()
    if(new_col>=col): return src
    for k in range(col-new_col):
        print(k)
        binary_matrix = cutColoum(getGreyImge(new_image))
        i_length,j_length = binary_matrix.shape
        buff = new_image.copy()
        new_image = np.zeros((i_length,j_length-1,_),dtype=np.uint8)
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
    row, col, _ = src.shape
    new_image = src.copy()
    if(new_row>=row): return src
    for k in range(row-new_row):
        print(k)
        binary_matrix = cutRow(getGreyImge(new_image))
        j_length,i_length = binary_matrix.shape
        rotate_binary_matrix = np.zeros((i_length, j_length), dtype=int)
        for i in range(i_length):
            for j in range(j_length):
                rotate_binary_matrix[i][j] = binary_matrix[j_length-j-1][i]
        buff = new_image.copy()
        new_image = np.zeros((i_length-1,j_length,_),dtype=np.uint8)
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
    new_image = src.copy()
    new_image = mainRow(new_image,new_row)
    new_image = mainCol(new_image,new_col)
    return new_image
        
if __name__ == '__main__':
    img = cv.imread("ex2.jpg")
    resized = cv.resize(img, (1200, 861), interpolation = cv.INTER_AREA)
    cv.imshow("Resized image", resized)
    cv.waitKey(0)
"""
    img = cv.imread("ex3.jpg")
    crop_img = img[0:870, 0:1200]
    cv.imshow("cropped", crop_img)
    cv.waitKey(0)
"""
'''
    src_1 = plt.imread("./my.jpg")
    print(src_1.shape)
    src_buff = cv.filter2D(src_1,-1,Gaussian_Blur(0.6,3))
    img_1 = Image.fromarray(main(src_buff, 870, 1200),'RGB')
    img_1.save('my2.jpg')
    img_1.show()
'''
    
