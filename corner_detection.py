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

def getGreyImge(img):
# Change the rgb value to grey. #
    rgb_weights = np.array([0.2989, 0.5870, 0.1140])
    return np.dot(img[...,:3], rgb_weights)
    
def sumGradient(conv_x,conv_y,conv_xy,i,j,kernel_size,src,window):
    res = 0
    i_boundary, j_boundary = src.shape
    start = int(-(kernel_size-1)/2)
    end = int((kernel_size-1)/2 + 1)
    index_i = 0
    M = np.zeros((2,2),dtype=float)
    for x in range(start,end):
        index_j = 0
        for y in range(start,end):
            if(x+i<0 or y+j<0 or x+i>=i_boundary or y+j>=j_boundary): continue
            else:
                M[0][0] = conv_x[i+x][j+y] * window[index_i][index_j]
                M[0][1] = conv_xy[i+x][j+y] * window[index_i][index_j]
                M[1][0] = conv_xy[i+x][j+y] * window[index_i][index_j]
                M[1][1] = conv_y[i+x][j+y] * window[index_i][index_j]
            index_j += 1
        index_i += 1
        
    lambda_1 = (1/2) * (M[0][0]+M[1][1] + math.sqrt(4*M[0][1]*M[1][0]+(M[0][0]-M[1][1])**2))
    lambda_2 = (1/2) * (M[0][0]+M[1][1] - math.sqrt(4*M[0][1]*M[1][0]+(M[0][0]-M[1][1])**2))
        
    return lambda_1, lambda_2
    
def eigenvalues(src,kernel_size):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    conv_x = cv.filter2D(src,-1,sobel_x)
    conv_y = cv.filter2D(src,-1,sobel_y)
    x_length, y_length = src.shape
    conv_xy = np.zeros((x_length, y_length),dtype=float)
    
    for i in range(x_length):
        for j in range(y_length):
            conv_xy[i][j] = conv_x[i][j]*conv_y[i][j]
            conv_x[i][j] = conv_x[i][j]**2
            conv_y[i][j] = conv_y[i][j]**2
            
    windows = Gaussian_Blur(kernel_size/6,kernel_size)
    
    eigenvalues_1 = np.zeros((x_length,y_length),dtype=float)
    eigenvalues_2 = np.zeros((x_length,y_length),dtype=float)
    
    for i in range(x_length):
        for j in range(y_length):
            lambda_1, lambda_2 = sumGradient(conv_x,conv_y,conv_xy,i,j,kernel_size,src,windows)
            eigenvalues_1[i][j] = lambda_1
            eigenvalues_2[i][j] = lambda_2
    return eigenvalues_1, eigenvalues_2
    
def scatterplot(src):
    eigenvalues_1, eigenvalues_2 = eigenvalues(src,3)
    print(eigenvalues_1)
    #print(eigenvalues_2)
    
    plt.scatter(eigenvalues_1, eigenvalues_2, alpha=0.6)
    plt.show()
    
def corner_detection(src):
    eigenvalues_1, eigenvalues_2 = eigenvalues(src,3)
    x_length, y_length = src.shape
    R = np.zeros((x_length,y_length),dtype=float)
    for i in range(x_length):
        for j in range(y_length):
            if min(eigenvalues_1[i][j],eigenvalues_2[i][j])>0:
                R[i][j] = 255
    return R
    
if __name__ == '__main__':
    src_1 = plt.imread("./ex4.jpg")
    #scatterplot(getGreyImge(src_1))
    R = corner_detection(getGreyImge(src_1))
    Image.fromarray(R).show()


    
    
