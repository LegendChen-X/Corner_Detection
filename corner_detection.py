import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import scipy.linalg as la

### This part code is from my A1. ###
################################# Begin #################################
def convolution(matrix,x,y,src):
# This function is only for grey scale image. #
# Please use getGreyImge first, if input is a RGB image. #
# Can be optimized by the product of two vectors. #
    x_boundary, y_boundary = src.shape
    kernel_size = len(matrix)
# Help to track the row in Matrix. #
    index_i = 0
    res = 0
# Get the start and end of k. #
    start = int(-(kernel_size-1)/2)
    end = int((kernel_size-1)/2 + 1)
    for u in range(start,end):
# Help to track the coloum in Matrix. #
        index_j = 0
        for v in range(start,end):
# Boundary check. Smarter than padding the image. #
            if(x-u<0 or y-v<0 or x-u>=x_boundary or y-v>=y_boundary): res += 0
            else: res += src[x-u][y-v] * matrix[index_i][index_j]
            index_j += 1
        index_i += 1
    return res

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
################################# End #################################
    
def eigenvalues(src,kernel_size):
# Get Sobel Operator
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
# If you do not want me to use the library, we can use the convolution function above, but it is too slow.
    grad_x = cv.filter2D(src,-1,sobel_x)
    grad_y = cv.filter2D(src,-1,sobel_y)
# Get I_xy
    I_xy = grad_x * grad_y
# Get I_x^2
    I_x = grad_x * grad_x
# Get I_y^2
    I_y = grad_y * grad_y
# Get Gaussian Matrix
# Change kernel size if you want a different size of Gaussian Matrix
    window = Gaussian_Blur(kernel_size/6,kernel_size)
    # Use Gaussian Matrix to do the convolution (same as sum with Gaussian Matrix, central symmetry).
    # If you do not want me to use the library, we can use the convolution function above, but it is too slow.
    I_x = cv.filter2D(I_x,-1,window)
    I_y = cv.filter2D(I_y,-1,window)
    I_xy = cv.filter2D(I_xy,-1,window)
    x_length, y_length = src.shape
    eigenvalues_1 = np.zeros((x_length,y_length),dtype=complex)
    eigenvalues_2 = np.zeros((x_length,y_length),dtype=complex)
    for i in range(x_length):
        for j in range(y_length):
            # Get M.
            M = np.array([[I_x[i][j],I_xy[i][j]],[I_xy[i][j],I_y[i][j]]])
            eigvals, eigvecs = la.eig(M)
            # If you don't want me to use library, here is the alternative way to get eigenvalue of 2*2 Matricx
            # eigenvalues_1[i][j] = (1/2) * (M[0][0]+M[1][1] - math.sqrt(4*M[0][1]*M[1][0]+(M[0][0]-M[1][1])**2))
            # eigenvalues_2[i][j] = (1/2) * (M[0][0]+M[1][1] + math.sqrt(4*M[0][1]*M[1][0]+(M[0][0]-M[1][1])**2))
            eigenvalues_1[i][j] = eigvals[0]
            eigenvalues_2[i][j] = eigvals[1]
    return eigenvalues_1, eigenvalues_2
    
def scatterplot(src):
    eigenvalues_1, eigenvalues_2 = eigenvalues(src,3)
    # Choose alpha to make my image more beautiful.
    plt.scatter(eigenvalues_1, eigenvalues_2, alpha=0.6)
    plt.show()
    
def corner_detection(src):
# This function will return an array for binary image.
    eigenvalues_1, eigenvalues_2 = eigenvalues(src,3)
    x_length, y_length = src.shape
    R = np.zeros((x_length,y_length),dtype=float)
    for i in range(x_length):
        for j in range(y_length):
# Threshold choosen as 18888. I like this number.
            if min(eigenvalues_1[i][j],eigenvalues_2[i][j])>18888:
                R[i][j] = 255
    Image.fromarray(R).show()
    return R

def label(src, R):
    r, c, _ = src.shape
    new_image = np.zeros((r,c,_),dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            if R[i][j]:
            # If it is a corner, we set it into red(RGB=(255,0,0)).
                new_image[i][j][0] = 255
                new_image[i][j][1] = 0
                new_image[i][j][2] = 0
            else:
            # If it is not a corner, we set it into the same value woth src.
                new_image[i][j][0] = src_1[i][j][0]
                new_image[i][j][1] = src_1[i][j][1]
                new_image[i][j][2] = src_1[i][j][2]
    Image.fromarray(new_image).show()
    return new_image
    
if __name__ == '__main__':
    src_1 = plt.imread("./ex5.jpg")
    scatterplot(getGreyImge(src_1))
    R = corner_detection(getGreyImge(src_1))
    new_image_1 = label(src_1, R)
