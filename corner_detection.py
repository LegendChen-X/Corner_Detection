import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import scipy.linalg as la

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
    
def eigenvalues(src,kernel_size):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    grad_x = cv.filter2D(src,-1,sobel_x)
    grad_y = cv.filter2D(src,-1,sobel_y)
    
    conv_xy = grad_x * grad_y
    
    conv_x = grad_x * grad_x
    conv_y = grad_y * grad_y
    
    window = Gaussian_Blur(kernel_size/6,kernel_size)
    conv_x = cv.filter2D(conv_x,-1,window)
    conv_y = cv.filter2D(conv_y,-1,window)
    conv_xy = cv.filter2D(conv_xy,-1,window)
    
    x_length, y_length = src.shape
    
    eigenvalues_1 = np.zeros((x_length,y_length),dtype=complex)
    eigenvalues_2 = np.zeros((x_length,y_length),dtype=complex)
    
    for i in range(x_length):
        for j in range(y_length):
            M = np.array([[conv_x[i][j],conv_xy[i][j]],[conv_xy[i][j],conv_y[i][j]]])
            eigvals, eigvecs = la.eig(M)
            # eigenvalues_1[i][j] = (1/2) * (M[0][0]+M[1][1] - math.sqrt(4*M[0][1]*M[1][0]+(M[0][0]-M[1][1])**2))
            # eigenvalues_2[i][j] = (1/2) * (M[0][0]+M[1][1] + math.sqrt(4*M[0][1]*M[1][0]+(M[0][0]-M[1][1])**2))
            eigenvalues_1[i][j] = eigvals[0]
            eigenvalues_2[i][j] = eigvals[1]
    return eigenvalues_1, eigenvalues_2
    
def scatterplot(src):
    eigenvalues_1, eigenvalues_2 = eigenvalues(src,3)
    plt.scatter(eigenvalues_1, eigenvalues_2, alpha=0.6)
    plt.show()
    
def corner_detection(src):
    eigenvalues_1, eigenvalues_2 = eigenvalues(src,6)
    x_length, y_length = src.shape
    R = np.zeros((x_length,y_length),dtype=float)
    for i in range(x_length):
        for j in range(y_length):
            if min(eigenvalues_1[i][j],eigenvalues_2[i][j])>7000:
                R[i][j] = 255
    return R
    
if __name__ == '__main__':
    src_1 = plt.imread("./ex4.jpg")
    #scatterplot(getGreyImge(src_1))
    R = corner_detection(getGreyImge(src_1))
    Image.fromarray(R).show()


    
    
