import numpy as np
import os
import cv2
import sys
from scipy import signal

k_gauss = np.array([[2, 4, 5, 4, 2],
              [4, 9, 12, 9, 4],
              [5, 12, 15, 12, 5],
              [4, 9, 12, 9, 4],
              [2, 4, 5, 4, 2]]) /139

k_robert_x = np.array([
    [-1, 1],
    [-1, 1]
])

k_robert_y = np.array([
    [1, 1],
    [-1, -1]
])

k_sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

k_sobel_y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

src = np.array([[1,2,3,4,5],
                [2,3,4,5,6],
                [3,4,5,6,7],
                [3,1,2,3,4],
                [2,2,1,3,3]])

def Filter(srcImg, op):

    (rows, cols) = srcImg.shape
    op_size = op.shape[0]

    tempImg = srcImg.copy().astype('float')
    desImg = np.vstack((np.zeros((op_size//2, cols)), tempImg, np.zeros((op_size//2, cols))))
    desImg = np.hstack((np.zeros((rows+2*(op_size//2), op_size//2)), desImg, np.zeros((rows+2*(op_size//2), op_size//2))))
    _k = np.array(op.flat)
    _k = _k[::-1]
    _k = np.reshape(_k, (op_size, op_size))
    print(_k)
    for i in range(0, rows):
        for j in range(0, cols):
            subImg = desImg[i:i+op_size, j:j+op_size]
            tempImg[i, j] = np.sum(subImg * _k)
    return tempImg

def CalGradient(srcImg, op_x, op_y):
    _x = Filter(srcImg, op_x)
    _y = Filter(srcImg, op_y)
    _g = np.sqrt(np.power(_x, 2) + np.power(_y, 2))
    _r = np.arctan2(_y, _x)
    return _g, _r, _x, _y

def LimitNonMax(gradient, orient, grad_x, grad_y):
    (rows, cols) = gradient.shape
    _orient = orient.copy()
    _orient[_orient<0] = np.pi - _orient[_orient<0]
    limit = gradient.copy()
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if _orient[i, j] == 0:
                dtmpA = gradient[i, j-1]
                dtmpB = gradient[i, j+1]
            elif _orient[i, j] > 0 and _orient[i, j] < np.pi/4:
                w = abs(grad_y[i, j]/grad_x[i, j])
                dtmpA = w * gradient[i, j+1] + (1-w) * gradient[i-1, j+1]
                dtmpB = w* gradient[i, j-1] + (1-w) * gradient[i+1, j-1]
            elif _orient[i, j] == np.pi/4:
                dtmpA = gradient[i+1, j-1]
                dtmpB = gradient[i-1, j+1]
            elif _orient[i, j] > np.pi/4 and _orient[i, j] < np.pi/2:
                w = abs(grad_x[i, j]/grad_y[i, j])
                dtmpA = w * gradient[i-1, j] + (1-w) * gradient[i-1, j+1]
                dtmpB = w * gradient[i+1, j] + (1-w) * gradient[i+1, j-1]
            elif _orient[i, j] == np.pi/2:
                dtmpA = gradient[i-1, j]
                dtmpB = gradient[i+1, j]
            elif _orient[i, j] > np.pi/2 and _orient[i, j] < np.pi*3/4:
                w = abs(grad_x[i, j] / grad_y[i, j])
                dtmpA = w * gradient[i-1, j] + (1-w) * gradient[i-1, j-1]
                dtmpB = w * gradient[i+1, j] + (1-w) * gradient[i+1, j+1]
            elif _orient[i, j] == np.pi*3/4:
                dtmpA = gradient[i-1, j-1]
                dtmpB = gradient[i+1, j+1]
            elif _orient[i, j] > np.pi*3/4 and _orient[i, j] < np.pi:
                w = abs(grad_y[i, j] / grad_x[i, j])
                dtmpA = w * gradient[i, j-1] + (1 - w) * gradient[i-1, j-1]
                dtmpB = w * gradient[i, j+1] + (1 - w) * gradient[i+1, j+1]


            if gradient[i, j] < dtmpA or gradient[i, j] < dtmpB:
                limit[i, j] = 0
    return limit

def threshold(src, low, high):
    (rows, cols) = src.shape
    binary = src.copy()
    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(suppress=True)
    #print(binary)
    binary[src<=low] = 0
    binary[src>=high] = 250
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if src[i, j] > low and src[i, j] < high:
                if np.max(src[i-1:i+1, j-1:j+1]) == 250:
                    binary[i, j] = 250
                else:
                    binary[i, j] = 0
    return binary

def Canny(src, low, high):
    gauss = Filter(src, k_gauss)
    _g, _r, _x, _y = CalGradient(gauss, k_robert_x, k_robert_y)
    limit = LimitNonMax(_g, _r, _x, _y)
    canny = threshold(limit, 10, 40)
    return canny

if __name__ == '__main__':
    #print(Filter(src, k_gauss))
    #print(signal.convolve2d(src, k_gauss, 'same'))
    src = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    canny = Canny(src, 10, 40)
    cv2.imwrite('canny.png', canny)

    '''
    edge = cv2.Canny(gauss.astype('uint8'), 80, 160)
    cv2.imwrite('canny1.png', edge)
    '''
