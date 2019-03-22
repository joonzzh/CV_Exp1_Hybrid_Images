import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np

def cross_correlation_2d(img, kernel):    #互相关
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
   
    if img.ndim == 2:  #灰度的图像
        m=kernel.shape[0]
        n=kernel.shape[1]
        height1 = int(m/2) #上下
        width1 =  int(n/2)  #左右
        new_img = np.pad(img,((height1,height1),(width1,width1)),'constant',constant_values = (0,0))
        height = new_img.shape[0]
        width = new_img.shape[1]
        tem_img = np.zeros(shape=new_img.shape)

        for i in range(height1,height-height1):
            for j in range(width1,width-width1):
                for u in range(m):
                    for v in range(n):
                        tem_img[i][j] += kernel[u][v]*new_img[i-height1+u][j-width1+v]


        return tem_img[height1:height-height1,width1:width-width1]
    else:
        b,g,r = cv2.split(img)  #拆分
        b = cross_correlation_2d(b,kernel)
        g = cross_correlation_2d(g,kernel)
        r = cross_correlation_2d(r,kernel)
        img = cv2.merge([b,g,r]) #合成
        return img


    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):      #卷积
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    #kernel = np.flip(kernel,-1)   #实现左右上下翻转
    kernel = np.flip(kernel,0)
    kernel = np.flip(kernel,1)
    img = cross_correlation_2d(img,kernel)
    return img

    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):   #高斯滤波
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
   
    x = height // 2
    y = width // 2
    tem = np.zeros((height,width))
    count = 0
    for i in range(height):
        for j in range(width):
            tem[i][j] = np.exp(-((i-x)**2+(j-y)**2)/(2*sigma**2))/(2*np.pi*sigma**2)
            count += tem[i][j]
    tem = tem / count   #保证像素和为1
    return tem

    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernel = gaussian_blur_kernel_2d(sigma,size,size)
    img = convolve_2d(img,kernel)
    return img

    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    img = img-low_pass(img,sigma,size)
    return img

    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


