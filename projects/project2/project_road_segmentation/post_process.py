
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from scipy import ndimage
from skimage import io
from skimage import feature
from skimage import transform


def normalized(img):
    img = img - np.min(img)
    return img / np.max(img)



def downscale(img, factor):
    return img[::factor, ::factor]

def upscale(img, factor):
    b = img.repeat(factor, axis=0)
    return b.repeat(factor, axis=1)

    
def process(a, patch_size):
    # extract image orientation
    c = feature.canny(a, sigma = 5.0)
    h = transform.hough_line(c)[0]
    io.imshow(normalized(h))
    coords = np.unravel_index(h.argmax(), h.shape)
    h[coords] = -h[coords]
    angle = coords[1]
    
    coords2 = np.unravel_index(h.argmax(), h.shape)
    print(coords, coords2)

    # main kernel size
    size = 101
    mid = int(size // 2.0)
    
    # create kernel
    kernel = np.zeros((size, size))
    kernel[mid, :] = kernel[:, mid] = 1
    kernel /= size
    kernel[mid, mid] = 1
    
    # orient kernel to match image
    kernel = ndimage.rotate(kernel, -angle)

    # convolution
    b = ndimage.convolve(downscale(a, patch_size), kernel)
    b = normalized(b)
   
    return upscale(b, patch_size)
