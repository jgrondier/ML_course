
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
    c = feature.canny(a, sigma = 5.0)
    h = transform.hough_line(c)[0]
    io.imshow(normalized(h))
    coords = np.unravel_index(h.argmax(), h.shape)
    angle = coords[1]

    kernel = np.zeros((101, 101))
    mid = int(kernel.shape[0] // 2.0)
    kernel[mid, :] = kernel[:, mid] = 1
    kernel /= np.sum(kernel)
    kernel = ndimage.rotate(kernel, -angle)

    b = ndimage.convolve(downscale(a, patch_size), kernel)
    b = normalized(b)
   
    return upscale(b, patch_size)
