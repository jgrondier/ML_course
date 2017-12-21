
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from scipy import ndimage
from skimage import io
from skimage import feature
from skimage import transform
import scipy.ndimage.filters as filters


def normalized(img):
    img = img - np.min(img)
    return img / np.max(img)



def downscale(img, factor):
    return img[::factor, ::factor]

def upscale(img, factor):
    b = img.repeat(factor, axis=0)
    return b.repeat(factor, axis=1)


def peak_angles(data):
    def best(angles):
        if len(angles):
            best = max([np.abs(45 - a) for a in angles])
            return [a for a in angles if np.abs(45 - a) == best]
        return []
    maxes = np.max(data, axis=0)
    bests = np.unique(maxes)[-7:]
    angles = [i for i, v in enumerate(maxes) if v in bests]
    return best([a % 90 for a in angles if 10 < a % 90 < 80])
    
    
def process(a, patch_size):
    # extract image orientation
    c = feature.canny(a, sigma = 5.0)
    h = transform.hough_line(c)[0]
    #coords = np.unravel_index(h.argmax(), h.shape)
    #angle = coords[1]
    
    angles = peak_angles(h)

    # main kernel size
    size = 101
    mid = int(size // 2.0)
    
    # create kernel
    kernel = np.zeros((size, size))
    kernel[mid, :] = kernel[:, mid] = 1
    kernel /= size
    kernel[mid, mid] = 1  
    
    # downscale to 1 pixel per patch
    d = downscale(a, patch_size)
    
    
    # convolution
    b = ndimage.convolve(d, kernel)
    for a in angles:
        print("using rotated kernel:", str(a) + "°")
        rot = ndimage.rotate(kernel, -a)
        b = b + ndimage.convolve(d, rot)

    b = normalized(b)
   
    # upscale back to size
    return upscale(b, patch_size)
