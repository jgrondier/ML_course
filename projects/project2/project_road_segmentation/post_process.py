
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from scipy import ndimage
from skimage import io
from skimage import feature
from skimage import transform
import skimage.morphology as morphology
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
    

def fill(a):
    kernel = np.array([[0.0 , 0.25, 0.0 ],
                       [0.25, 0.0 , 0.25],
                       [0.0 , 0.25, 0.0 ]])
    return ndimage.convolve(a, kernel)
    
def keep_connected(img):
    mat = np.where(img > 0.5, 1, 0)
    labels, count = morphology.label(mat, connectivity=2, return_num=True)
    
    
    scores = [(labels == (i + 1)).sum() for i in range(count)]
    
    best_score = max(scores)
    keep_score = best_score * 0.025
    keep = [i for i in range(count) if scores[i] > keep_score]
    
    out = np.zeros(img.shape)
    
    for k in keep:
        out[labels == (k + 1)] = 1
            
    return out


def process(a, patch_size):
    USE_SINGLE_ANGLE = True

    # extract image orientation
    # c = morphology.skeletonize(a > 0.5)
    c = feature.canny(a, sigma = 5.0)
    h = transform.hough_line(c)[0]
    
    if USE_SINGLE_ANGLE:
        coords = np.unravel_index(h.argmax(), h.shape)
        angle = coords[1]
    else:
        angles = peak_angles(h)

    # main kernel size
    size = 7
    mid = int(size // 2.0)
    
    # create + kernel
    kernel = np.zeros((size, size))
    kernel[mid, :] = 1
    kernel[:, mid] = 1
    kernel /= size
    kernel[mid, mid] = 1  
    
    # downscale to 1 pixel per patch"""
    d = downscale(a, patch_size)
    
    d = fill(d)
    
    # convolution
    
    if USE_SINGLE_ANGLE:
        rot = ndimage.rotate(kernel, -angle)
        b = ndimage.convolve(d, rot)
        b /= kernel.sum()
    else:
        b = ndimage.convolve(d, kernel)
        for a in angles:
            print("using rotated kernel:", str(a) + "°")
            rot = ndimage.rotate(kernel, -a)
            b = b + ndimage.convolve(d, rot)
            

        b /= kernel.sum() * (len(angles) + 1)
    
        
    #b = ndimage.convolve(morphology.skeletonize(b > 0.5), np.ones((4, 4)))
    
    # remove small island
    #b = keep_connected(b)
    
    # upscale back to size
    return upscale(b, patch_size)
