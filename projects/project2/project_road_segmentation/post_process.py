
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


def process(a):

    tmp = a < 0.5

    tmp = ndimage.binary_fill_holes(tmp, structure=np.ones((4,4)))

    return a
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

    b = ndimage.convolve(a, kernel)
    b = normalized(b)

    return b
