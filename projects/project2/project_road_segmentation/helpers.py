import numpy as np
import matplotlib.image as mpimg
from scipy import ndimage as ndi
import os, sys

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def load_training_set(root_dir, max_images = 20):
    im_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    #cn_dir = root_dir + "cannyedges/"
    
    files = os.listdir(im_dir)
    n = min(max_images, len(files))
    
    imgs    = [load_image(im_dir + files[i]) for i in range(n)]
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    pr_imgs    = [process(i) for i in imgs]

    return imgs, pr_imgs, gt_imgs

def process(rgb):
    return rgb

def sharpen(rgb, alpha = 0.1):
    blurred = blur(rgb, 1.0)
    return np.clip(rgb + alpha * (rgb - blurred), 0.0, 255.0)


def median(rgb, size = 5):
    bl = np.zeros(rgb.shape)          
    bl[:, :, 0] = ndi.median_filter(rgb[:, :, 0], size)
    bl[:, :, 1] = ndi.median_filter(rgb[:, :, 1], size)
    bl[:, :, 2] = ndi.median_filter(rgb[:, :, 2], size)
    return bl

def blur(rgb, sig = 3.0):
    bl = np.zeros(rgb.shape)          
    bl[:, :, 0] = ndi.gaussian_filter(rgb[:, :, 0], sig)
    bl[:, :, 1] = ndi.gaussian_filter(rgb[:, :, 1], sig)
    bl[:, :, 2] = ndi.gaussian_filter(rgb[:, :, 2], sig)
    return bl

def saturate(rgb, amount):
    bl = np.zeros(rgb.shape)          
    bl[:, :, 0] = ndi.gaussian_filter(rgb[:, :, 0], sig)
    bl[:, :, 1] = ndi.gaussian_filter(rgb[:, :, 1], sig)
    bl[:, :, 2] = ndi.gaussian_filter(rgb[:, :, 2], sig)
    return bl

def to_bool(gray, threshold = 0.75):
    return gray > threshold

def to_lum(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def to_gray(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return (r + g + b) / 3.0

def to_rgb(img):
    if len(img.shape) == 3:
        return img

    img_3c = np.zeros((img.shape[0], img.shape[1], 3))          
    img_3c[:, :, 0] = img
    img_3c[:, :, 1] = img
    img_3c[:, :, 2] = img
    return img_3c
