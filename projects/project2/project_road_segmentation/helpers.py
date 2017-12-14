import numpy as np
import matplotlib.image as mpimg
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

    return imgs, gt_imgs


def to_lum(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def to_gray_(rgb):
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