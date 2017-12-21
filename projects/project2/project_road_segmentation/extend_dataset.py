import os
import numpy as np
import matplotlib.image as mpimg
from PIL import Image


def per_channel(img, fun):
    if len(img.shape) == 2:
        return fun(img)
    out = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[2]):
        out[:, :, i] = fun(img[:, :, i])
    return out
    
def to_rgb(img):
    if len(img.shape) == 3:
        return img
    i = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    i[:,:,0] = img
    i[:,:,1] = img
    i[:,:,2] = img
    return i

if __name__ == '__main__':
    tr = [lambda i: i.T, lambda i: np.flipud(i), lambda i: np.fliplr(i), lambda i: np.flipud(np.fliplr(i))]
    for i in range(1, 101):
        dirs = ['training/images/', 'training/groundtruth/']
        for dir in dirs:
            image_filename = dir + ('satImage_%.3d' % i) + '.png'
            for j in range(len(tr)):
                out_filename = dir + ('satImage_%.3d' % (j * 100 + i + 100)) + '.png'
                img = mpimg.imread(image_filename) * 255
                out = per_channel(img, tr[j])
                Image.fromarray(to_rgb(out)).save(out_filename)