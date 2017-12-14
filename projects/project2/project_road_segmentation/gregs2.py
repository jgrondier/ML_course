
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage as ndi
from PIL import Image

from skimage.morphology import watershed
from skimage import feature




def to_lum_img(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def to_gray_img(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return (r + g + b) / 3.0

def to_rgb_img(img):
    if len(img.shape) == 3:
        return img

    img_3c = np.zeros((img.shape[0], img.shape[1], 3))          
    img_3c[:, :, 0] = img
    img_3c[:, :, 1] = img
    img_3c[:, :, 2] = img
    return img_3c

def canny(img, sig = 0.25):
    cn = feature.canny(img[:, :, 1], sigma = sig)
    return cn #np.where(cn[:, :], 1.0, 0.0)

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
    cn_imgs = [canny(i) for i in imgs]

    return imgs, cn_imgs
    

imgs, cn_imgs = load_training_set("training/")
image = np.where(cn_imgs[0][:, :], False, True)
image = image[150 : 250, 100 : 200]
ref = imgs[0][150 : 250, 100 : 200, :]


# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
footprint_size = 3
distance = ndi.distance_transform_edt(image)
local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((footprint_size, footprint_size)),labels=image)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=4, figsize=(9, 4), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(ref, interpolation='nearest')
ax[0].set_title('Reference')
ax[1].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('Overlapping objects')
ax[2].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('Distances')
ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
ax[3].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()