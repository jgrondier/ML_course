0

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image

from skimage.morphology import watershed
from skimage import feature
import helpers

def compute_segmentation(image, sig = 0.5, footprint = 10):
    grayscale = helpers.to_lum(image)
    cn = np.where(feature.canny(grayscale, sigma = sig), False, True)

    distance = ndi.distance_transform_edt(cn)
    #distance = ndi.gaussian_filter(np.where(cn, 0.0, 1.0), 1.0 / sig)

    footprint_mat = np.ones((footprint, footprint))
    local_max = feature.peak_local_max(distance, indices = False, footprint = footprint_mat, labels = cn)
    markers = ndi.label(local_max)[0]

    labels = watershed(-distance, markers)

    return labels, markers, distance, cn


def group_data(image, labels, group):
    return image[np.where(labels == group)]

def group_features(data):
    if len(data) == 0:
        return [0, 0.0, 0.0]
    return [len(data), np.mean(data), np.var(data)]

def group_truth(gt_image, labels, group, threshold = 0.7):
    df = np.sum(gt_image[np.where(labels == group)])
    tot = (labels == group).sum()
    return df > tot * threshold

    
imgs, gt_imgs = helpers.load_training_set("training/")
x_start = 150
y_start = 150
image = imgs[0][x_start : x_start + 100, y_start : y_start + 100]
gt_image = gt_imgs[0][x_start : x_start + 100, y_start : y_start + 100]

labels, markers, distance, canny = compute_segmentation(image)

label_count = np.max(labels) + 1
print(label_count, "labels")


truth = np.zeros(gt_image.shape)
for i in range(0, label_count):
    group = i + 1
    if group_truth(gt_image, labels, group):
        truth[np.where(labels == group)] = 255.0

fig, axes = plt.subplots(ncols = 3, nrows = 2, figsize = (9, 4), sharex = True, sharey = True, subplot_kw = {'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(image, interpolation = 'nearest')
ax[0].set_title('Reference')
ax[1].imshow(canny, cmap = plt.cm.gray, interpolation = 'nearest')
ax[1].set_title('Canny')
ax[2].imshow(-distance, cmap = plt.cm.gray, interpolation = 'nearest')
ax[2].set_title('Distances')
ax[3].imshow(labels, cmap = plt.cm.spectral, interpolation = 'nearest')
ax[3].set_title('Separated objects')
ax[4].imshow(gt_image, cmap = plt.cm.gray, interpolation = 'nearest')
ax[4].set_title('Ground truth')
ax[5].imshow(truth, cmap = plt.cm.gray, interpolation = 'nearest')
ax[5].set_title('Per group truth')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()