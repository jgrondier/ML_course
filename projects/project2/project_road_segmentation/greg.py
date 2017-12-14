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
    

imgs, ct_imgs = helpers.load_training_set("training/")
image = imgs[0][150 : 250, 100 : 200]
#image = helpers.load_image("./test.bmp")

labels, markers, distance, canny = compute_segmentation(image)

label_count = np.max(labels) + 1
print(label_count, "labels")

fig, axes = plt.subplots(ncols = 4, figsize = (9, 4), sharex = True, sharey = True, subplot_kw = {'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(image, interpolation = 'nearest')
ax[0].set_title('Reference')
ax[1].imshow(canny, cmap = plt.cm.gray, interpolation = 'nearest')
ax[1].set_title('Canny')
ax[2].imshow(-distance, cmap = plt.cm.gray, interpolation = 'nearest')
ax[2].set_title('Distances')
ax[3].imshow(labels, cmap = plt.cm.spectral, interpolation = 'nearest')
ax[3].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()