import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image

from sklearn import linear_model

from skimage.morphology import watershed
from skimage import feature

import helpers

try: from tqdm import tqdm
except: 
    print("tqdm not found!")
    tqdm = lambda x: x



# -------------------------------- segmentation -------------------------------

def compute_segmentation(image, sig = 1.0, footprint = 10):
    grayscale = helpers.to_lum(image)
    cn = np.where(feature.canny(grayscale, sigma = sig), False, True)

    distance = ndi.distance_transform_edt(cn)
    #distance = ndi.gaussian_filter(np.where(cn, 0.0, 1.0), 1.0 / sig)

    footprint_mat = np.ones((footprint, footprint))
    local_max = feature.peak_local_max(distance, indices = False, footprint = footprint_mat, labels = cn)
    markers = ndi.label(local_max)[0]

    labels = watershed(-distance, markers)

    return labels, markers, distance, cn


# -------------------------------- groups -------------------------------

def group_data(image, labels, group):
    w = np.where(labels == group)
    return image[w]

def group_truth(gt_image, labels, group, threshold = 0.25):
    df = np.sum(gt_image[np.where(labels == group)])
    tot = (labels == group).sum()
    return df > tot * threshold

def group_truth_image(image, gt_image):
    labels, _, _, _ = compute_segmentation(image)

    truth = np.zeros(gt_image.shape)
    for group in np.unique(labels):
        if group_truth(gt_image, labels, group):
            truth[np.where(labels == group)] = 255.0
    return truth

def image_features(image):
    labels, _, _, _ = compute_segmentation(image)

    groups = []
    for group in np.unique(labels):
        groups.append(group_features(group_data(image, labels, group)))

    return groups, labels

def predicted_image(labels, Zi):
    pred = np.zeros(labels.shape)
    idx = 1
    for i, group in enumerate(np.unique(labels)):
        if Zi[i] < 0.0:
            pred[np.where(labels == group)] = idx
            idx += 1
    return pred



# -------------------------------- features -------------------------------

def group_features(data):
    if len(data) == 0:
        print("EMPTY GROUP")

    sat = np.max(data)
    r = data[:, 0] / sat
    g = data[:, 1] / sat 
    b = data[:, 2] / sat
    lum = 0.2989 * r + 0.5870 * g + 0.1140 * b
    lum_m = np.mean(lum)
    mini = np.min(np.c_[r, g, b])
    maxi = np.max(np.c_[r, g, b])
    return [lum_m, mini, maxi, np.var(lum)]






# -------------------------------- debug -------------------------------

def shuffle_labels(labels):
    uniqs = np.unique(labels)
    sh = np.copy(uniqs)
    np.random.shuffle(sh)
    shuffled = np.zeros(labels.shape)
    for s, l in zip(sh, uniqs):
        shuffled[labels == l] = s
    return shuffled


def quick_visu():
    imgs, gt_imgs = helpers.load_training_set("training/")
    x_start = 150
    y_start = 150
    image = imgs[0][x_start : x_start + 100, y_start : y_start + 100]
    gt_image = gt_imgs[0][x_start : x_start + 100, y_start : y_start + 100]

    labels, markers, distance, canny = compute_segmentation(image)

    label_count = np.max(labels) + 1
    print(label_count, "labels")

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
    ax[5].imshow(group_truth_image(image, gt_image), cmap = plt.cm.gray, interpolation = 'nearest')
    ax[5].set_title('Per group truth')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()



# -------------------------------- learning -------------------------------

def learn(imgs, gt_imgs):
    groups = []
    truths = []
    for i in tqdm(range(len(imgs))):
        image = imgs[i]
        gt_image = gt_imgs[i]

        labels, _, _, _ = compute_segmentation(image)

        for group in np.unique(labels):
            #print(group_features(group_data(image, labels, group)))
            groups.append(group_features(group_data(image, labels, group)))
            truths.append(group_truth(gt_image, labels, group))

    X = np.asarray(groups)
    Y = np.where(np.asarray(truths), -1.0, 1.0) # this is so that we don't work on bools cause they make everything more complicated

    logreg = linear_model.LogisticRegression(C = 1e5, class_weight = "balanced")
    logreg.fit(X, Y)

    return logreg

def predict(image, logreg):
    Xi, labels = image_features(image)
    Zi = logreg.predict(Xi)
    print("scores in ", np.unique(Zi))
    return predicted_image(labels, Zi)






# -------------------------------- main -------------------------------

max_train = 99
refs, imgs, gt_imgs = helpers.load_training_set("training/", max_train + 1)

logreg = learn(imgs[:max_train], gt_imgs[:max_train])

ref = refs[max_train]
image = imgs[max_train]
gt_image = gt_imgs[max_train]
pred = predict(image, logreg)


labels, _, _, canny = compute_segmentation(image)

fig, axes = plt.subplots(ncols = 2, nrows = 2, subplot_kw = {'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(ref, interpolation = 'nearest')
ax[0].set_title('Reference')
ax[1].imshow(pred, cmap = plt.cm.spectral, interpolation = 'nearest')
ax[1].set_title('Prediction')
ax[2].imshow(gt_image, cmap = plt.cm.gray, interpolation = 'nearest')
ax[2].set_title('Ground truth')
ax[3].imshow(group_truth_image(image, gt_image), cmap = plt.cm.gray, interpolation = 'nearest')
ax[3].set_title('Per group truth')
fig.tight_layout()


plt.figure(1)
fig, axes = plt.subplots(ncols = 3, nrows = 1, subplot_kw = {'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(shuffle_labels(labels), cmap = plt.cm.spectral, interpolation = 'nearest')
ax[0].set_title('Separated objects')
ax[1].imshow(image, interpolation = 'nearest')
ax[1].set_title('Processed reference')
ax[2].imshow(canny, cmap = plt.cm.gray, interpolation = 'nearest')
ax[2].set_title('Canny')
fig.tight_layout()
plt.show()