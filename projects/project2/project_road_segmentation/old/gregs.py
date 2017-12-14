import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from sklearn import linear_model
from skimage import feature

import extract_features as extract


# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg
    
    
def to_rgb_img(img):
    if len(img.shape) == 3:
        return img
    
    w = img.shape[0]
    h = img.shape[1]
    img_3c = np.zeros((w, h, 3))          
    img_3c[:, :, 0] = img
    img_3c[:, :, 1] = img
    img_3c[:, :, 2] = img
    return img_3c

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    return np.concatenate((to_rgb_img(img), to_rgb_img(gt_img)), axis = 1)
    
def into_patches(im):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, patch_size):
        for j in range(0, imgwidth, patch_size):
            # get the center path
            if is_2d:
               img_patch = [im[j : j + patch_size, i : i + patch_size]]
            else:
               img_patch = [im[j : j + patch_size, i : i + patch_size, :]]
            
            #every other patch
            for x in range(-patch_margin, patch_margin + 1):
                for y in range(-patch_margin, patch_margin + 1):
                    if not (x == 0 and y == 0):
                        x_offset = x * patch_size
                        j_beg = j + x_offset
                        j_end = j_beg + patch_size
                        
                        y_offset = y * patch_size
                        i_beg = i + y_offset
                        i_end = i_beg + patch_size
                        
                        if j_beg >= 0 and i_beg >= 0 and j_end < im.shape[0] and i_end < im.shape[1]:
                            if is_2d:
                                img_patch.append(im[j_beg : j_end, i_beg : i_end])
                            else:
                                img_patch.append(im[j_beg : j_end, i_beg : i_end, :])
            for p in img_patch:
                assert(p.shape[0] == patch_size)
                assert(p.shape[1] == patch_size)
                assert(p.shape == img_patch[0].shape)
            
            list_patches.append(np.asarray(img_patch))
    return list_patches
    
def extract_features(img):
    return np.array(extract.canny(img, patch_margin))
    #return features.extract_delta_to_gray(img)

# Extract features for a given image
def extract_img_features(img):
    img = into_patches(img)
    return np.asarray([extract_features(patch) for patch in img])

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j : j + w, i : i + h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def load_training_set(root_dir, max_images = 20):
    im_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    #cn_dir = root_dir + "cannyedges/"
    
    files = os.listdir(im_dir)
    n = min(max_images, len(files))
    
    imgs    = [load_image(im_dir + files[i]) for i in range(n)]
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    #cn_imgs = [load_image(cn_dir + files[i]) for i in range(n)]
    
    for i in range(n):
        assert(imgs[i].shape[0] == gt_imgs[i].shape[0])
        assert(imgs[i].shape[1] == gt_imgs[i].shape[1])
        #assert(imgs[i].shape[0] == cn_imgs[i].shape[0])
        #assert(imgs[i].shape[1] == cn_imgs[i].shape[1])
    
    return (imgs, gt_imgs)


def compute_patches(imgs):
    images = [into_patches(img) for img in imgs]
    return np.asarray([patch for img in images for patch in img])





# Loaded a set of images
imgs, gt_imgs = load_training_set("training/")

# Extract patches from input images
patch_margin = 0
patch_size = 40 

img_patches = compute_patches(imgs)
gt_patches = compute_patches(gt_imgs)

# Compute features for each image patch
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

X = np.asarray([extract_features(patch) for patch in img_patches])
Y = np.asarray([value_to_class(np.mean(patch)) for patch in gt_patches])

print('Computed ' + str(X.shape[0]) + ' features')
print('Feature dimension = ' + str(X.shape))
print('Patch size = ' + str(gt_patches[0].shape))
print('Number of classes = ' + str(np.max(Y) + 1))

# we create an instance of the classifier and fit the data
logreg = linear_model.LogisticRegression(C = 1e5, class_weight = "balanced")
logreg.fit(X, Y)

# Predict on the training set
Z = logreg.predict(X)

# Get non-zeros in prediction and grountruth arrays
Zn = np.nonzero(Z)[0]
Yn = np.nonzero(Y)[0]

TPR = len(list(set(Yn) & set(Zn))) / float(len(Z))
print('True positive rate = ' + str(TPR))

# Run prediction on the img_idx-th image
img_idx = 12

Xi = extract_img_features(imgs[img_idx])
Zi = logreg.predict(Xi)

w = gt_imgs[img_idx].shape[0]
h = gt_imgs[img_idx].shape[1]
predicted_im = label_to_img(w, h, patch_size, patch_size, Zi)
cimg = concatenate_images(imgs[img_idx], predicted_im)

new_img = make_img_overlay(to_rgb_img(imgs[img_idx]), predicted_im)
new_img.show()
