
import numpy as np



def to_lum(rgb):
    return rgb[0] * 0.2989 + rgb[1] * 0.5870 + rgb[2] * 0.1140 
    
def to_lum_img(rgb):
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    
    
    
def extract_mean_var(img, patch_margin):
    center_patch = img[0]
    margin_patches = img[1:]
    
    center_m = np.mean(center_patch)
    center_v = np.var(center_patch)
    
    if patch_margin == 0:
        return [center_m, center_v]
        
    margin_m = np.mean(margin_patches)
    margin_v = np.var(margin_patches)
    
    return [center_m, center_v, margin_m, margin_v]
    
def extract_rgb_means(img, patch_margin):
    center_patch = img[0]
    margin_patches = img[1:]
    
    center_rgb = [np.mean(center_patch[:, :, i]) for i in range(0, 3)]
    
    if patch_margin == 0:
        return center_rgb
        
    margin_rgb = [np.mean(margin_patches[:, :, :, i]) for i in range(0, 3)]
    
    return center_rgb + margin_rgb
    
def extract_rgb_mean_var(img, patch_margin):
    center_patch = img[0]
    margin_patches = img[1:]
    
    center_rgb_m = [np.mean(center_patch[:, :, i]) for i in range(0, 3)]
    center_rgb_v = [np.var(center_patch[:, :, i]) for i in range(0, 3)]
    
    if patch_margin == 0:
        return center_rgb_m + center_rgb_v
        
    margin_rgb_m = [np.mean(margin_patches[:, :, :, i]) for i in range(0, 3)]
    margin_rgb_v = [np.var(margin_patches[:, :, :, i]) for i in range(0, 3)]
    
    return center_rgb_m + center_rgb_v + margin_rgb_m + margin_rgb_v
    
    
def extract_delta_to_gray(img, patch_margin):
    center_patch = img[0]
    margin_patches = img[1:]
    
    gray = to_lum_img(center_patch)
    rgb_delta = [np.abs(center_patch[:, :, i] - gray) for i in range(0, 3)]
    rgb_delta = [np.max(rgb_delta[i]) for i in range(0, 3)]
    
    return rgb_delta