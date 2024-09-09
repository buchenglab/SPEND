from scipy import ndimage
import math
import random
from skimage import data, img_as_float
from skimage import exposure
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from skimage import io

def noise_level(Img_1, axis):
    noise = 0
    img_1 = Img_1
    img_2 = np.float64(img_1)
    [m, xx_1, yy_1] = np.shape(img_2)
    if (axis==0):
        for ii in range(m-1):
            noise = noise + np.mean(np.abs(img_2[ii, :, :]-img_2[ii+1, :, :]))

    if (axis==1):
        for ii in range(xx_1-1):
            noise = noise + np.mean(np.abs(img_2[:,ii,:]-img_2[:, ii+1, :]))

    if (axis==2):
        for ii in range(yy_1-1):
            noise = noise + np.mean(np.abs(img_2[:,:, ii]-img_2[:, :, ii+1]))

    noise = noise/50
    return noise

def perm_direc(Img, x_1, y_1, d):
    img = Img
    # io.imsave('miapaca.tif', img[1:d+1, x_1:x_1+d, y_1:y_1+d])
    noise_axis = np.zeros(3)
    for ii in range(3):
        noise_axis[ii] = noise_level(img[1:d+1, x_1:x_1+d, y_1:y_1+d], ii)

    print(noise_axis, np.argmin(noise_axis))

    if (np.argmin(noise_axis)!=0):
        return 0
    else:
        return np.argmax(noise_axis)
def img_save(img, Name, Target_folder, index, count):
    save_path = Target_folder + '/' + Name + '_' + str(index) + '_' + str(count) + '.tif'
    io.imsave(save_path, img)

def Img_Split_Same(Img, n, Target_path_low, Target_path_GT):
    img_p = Img
    [m, xx, yy] = np.shape(img_p)
    img1 = np.zeros((m, xx, yy), dtype=img.dtype)
    img2 = np.zeros((m, xx, yy), dtype=img.dtype)
    for ii in range(m):
        img1[ii, :, :] = img_p[ii, :, :]
        img2[ii, :, :] = (img_p[ii, :, :]+img_p[ii-1, :, :])/2
    img_save(img1, Target_path_low, index, n)
    img_save(img2, Target_path_GT, index, n)

def Img_Split_Half(Img, Name, n, Target_path_low, Target_path_GT):
    img_p = Img
    [m, xx, yy] = np.shape(img_p)
    img1 = np.zeros((m//2, xx, yy), dtype=img.dtype)
    img2 = np.zeros((m//2, xx, yy), dtype=img.dtype)
    frm_n = 0
    p = -1
    while frm_n < m:
        p += 1
        img1[p, :, :] = img[frm_n, :, :]
        img2[p, :, :] = img[frm_n+1, :, :]
        frm_n += 2
    img_save(img1, Name, Target_path_low, index, n)
    img_save(img2, Name, Target_path_GT, index, n)

def Img_Split_Conc(Img, axis, Name, n, Target_path_low, Target_path_GT):
    img_p = Img
    [m, xx, yy] = np.shape(img_p)
    if (axis==0):
        img1_p = img_p[::2, :, :]
        img2_p = img_p[1::2, :, :]
        img3_p = np.concatenate((img1_p, img2_p), axis=0)
        img4_p = np.concatenate((img2_p, img1_p), axis=0)

    if (axis==1):
        img1_p = img_p[:, ::2, :]
        img2_p = img_p[:, 1::2, :]
        img3_p = np.concatenate((img1_p, img2_p), axis=1)
        img4_p = np.concatenate((img2_p, img1_p), axis=1)

    if (axis==2):
        img1_p = img_p[:, :, ::2]
        img2_p = img_p[:, :, 1::2]
        img3_p = np.concatenate((img1_p, img2_p), axis=2)
        img4_p = np.concatenate((img2_p, img1_p), axis=2)

    img_save(img3_p, Name, Target_path_low, index, n)
    img_save(img4_p, Name, Target_path_GT, index, n)


folder_path = "./data/20240306_OVCAR5_50mW"
target_path_1 = "./data/Input"
target_path_2 = "./data/Target"

re_scale = False

if (re_scale):
    target_path_1 = target_path_1 + "_enhance"
    target_path_2 = target_path_2 + "_enhance"

if not os.path.exists(target_path_1):
    os.mkdir(target_path_1)
if not os.path.exists(target_path_2):
    os.mkdir(target_path_2)


img_name = os.listdir(folder_path)
for index, name in enumerate(img_name):
    img_path = folder_path + '/' + name
    img = io.imread(img_path)
    if (re_scale):
        p2, p98 = np.percentile(img, (2, 98))
        [m, xx, yy] = np.shape(img)
        img = exposure.rescale_intensity(img[:, :, :], in_range=(p2,p98))


    which_axis = perm_direc(img,7,7,50)
    print('which = ', which_axis)
    Img_Split_Conc(img, which_axis, name, 0,  target_path_1, target_path_2)
    img1 = np.flip(img, 1)
    Img_Split_Conc(img1, which_axis, name, 1, target_path_1, target_path_2)
    img2 = np.flip(img, 2)
    Img_Split_Conc(img2, which_axis, name, 2, target_path_1, target_path_2)
    img3 = np.flip(img1,2)
    Img_Split_Conc(img3, which_axis, name, 3, target_path_1, target_path_2)
# Img_Move(folder_path, target_path_1, target_path_2, 2)