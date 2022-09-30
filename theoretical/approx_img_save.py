'''
In this script we save the edge maps of the transformed images.
'''

import os
from time import time
import numpy as np
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import argparse
import cv2
import pandas as pd
from tqdm import tqdm
from glob import glob

def magnitude_theoretical_1d(curve):
    curve = curve.flatten()
    curve_pad = np.pad(curve,1,mode='symmetric')

    boundary_x_1 = np.zeros_like(curve)
    boundary_x_1[0] = boundary_x_1[-1] = 1

    pref_x_1_plus = 1-np.exp(-np.abs(curve-curve_pad[0:-2]))
    delta_x_1_plus = pref_x_1_plus != 0

    pref_x_1_minus = 1-np.exp(-np.abs(curve-curve_pad[2:]))
    delta_x_1_minus = pref_x_1_minus != 0

    mag_vec_x_1 = 0.5*(0.001+boundary_x_1+pref_x_1_plus*delta_x_1_plus+pref_x_1_minus*delta_x_1_minus)

    return mag_vec_x_1

def rank_1_approx(img):
    u,s,v = np.linalg.svd(img)
    x = v[0]
    y = u[:,0]
    factor = s[0]

    mag_x = magnitude_theoretical_1d(np.sqrt(factor)*x)
    mag_y = magnitude_theoretical_1d(np.sqrt(factor)*y)

    return np.outer(mag_y.reshape(1,-1),mag_x)

def local_approx(img):
    img_pad = np.pad(img,1,mode='symmetric')

    boundary_x_1 = np.zeros_like(img)
    boundary_x_1[:,0] = boundary_x_1[:,-1] = 1

    pref_x_1_plus = 1-np.exp(-np.abs(img-img_pad[1:-1,2:]))
    delta_x_1_plus = pref_x_1_plus != 0

    pref_x_1_minus = 1-np.exp(-np.abs(img-img_pad[1:-1,:-2]))
    delta_x_1_minus = pref_x_1_minus != 0

    mag_vec_x_1 = 0.5*(0.01+boundary_x_1+pref_x_1_plus*delta_x_1_plus+pref_x_1_minus*delta_x_1_minus)

    boundary_x_2 = np.zeros_like(img)
    boundary_x_2[0,:] = boundary_x_2[-1,:] = 1

    pref_x_2_plus = 1-np.exp(-np.abs(img-img_pad[2:,1:-1]))
    delta_x_2_plus = pref_x_2_plus != 0

    pref_x_2_minus = 1-np.exp(-np.abs(img-img_pad[:-2,1:-1]))
    delta_x_2_minus = pref_x_2_minus != 0

    mag_vec_x_2 = 0.5*(0.01+boundary_x_2+pref_x_2_plus*delta_x_2_plus+pref_x_2_minus*delta_x_2_minus)

    return mag_vec_x_1*mag_vec_x_2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store',default='None',type=str)
    args = parser.parse_args()

    config = {'metric':1.0,
              'overlap':2}

    os.makedirs(os.path.join('Theory','results'), exist_ok=True)

    data_list = glob(os.path.join(args.data_path,'test','rgbr','real','*.jpg'))
    os.makedirs(os.path.join('results','edges_pred'), exist_ok=True)
    for img in tqdm(data_list):
        img_name = img.split('/')[-1].split('.')[0]
        img = cv2.imread(img)

        img_t = torch.from_numpy(img).permute(2,0,1)/255.
        img_t = 0.2989*img_t[0] + 0.5870*img_t[1] + 0.1140*img_t[2]
        img_t = img_t.unsqueeze(0)
        img_t = F.pad(img_t,1,padding_mode='reflect').unsqueeze(0)


        y_shape = img_t.shape[2]
        x_shape = img_t.shape[3]


        ########## Local ##################
        tic = time()
        local_mag_vec = local_approx(img_t.squeeze().numpy())
        toc = time()
        local_mag_vec = (local_mag_vec-local_mag_vec.min())/(local_mag_vec.max()-local_mag_vec.min())
        plt.imsave(os.path.join('results','edges_pred',f'{img_name}_local.png'), np.round(255*local_mag_vec[1:-1,1:-1]),cmap='Greys')

        ######### Rank 1 #########################

        tic = time()
        rank_1_mag_vec = rank_1_approx(img_t.squeeze().numpy())
        toc = time()
        rank_1_mag_vec = (rank_1_mag_vec-rank_1_mag_vec.min())/(rank_1_mag_vec.max()-rank_1_mag_vec.min())
        plt.imsave(os.path.join('results','edges_pred',f'{img_name}_rank_1.png'), np.round(255*rank_1_mag_vec[1:-1,1:-1]),cmap='Greys')

if __name__ == '__main__':
    main()
