'''
In this script we calculate the approximation benchmarks
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



class ImageTransformer(torch.nn.Module):
    '''
    A light version of the Image Transformer. Here we need only the vanilla magnitude
    '''
    def __init__(self,x_y_patches=(2,2),overlap=1,metric=1.0):
        super().__init__()
        self.x_y_patches = x_y_patches
        self.overlap = overlap
        self.metric=metric
    def forward(self,x):
        patches,patch_grid = self._make_patches(x)
        patches_vecs = []
        for p in tqdm(patches,desc='Patches',leave=False):
            tmp = p.contiguous()
            with torch.no_grad():
                mag_patch = self._calculate_mag(tmp).unsqueeze(0)
            patches_vecs.append(mag_patch)
        mag_img_approx = self._stitch_patches(patches_vecs,patch_grid)
        return mag_img_approx
    def _make_patches(self,x):
        _,self.n_channels,height,width = x.shape
        x_patches, y_patches = self.x_y_patches
        x_patch_size = int(width/x_patches)
        y_patch_size = int(height/y_patches)

        if (width/x_patches)%1 != 0.0:
            raise ValueError('The image height must be divisible by the number of patches!')
        elif (height/y_patches)%1 != 0.0:
            raise ValueError('The image width must be divisible by the number of patches!')

        patches = []
        patch_grid = []

        for i in range(x_patches):
            for j in range(y_patches):
                patch_grid.append((j,i))
                patches.append(x[:,:,max(j*y_patch_size-self.overlap,0):min((j+1)*y_patch_size+self.overlap,height),max(i*x_patch_size-self.overlap,0):min((i+1)*x_patch_size+self.overlap,width)])
        return patches, patch_grid
    def _stitch_patches(self,patches,patch_grid):
        x_patches, y_patches = self.x_y_patches
        for p,g in enumerate(patch_grid):
            j,i = g
            if i == j == 0:
                if (x_patches == 1) and (y_patches == 1):
                    batch_size,channels,patch_y_shape,patch_x_shape = patches[p].shape
                elif (x_patches == 1) and (y_patches > 1):
                    batch_size,channels,patch_y_shape,patch_x_shape = patches[p][:,:,:-self.overlap,:].shape
                elif (x_patches > 1) and (y_patches == 1):
                    batch_size,channels,patch_y_shape,patch_x_shape = patches[p][:,:,:,:-self.overlap].shape
                else:
                    batch_size,channels,patch_y_shape,patch_x_shape = patches[p][:,:,:-self.overlap,:-self.overlap].shape
                img = torch.zeros((batch_size,channels,y_patches*patch_y_shape,x_patches*patch_x_shape),dtype=torch.float32)
            if 1<y_patches<y_patches*patch_y_shape:
                if j == 0:
                    patches[p] = patches[p][:,:,:-self.overlap,:]
                elif j == y_patches-1:
                    patches[p] = patches[p][:,:,self.overlap:,:]
                else:
                    patches[p] = patches[p][:,:,self.overlap:-self.overlap,:]
            elif patch_y_shape == 1:
                if j < self.overlap:
                    patches[p] = patches[p][:,:,j:-self.overlap,:]
                elif j > y_patches-self.overlap-1:
                    if (j+1-y_patches) < 0:
                        patches[p] = patches[p][:,:,self.overlap:j+1-y_patches,:]
                    else:
                        patches[p] = patches[p][:,:,self.overlap:,:]
                else:
                    patches[p] = patches[p][:,:,self.overlap:-self.overlap,:]
            if 1<x_patches<x_patches*patch_x_shape:
                if i == 0:
                    patches[p] = patches[p][:,:,:,:-self.overlap]
                elif i == x_patches-1:
                    patches[p] = patches[p][:,:,:,self.overlap:]
                else:
                    patches[p] = patches[p][:,:,:,self.overlap:-self.overlap]
            elif patch_x_shape == 1:
                if i < self.overlap:
                    patches[p] = patches[p][:,:,:,i:-self.overlap]
                elif i > x_patches-self.overlap-1:
                    if (i+1-x_patches) < 0:
                        patches[p] = patches[p][:,:,:,self.overlap:i+1-x_patches]
                    else:
                        patches[p] = patches[p][:,:,:,self.overlap:]
                else:
                    patches[p] = patches[p][:,:,:,self.overlap:-self.overlap]
            img[:,:,j*patch_y_shape:(j+1)*patch_y_shape,i*patch_x_shape:(i+1)*patch_x_shape] = patches[p]
        return img
    def _calculate_mag(self,x):
        image_channels = x.shape[1]
        self.patch_shape = x.shape[2:]
        self.grid = self._generate_grid(x.shape[2],x.shape[3])

        x = x.squeeze(0).view(image_channels,-1).permute(1,0)
        input = torch.cat((self.grid,x),dim=1)
        return self._magnitude_vec(input)
    def _magnitude_vec(self,z):
        tmp_matrix = torch.cdist(z,z,p=self.metric)
        tmp_matrix = torch.exp(-tmp_matrix)

        mag_vec = torch.linalg.solve(tmp_matrix,torch.ones(tmp_matrix.shape[1],1)).squeeze()
        return mag_vec.view(1,*self.patch_shape)
    def _generate_grid(self,x_pixel,y_pixel):
        xx = torch.linspace(0,x_pixel-1,x_pixel)
        yy = torch.linspace(0,y_pixel-1,y_pixel)
        grid = torch.meshgrid(xx,yy,indexing='ij')
        grid_t = torch.stack(grid).view(2,-1).permute(1,0)
        return grid_t

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
    parser.add_argument('--image',action='store',default=None,type=str)
    args = parser.parse_args()

    config = {'metric':1.0,
              'overlap':2}

    os.makedirs(os.path.join('Theory','results'), exist_ok=True)

    if args.data_path != 'None':
        img_name = args.image.split('.')[0]
        img = cv2.imread(os.path.join(args.data_path,args.image))
    else:
        img_name = args.image.split('/')[-1].split('.')[0]
        img = cv2.imread(args.image)
    img = cv2.resize(img, dsize=(200,200))

    print(f'The image {args.image} has dimensions (h,w,c): {img.shape}')

    img_t = torch.from_numpy(img).permute(2,0,1)/255.
    img_t = 0.2989*img_t[0] + 0.5870*img_t[1] + 0.1140*img_t[2]
    img_t = img_t.unsqueeze(0).unsqueeze(0)

    print(f'The rescaled image {args.image} has dimensions (b,c,h,w): {img_t.shape}')

    y_shape = img_t.shape[2]
    x_shape = img_t.shape[3]


    out_df = pd.DataFrame(columns=['Model','Patches','l_inf','l_2','corr','time'])

    # This is the number of patches in each dimensio.
    for p_number in [1,2,4,8,10]:
        aggregrator = ImageTransformer(x_y_patches=(p_number,p_number),overlap=config['overlap'],metric=config['metric'])
        tic = time()
        mag_img = aggregrator.forward(img_t).squeeze().numpy()
        toc = time()
        print(f'Elapsed time during magnitude calculation: {(toc-tic):.2f}s')

        mag_img = (mag_img-mag_img.min())/(mag_img.max()-mag_img.min())
        if p_number == 1:
            gt = mag_img

        l_inf = np.max(np.abs(mag_img-gt))
        l_2 = np.linalg.norm(mag_img-gt)/np.linalg.norm(gt)
        corr = np.corrcoef(mag_img.ravel(),gt.ravel())[0,1]
        tmp_df = pd.DataFrame({'Model':'Patch','Patches':p_number,'l_inf':l_inf,'l_2':l_2,'corr':corr,'time':toc-tic},index=[0])
        out_df = pd.concat([out_df,tmp_df],axis=0,ignore_index=True)

    ########## Local ##################
    tic = time()
    local_mag_vec = local_approx(img_t.squeeze().numpy())
    toc = time()
    local_mag_vec = (local_mag_vec-local_mag_vec.min())/(local_mag_vec.max()-local_mag_vec.min())

    l_inf = np.max(np.abs(local_mag_vec-gt))
    l_2 = np.linalg.norm(local_mag_vec-gt)/np.linalg.norm(gt)
    corr = np.corrcoef(local_mag_vec.ravel(),gt.ravel())[0,1]
    tmp_df = pd.DataFrame({'Model':'Local','l_inf':l_inf,'l_2':l_2,'corr':corr,'time':toc-tic},index=[0])
    out_df = pd.concat([out_df,tmp_df],axis=0,ignore_index=True)

    ######### Rank 1 #########################

    tic = time()
    rank_1_mag_vec = rank_1_approx(img_t.squeeze().numpy())
    toc = time()
    rank_1_mag_vec = (rank_1_mag_vec-rank_1_mag_vec.min())/(rank_1_mag_vec.max()-rank_1_mag_vec.min())

    l_inf = np.max(np.abs(rank_1_mag_vec-gt))
    l_2 = np.linalg.norm(rank_1_mag_vec-gt)/np.linalg.norm(gt)
    corr = np.corrcoef(rank_1_mag_vec.ravel(),gt.ravel())[0,1]
    tmp_df = pd.DataFrame({'Model':'Rank_1','l_inf':l_inf,'l_2':l_2,'corr':corr,'time':toc-tic},index=[0])
    out_df = pd.concat([out_df,tmp_df],axis=0,ignore_index=True)

    out_df.to_csv(f'results_{img_name}.csv',index=False)

if __name__ == '__main__':
    main()
