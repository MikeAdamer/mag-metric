import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_img
from torchvision import datasets, transforms, models
import pytorch_lightning as pl
import os

import numpy as np
from numpy import corrcoef
from tqdm import tqdm

import cv2

class ImageTransformer(pl.LightningModule):
    def __init__(self,mag_model,x_y_patches=(2,2),overlap=1):
        super().__init__()
        self.mag_model = mag_model
        self.x_y_patches = x_y_patches
        self.overlap = overlap
        self.scaler = MinMaxLayer()
    def forward(self,x):
        patches,patch_grid = self._make_patches(x)
        patches_vecs = []
        for p in tqdm(patches,desc='Patches',leave=False):
            tmp = p.contiguous()#.to(self.device)
            with torch.no_grad():
                mag_patch = self.mag_model.forward(tmp).unsqueeze(0)
            patches_vecs.append(mag_patch)
        mag_img_approx = self._stitch_patches(patches_vecs,patch_grid)
        return self.scaler(torch.abs(mag_img_approx))#torch.clamp(mag_img_approx,min=0.)
    def forward_mag_only(self,x):
        patches,patch_grid = self._make_patches(x)
        patches_vecs = []
        for p in tqdm(patches,desc='Patches',leave=False):
            tmp = p.contiguous()
            with torch.no_grad():
                mag_patch = self.mag_model.forward_mag_only(tmp).unsqueeze(0)
            patches_vecs.append(mag_patch)
        mag_img_approx = self._stitch_patches(patches_vecs,patch_grid)
        return self.scaler(torch.abs(mag_img_approx))
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
                img = torch.zeros((batch_size,channels,y_patches*patch_y_shape,x_patches*patch_x_shape),dtype=torch.float32,device=self.device)
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

'''
Pullback models
'''

class PullbackModel(pl.LightningModule):
    def __init__(self,params):
        super().__init__()
        self.params = params
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.feature_extractor = eval(params['feature_extractor'])
        self.grid = None
        self.patch_shape = None
        self.save_hyperparameters()

    def _build_encoder(self):
        # Set the layer sizes according to the config file
        layer_sizes = [self.params['input_dim']]+self.params['hidden_sizes']
        layers = []

        # Set the activation function
        if 'activation' in self.params.keys():
            activation = self.params['activation']
            activation = eval(f'nn.{activation}()')
        else:
            activation = nn.ReLU()

        # Build the first n-1 non-linear layers
        for i in range(len(layer_sizes)-2):
            layers.extend([nn.Linear(layer_sizes[i],layer_sizes[i+1]),
                               activation,
                               nn.Dropout(self.params['p'])])

        # Output layer
        layers.append(nn.Linear(layer_sizes[-2],layer_sizes[-1]))

        return nn.Sequential(*layers)

    def _build_decoder(self):
        # Set the layer sizes according to the config file
        layer_sizes = [self.params['input_dim']]+self.params['hidden_sizes']
        layer_sizes.reverse()
        layers = []

        # Set the activation function
        if 'activation' in self.params.keys():
            activation = self.params['activation']
            activation = eval(f'nn.{activation}()')
        else:
            activation = nn.ReLU()

        # Build the first n-1 non-linear layers
        for i in range(len(layer_sizes)-2):
            layers.extend([nn.Linear(layer_sizes[i],layer_sizes[i+1]),
                               activation,
                               nn.Dropout(self.params['p'])])

        # Output layer
        layers.append(nn.Linear(layer_sizes[-2],layer_sizes[-1]))

        return nn.Sequential(*layers)

    def _generate_grid(self,x_pixel,y_pixel):
        xx = torch.linspace(0,x_pixel-1,x_pixel,device=self.device)
        yy = torch.linspace(0,y_pixel-1,y_pixel,device=self.device)
        grid = torch.meshgrid(xx,yy,indexing='ij')
        grid_t = torch.stack(grid).view(2,-1).permute(1,0)
        return grid_t

    def _magnitude_vec(self,z):
        tmp_matrix = torch.cdist(z,z,p=self.params['metric'])
        tmp_matrix = torch.exp(-tmp_matrix)

        mag_vec = torch.linalg.solve(tmp_matrix,torch.ones(tmp_matrix.shape[1],1,device=self.device)).squeeze()
        return mag_vec.view(1,*self.patch_shape)

    def forward(self,x):
        # During inference this might change
        self.patch_shape = x.shape[2:]
        self.grid = self._generate_grid(x.shape[2],x.shape[3])

        if self.feature_extractor is not None:
            features = self.feature_extractor(x)
            x = x.squeeze(0).view(3,-1).permute(1,0)
            input = torch.cat((self.grid,x,features),dim=1)
        else:
            x = x.squeeze(0).view(3,-1).permute(1,0)
            input = torch.cat((self.grid,x),dim=1)
        z = self.encoder(input)
        return self._magnitude_vec(z)

    def forward_mag_only(self,x):
        # Just the vanilla mangitude calculation
        # During inference this might change
        image_channels = x.shape[1]
        self.patch_shape = x.shape[2:]
        self.grid = self._generate_grid(x.shape[2],x.shape[3])

        x = x.squeeze(0).view(image_channels,-1).permute(1,0)
        input = torch.cat((self.grid,x),dim=1)
        return self._magnitude_vec(input)

    def training_step(self,batch,batch_idx):
        x, y = batch

        # During training the patch shape doesn't change
        if self.patch_shape is None:
            self.patch_shape = x.shape[2:]
        if self.grid is None:
            self.grid = self._generate_grid(x.shape[2],x.shape[3])

        if self.feature_extractor is not None:
            features = self.feature_extractor(x)
            x = x.squeeze(0).view(3,-1).permute(1,0)
            input = torch.cat((self.grid,x,features),dim=1)
        else:
            x = x.squeeze(0).view(3,-1).permute(1,0)
            input = torch.cat((self.grid,x),dim=1)
        z = self.encoder(input)
        mag_vec = self._magnitude_vec(z)[:,self.params['overlap']:-self.params['overlap'],self.params['overlap']:-self.params['overlap']]
        out = self.decoder(z)

        ae_loss = F.mse_loss(input,out,reduction='mean')
        if y[y==1].shape[0]>0:
            mag_loss_1 = F.mse_loss(mag_vec[y==1],y[y==1])
            mag_loss_0 = F.l1_loss(mag_vec[y==0],y[y==0])
            mag_loss = mag_loss_1 + mag_loss_0
            self.log('mag_loss_0',mag_loss_0)
            self.log('mag_loss_1',mag_loss_1)
        else:
            mag_loss = F.l1_loss(mag_vec,y)

        loss = self.params['l_mag']*mag_loss + ae_loss #+ hed_loss

        self.log('train_loss',loss)
        self.log('ae_train_loss',ae_loss)
        self.log('mag_train_loss',mag_loss)
        return loss

    def validation_step(self,batch,batch_idx):
        x, y = batch

        # During validation the patch shape doesn't change
        # Shouldn't be necessary
        if self.patch_shape is None:
            self.patch_shape = x.shape[2:]
        if self.grid is None: # Ditto, not necessary
            self.grid = self._generate_grid(x.shape[2],x.shape[3])

        if self.feature_extractor is not None:
            features = self.feature_extractor(x)
            x = x.squeeze(0).view(3,-1).permute(1,0)
            input = torch.cat((self.grid,x,features),dim=1)
        else:
            x = x.squeeze(0).view(3,-1).permute(1,0)
            input = torch.cat((self.grid,x),dim=1)
        z = self.encoder(input)
        mag_vec = self._magnitude_vec(z)[:,self.params['overlap']:-self.params['overlap'],self.params['overlap']:-self.params['overlap']]
        out = self.decoder(z)

        ae_loss = F.mse_loss(input,out,reduction='mean')
        if y[y==1].shape[0]>0:
            mag_loss = F.l1_loss(mag_vec[y==1],y[y==1]) + F.l1_loss(mag_vec[y==0],y[y==0])
        else:
            mag_loss = F.l1_loss(mag_vec,y)
        self.log('ae_val_step__loss',ae_loss)
        self.log('mag_val_step_loss',mag_loss)
        return {'y':y,'y_hat':mag_vec,'ae_loss':ae_loss}

    def validation_epoch_end(self, outputs):
        y = torch.stack([x['y'] for x in outputs]).detach()
        y_hat = torch.stack([x['y_hat'] for x in outputs]).detach()
        ae_loss = torch.stack([x['ae_loss'] for x in outputs]).detach().mean()
        MAE = F.l1_loss(y,y_hat)
        rs = np.array([corrcoef(y[i].view(-1).cpu().numpy(),y_hat[i].view(-1).cpu().numpy())[0,1] for i in range(y.shape[0])])
        self.log('val_loss',ae_loss+MAE)
        self.log('mag_val_loss',MAE)
        self.log('mag_val_r',np.mean(rs[~np.isnan(rs)]))

    def test_step(self,batch,batch_idx):
        x, y = batch

        # During testing the patch shape doesn't change
        # Shouldn't be necessary
        if self.patch_shape is None:
            self.patch_shape = x.shape[2:]
        if self.grid is None: # Ditto, not necessary
            self.grid = self._generate_grid(x.shape[2],x.shape[3])

        if self.feature_extractor is not None:
            features = self.feature_extractor(x)
            x = x.squeeze(0).view(3,-1).permute(1,0)
            input = torch.cat((self.grid,x,features),dim=1)
        else:
            x = x.squeeze(0).view(3,-1).permute(1,0)
            input = torch.cat((self.grid,x),dim=1)
        z = self.encoder(input)
        mag_vec = self._magnitude_vec(z)[:,self.params['overlap']:-self.params['overlap'],self.params['overlap']:-self.params['overlap']]
        out = self.decoder(z)

        ae_loss = F.mse_loss(input,out,reduction='mean')
        mag_loss = F.l1_loss(mag_vec,y)
        self.log('ae_test_step__loss',ae_loss)
        self.log('mag_test_step_loss',mag_loss)
        return {'y':y,'y_hat':mag_vec}

    def test_epoch_end(self, outputs):
        y = torch.stack([x['y'] for x in outputs]).detach()
        y_hat = torch.stack([x['y_hat'] for x in outputs]).detach()
        MAE = F.l1_loss(y,y_hat)
        rs = np.array([corrcoef(y[i].view(-1).cpu().numpy(),y_hat[i].view(-1).cpu().numpy())[0,1] for i in range(y.shape[0])])
        self.log('mag_test_loss',MAE)
        self.log('mag_test_r',np.mean(rs[~np.isnan(rs)]))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.params['lr'],eps=1e-08,weight_decay=self.params['l2'])
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "monitor": "val_loss",
            "frequency": 1
            },
        }

'''
Postprcessing Layers
'''
class MinMaxLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        mins = x.view(x.shape[0],x.shape[1],-1).min(-1).values.view(x.shape[0],x.shape[1],1,1)
        maxs = x.view(x.shape[0],x.shape[1],-1).max(-1).values.view(x.shape[0],x.shape[1],1,1)
        return (x-mins)/(maxs-mins)

'''
    Feature Extractors
'''
class SimpleConv(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Conv2d(3,15,kernel_size=3,padding=1,padding_mode='reflect')
    def forward(self,x):
        return self.model(x).squeeze(0).view(15,-1).permute(1,0)
