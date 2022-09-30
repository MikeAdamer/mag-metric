import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from glob import glob
import cv2
import os
from multiprocessing import cpu_count

class PatchImgDatamodule(pl.LightningDataModule):
    def __init__(self,data_path='.',
                      labels_path='.',
                      patch_shape=[20,20],
                      overlap=1,
                      val_set=True):
        super().__init__()
        self.data_path = data_path
        self.labels_path = labels_path
        self.patch_shape = patch_shape
        self.overlap = overlap
        self.batch_size = 1
        self.val_set = val_set
    def setup(self,stage=None):
        self.train_data_set = ImageDataset(data_path=self.data_path,
                                labels_path=self.labels_path,
                                phase='train',
                                patch_shape=self.patch_shape,
                                overlap=self.overlap)
        self.test_data_set = ImageDataset(data_path=self.data_path,
                                labels_path=self.labels_path,
                                phase='test',
                                patch_shape=self.patch_shape,
                                overlap=self.overlap)
        if self.val_set:
            self.train_idx, self.val_idx = train_test_split(list(range(len(self.train_data_set))),test_size=0.2)
    def train_dataloader(self):
        if self.val_set:
            train_sampler = torch.utils.data.SubsetRandomSampler(self.train_idx)
            # 3 if self.batch_size>1 else 0
            return DataLoader(self.train_data_set, batch_size=1,shuffle=False,num_workers=cpu_count(),sampler=train_sampler)
        else:
            return DataLoader(self.train_data_set, batch_size=1,shuffle=False,num_workers=cpu_count())
    def val_dataloader(self):
        if self.val_set:
            return DataLoader(self.train_data_set, batch_size=1,shuffle=False,num_workers=cpu_count(),sampler=self.val_idx)
        else:
            pass
    def test_dataloader(self):
        return DataLoader(self.test_data_set, batch_size=1,shuffle=False,num_workers=cpu_count())

class SingleShotDatamodule(pl.LightningDataModule):
    def __init__(self,img_name,
                      data_path='.',
                      labels_path='.',
                      patch_shape=[20,20],
                      overlap=1,
                      val_set=True):
        super().__init__()
        self.img_name = img_name
        self.data_path = data_path
        self.labels_path = labels_path
        self.patch_shape = patch_shape
        self.overlap = overlap
        self.batch_size = 1
        self.val_set = val_set
    def setup(self,stage=None):
        self.train_data_set = SingleShotImageDataset(self.img_name,
                                data_path=self.data_path,
                                labels_path=self.labels_path,
                                patch_shape=self.patch_shape,
                                overlap=self.overlap)
        self.test_data_set = ImageDataset(data_path=self.data_path,
                                labels_path=self.labels_path,
                                phase='test',
                                patch_shape=self.patch_shape,
                                overlap=self.overlap)
        if self.val_set:
            self.train_idx, self.val_idx = train_test_split(list(range(len(self.train_data_set))),test_size=0.2)
    def train_dataloader(self):
        if self.val_set:
            train_sampler = torch.utils.data.SubsetRandomSampler(self.train_idx)
            # 3 if self.batch_size>1 else 0
            return DataLoader(self.train_data_set, batch_size=1,shuffle=False,num_workers=cpu_count(),sampler=train_sampler)
        else:
            return DataLoader(self.train_data_set, batch_size=1,shuffle=False,num_workers=cpu_count())
    def val_dataloader(self):
        if self.val_set:
            return DataLoader(self.train_data_set, batch_size=1,shuffle=False,num_workers=cpu_count(),sampler=self.val_idx)
        else:
            pass
    def test_dataloader(self):
        return DataLoader(self.test_data_set, batch_size=1,shuffle=False,num_workers=cpu_count())

class ImageDataset(Dataset):
    def __init__(self,data_path='.',
                      labels_path='.',
                      phase='train',
                      patch_shape=(20,20),
                      overlap=1):
        super().__init__()
        self.patch_shape = patch_shape
        self.overlap = overlap
        if phase == 'train':
            self.data_list = glob(os.path.join(data_path,phase,'rgbr','real','*.jpg'))
            self.labels_list = glob(os.path.join(labels_path,phase,'rgbr','real','*.png'))
        elif phase == 'test':
            self.data_list = glob(os.path.join(data_path,phase,'rgbr','real','*.jpg'))
            self.labels_list = glob(os.path.join(labels_path,phase,'rgbr','*.png'))
        self.data_list.sort()
        self.labels_list.sort()
        assert len(self.data_list) == len(self.labels_list)
        assert all([self.data_list[i].split('/')[-1].strip('.jpg') == self.labels_list[i].split('/')[-1].strip('.png') for i in range(len(self.data_list))])


    @staticmethod
    def sample_random_patch(img,patch_shape,overlap):
        c,h,w = img.shape
        h_p, w_p = patch_shape

        return torch.randint(low=overlap,high=h-h_p-overlap,size=(1,)).item(), torch.randint(low=overlap,high=w-w_p-overlap,size=(1,)).item()

    def __getitem__(self,idx):
        # Read image
        img = cv2.imread(self.data_list[idx])
        img = cv2.GaussianBlur(img, (5,5), 0)
        img = torch.from_numpy(img).permute(2,0,1)/255.

        # pad to mitigate boundary effects
        img = F.pad(img,padding=self.overlap,padding_mode='reflect')

        # Read label
        label = cv2.imread(self.labels_list[idx])
        # label = cv2.GaussianBlur(label, (5,5), 0)
        label = torch.from_numpy(label).permute(2,0,1)/255.
        label = label.mean(dim=0,keepdim=False)

        # Create patch
        h_start,w_start = self.sample_random_patch(img,self.patch_shape,self.overlap)
        img_patch = img[:,h_start-self.overlap:h_start+self.patch_shape[0]+self.overlap,w_start-self.overlap:w_start+self.patch_shape[1]+self.overlap]
        label_patch = label[h_start-self.overlap:h_start+self.patch_shape[0]-self.overlap,w_start-self.overlap:w_start+self.patch_shape[1]-self.overlap]
        return (img_patch,label_patch)

    def __len__(self):
        return len(self.data_list)

class SingleShotImageDataset(Dataset):
    def __init__(self,img_name,
                      data_path='.',
                      labels_path='.',
                      patch_shape=(20,20),
                      overlap=1):
        super().__init__()
        self.patch_shape = patch_shape
        self.overlap = overlap
        self.data_list = self._make_patches(os.path.join(data_path,'train','rgbr','real',img_name+'.jpg'))
        self.labels_list = self._make_patches(os.path.join(labels_path,'train','rgbr','real',img_name+'.png'))


    def _make_patches(self,img_name):
        img = cv2.imread(img_name)

        # Blur only the image not the label
        if img_name.split('.')[-1] == 'jpg':
            img = cv2.GaussianBlur(img, (5,5), 0)

        img = torch.from_numpy(img).permute(2,0,1)/255.

        self.n_channels,height,width = img.shape
        x_patch_size, y_patch_size = self.patch_shape
        x_patches = int(width/x_patch_size)
        y_patches = int(height/y_patch_size)

        if (width/x_patch_size)%1 != 0.0:
            raise ValueError('The image height must be divisible by the number of patches!')
        elif (height/y_patch_size)%1 != 0.0:
            raise ValueError('The image width must be divisible by the number of patches!')

        # img = F.resize(img,(height-2*self.overlap,width-2*self.overlap),antialias=True)
        img = F.pad(img,padding=self.overlap,padding_mode='reflect')

        patches = []

        for i in range(x_patches):
            for j in range(y_patches):
                patches.append(img[:,j*y_patch_size:(j+1)*y_patch_size+2*self.overlap,i*x_patch_size:(i+1)*x_patch_size+2*self.overlap])
                # patches.append(img[:,max(j*y_patch_size-self.overlap,0):min((j+1)*y_patch_size+self.overlap,height),max(i*x_patch_size-self.overlap,0):min((i+1)*x_patch_size+self.overlap,width)])
        return patches

    def __getitem__(self,idx):
        return (self.data_list[idx],self.labels_list[idx].mean(dim=0,keepdim=False)[self.overlap:-self.overlap,self.overlap:-self.overlap])

    def __len__(self):
        return len(self.data_list)
