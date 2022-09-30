'''
In this script we can use an already trained model to transform the test set.
'''

import os
from time import time
import numpy as np
import torch
import torchvision.transforms.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import argparse
import cv2
import yaml
from tqdm import tqdm

from data_loader import PatchImgDatamodule
from model import PullbackModel, ImageTransformer

def main():
    # Import the configs
    with open('config.yaml','r') as f:
        configs = yaml.safe_load(f)

    overlap = configs['model_params']['overlap']

    # Set up the datamodule
    data = PatchImgDatamodule(**configs['data_loading_params'])
    data.setup()

    # Define the model
    model = PullbackModel(configs['model_params'])

    # Load the trained weights
    model_name = configs['model_params']['model_name']
    model.load_state_dict(torch.load(f'lightning_logs/{model_name}.ckpt')['state_dict'])

    # Here, we give the number of patches in each dimension
    aggregrator = ImageTransformer(model,x_y_patches=(32,18),overlap=overlap)

    os.makedirs(os.path.join('results','bipedv2_mag_patch'), exist_ok=True)

    # Transform the test set.
    for i in tqdm(range(len(data.test_data_set.data_list))):
         d = data.test_data_set.data_list[i]
         img = cv2.imread(d)
         img = cv2.GaussianBlur(img, (5,5), 0)
         img = torch.from_numpy(img).permute(2,0,1)/255.
         img = img.unsqueeze(0)

         img_shape = img.shape[2:]

         # The resizing is necessary as adding an overlap would alter the patches
         # The number of pixels in each dimension needs to be divisible by the number of patches
         img = F.resize(img,(img_shape[0]-2*overlap,img_shape[1]-2*overlap),antialias=True)

         # We pad to account for boundary effects
         img = F.pad(img,padding=overlap,padding_mode='reflect')

         y_shape = img.shape[2]
         x_shape = img.shape[3]

         # Calculate the edge map and reshape to the correct output form
         mag_img = aggregrator.forward(img).detach().squeeze(0).view(-1,y_shape,x_shape).permute(1,2,0).numpy()

         # Invert colours (we want black edges for better visibility)
         # Then convert back to 8-bit image
         edge_map = cv2.resize(255*(1-mag_img[overlap:-overlap,overlap:-overlap,:]), dsize=(img_shape[1],img_shape[0]))

         # save as png
         name = d.split('/')[-1].strip('.jpg')
         cv2.imwrite(os.path.join('results','bipedv2_mag_patch','mag_only',f'{name}.png'), edge_map)

if __name__ == '__main__':
    main()
