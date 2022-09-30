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
from tqdm import tqdm

from data_loader import PatchImgDatamodule
from model import PullbackModel, ImageTransformer

import yaml

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,action='store',required=False,default='yaml')
    parser.add_argument('--config',type=str,action='store',required=False,default='config')
    parser.add_argument('--dev_mode',action='store_true')
    args = parser.parse_args()

    # Import the configs
    with open(f'{args.config}.yaml','r') as f:
        configs = yaml.safe_load(f)
    assert configs['data_loading_params']['overlap'] == configs['model_params']['overlap']
    if args.model == 'yaml':
        model_name = configs['model_params']['model_name']
    else:
        model_name = args.model

    # Set up the data module
    data = PatchImgDatamodule(**configs['data_loading_params'])

    # Define the model
    model = PullbackModel(configs['model_params'])

    # Log using Weights & Biases (optional)
    wandb_logger = WandbLogger()

    # Checkpoint callback to find the best model
    checkpoint_callback = ModelCheckpoint(dirpath='lightning_logs',
                                          filename=f'Model_{model_name}',
                                          monitor='val_loss',
                                          mode='min',
                                          save_top_k=1,
                                          every_n_epochs=1)


    trainer = pl.Trainer(max_epochs=100,
                         callbacks=[checkpoint_callback],
                         logger=wandb_logger,
                         log_every_n_steps = 1,
                         fast_dev_run=args.dev_mode,
                         enable_progress_bar=True,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=1 if torch.cuda.is_available() else 0)
    trainer.fit(model,datamodule=data)

    # Now transform the test set if we are not debugging the model
    if not args.dev_mode:
        overlap = configs['model_params']['overlap']
        model = model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path).cpu()

        aggregrator = ImageTransformer(model,x_y_patches=(32,18),overlap=overlap)
        os.makedirs(os.path.join('results',f'bipedv2_mag_patch_{model_name}','edges_pred'), exist_ok=True)

        for d in tqdm(data.test_data_set.data_list):
             img = cv2.imread(d)
             img = cv2.GaussianBlur(img, (5,5), 0)
             img = torch.from_numpy(img).permute(2,0,1)/255.
             img = img.unsqueeze(0)

             img_shape = img.shape[2:]
             img = F.resize(img,(img_shape[0]-2*overlap,img_shape[1]-2*overlap),antialias=True)
             img = F.pad(img,padding=overlap,padding_mode='reflect')

             y_shape = img.shape[2]
             x_shape = img.shape[3]

             mag_img = aggregrator.forward(img).detach().squeeze(0).view(-1,y_shape,x_shape).permute(1,2,0).numpy()
             edge_map = cv2.resize(255*(1-mag_img[overlap:-overlap,overlap:-overlap,:]), dsize=(img_shape[1],img_shape[0]))

             name = d.split('/')[-1].strip('.jpg')
             cv2.imwrite(os.path.join('results',f'bipedv2_mag_patch_{model_name}','edges_pred',f'{name}.png'), edge_map)

if __name__ == '__main__':
    main()
