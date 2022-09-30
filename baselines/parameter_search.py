import os
from time import time, sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cv2
from glob import glob
from sklearn.metrics import log_loss, jaccard_score, mean_absolute_error
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from joblib import Parallel,delayed

def experiment(params,args,data_list,labels_list):
    if (args.method == 'canny') and (params['threshold1'] > params['threshold2']):
        return {**params,'final_score_bce':np.inf,'final_score_mae':np.inf}
    score = []
    for image,label in zip(data_list,labels_list):
        img = cv2.imread(image)
        lbl = cv2.imread(label).mean(axis=2)
        lbl = (lbl > 0).astype(int)
        img_name = image.split('/')[-1].strip('.jpg')

        img = img.transpose((2,0,1))
        img_grey = 0.2989*img[0] + 0.5870*img[1] + 0.1140*img[2]

        if args.method == 'sobel':
            img_blur = cv2.GaussianBlur(img_grey, (params['blur_size'],params['blur_size']), 0)
            dx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=params['ksize'])
            dy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=params['ksize'])
            edge_map = np.sqrt(dx**2+dy**2)
            edge_map =  (edge_map-edge_map.min())/(edge_map.max()-edge_map.min())
        else:
            edge_map = cv2.Canny(image=np.round(img_grey).astype('uint8'), threshold1=params['threshold1'], threshold2=params['threshold2'],apertureSize=params['apertureSize'],L2gradient=params['L2gradient'])
        mae_loss = mean_absolute_error(lbl[lbl==1].reshape(-1,1),edge_map[lbl==1].reshape(-1,1))+mean_absolute_error(lbl[lbl==0].reshape(-1,1),edge_map[lbl==0].reshape(-1,1))
        score.append([log_loss(lbl.reshape(-1,1),edge_map.reshape(-1,1),eps=1e-7),mae_loss])
    return {**params,'final_score_bce':np.array(score)[:,0].mean(),'final_score_mae':np.array(score)[:,1].mean()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store',default='.',type=str)
    parser.add_argument('--labels_path', action='store',default='.',type=str)
    parser.add_argument('--version', action='store', default='real',type=str)
    parser.add_argument('--method',default='canny',type=str)
    parser.add_argument('--evaluate_only',action='store_true')
    args = parser.parse_args()

    if not args.evaluate_only:
        if args.version == 'aug':
            data_list = glob(os.path.join(args.data_path,'train','rgbr',args.version,'*','*.jpg'))
            labels_list = glob(os.path.join(args.labels_path,'train','rgbr',args.version,'*','*.png'))
        elif args.version == 'real':
            data_list = glob(os.path.join(args.data_path,'train','rgbr',args.version,'*.jpg'))
            labels_list = glob(os.path.join(args.labels_path,'train','rgbr',args.version,'*.png'))
        data_list.sort()
        labels_list.sort()
        assert len(data_list) == len(labels_list)
        assert all([data_list[i].split('/')[-1].strip('.jpg') == labels_list[i].split('/')[-1].strip('.png') for i in range(len(data_list))])

        param_grid_canny = {'threshold1':list(range(0,260,10)),'threshold2':list(range(0,260,10)),'apertureSize':[3,5,7],'L2gradient':[True,False]}

        # Best practise configurations for the Canny edge detector which we found in the literature
        # canny_param_list = [{'threshold1':int(255/3),'threshold2':255,'apertureSize':3,'L2gradient':True},
        #                     {'threshold1':int(255*0.4),'threshold2':255,'apertureSize':3,'L2gradient':True},
        #                     {'threshold1':int(200/3),'threshold2':200,'apertureSize':3,'L2gradient':True},
        #                     {'threshold1':int(200*0.4),'threshold2':200,'apertureSize':3,'L2gradient':True},
        #                     {'threshold1':int(255*0.1),'threshold2':int(255*0.2),'apertureSize':3,'L2gradient':True},
        #                     ]

        df_items = Parallel(n_jobs=8,verbose=1)(delayed(experiment)(params,args,data_list,labels_list) for params in ParameterGrid(eval(f'param_grid_{args.method}')))

        final_scores_df = pd.DataFrame(df_items)
        final_scores_df.to_csv(os.path.join('output',f'scores_{args.method}.csv'),index=False)
    else:
        final_scores_df = pd.read_csv(os.path.join('output',f'scores_{args.method}.csv'))

    # Print the best configuration
    print(final_scores_df[final_scores_df['final_score_mae']==final_scores_df['final_score_mae'].min()])

    best_params = final_scores_df[final_scores_df['final_score_mae']==final_scores_df['final_score_mae'].min()].iloc[0].to_dict()
    test_data_list = glob(os.path.join(args.data_path,'test','rgbr','real','*.jpg'))
    test_labels_list = glob(os.path.join(args.labels_path,'test','rgbr','*.png'))
    test_data_list.sort()
    test_labels_list.sort()
    assert len(test_data_list) == len(test_labels_list)
    assert all([test_data_list[i].split('/')[-1].strip('.jpg') == test_labels_list[i].split('/')[-1].strip('.png') for i in range(len(test_data_list))])
    test_names = [d.split('/')[-1].strip('.jpg') for d in test_data_list]

    os.makedirs(os.path.join('results',f'bipedv2_{args.method}','edges_pred'), exist_ok=True)
    for image,label in zip(test_data_list,test_labels_list):
        img = cv2.imread(image)
        lbl = cv2.imread(label).mean(axis=2)
        lbl = (lbl > 0).astype(int)
        img_name = image.split('/')[-1].strip('.jpg')

        img = img.transpose((2,0,1))
        img_grey = 0.2989*img[0] + 0.5870*img[1] + 0.1140*img[2]

        if args.method == 'sobel':
            img_blur = cv2.GaussianBlur(img_grey, (5,5), 0)
            dx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
            dy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
            edge_map = np.sqrt(dx**2+dy**2)

            # Min-max scaling and converting back to an 8 bit image.
            edge_map =  255.*(edge_map-edge_map.min())/(edge_map.max()-edge_map.min())
        else:
            edge_map = cv2.Canny(image=np.round(img_grey).astype('uint8'), threshold1=best_params['threshold1'], threshold2=best_params['threshold2'],apertureSize=best_params['apertureSize'],L2gradient=best_params['L2gradient'])

        cv2.imwrite(os.path.join('results',f'bipedv2_{args.method}','edges_pred',f'{test_names.pop(0)}.png'), edge_map)

if __name__ == '__main__':
    main()
