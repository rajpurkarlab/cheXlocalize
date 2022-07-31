"""
When evaluating mIoU on the full dataset, we ensure that the final binary segmentation is consistent with model probability output by applying another layer of thresholding 
such that the segmentation mask is all zeros if the predicted probability was below a chosen level. The probability threshold is searched on the interval of [0,0.8] with steps of 0.1. 
The exact value is determined per pathology by maximizing the mIoU on the validation set. 
"""

import pickle
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
from utils import *
from pycocotools import mask
from eval_full_dataset import calculate_iou
from heatmap_to_segmentation import *
from eval_constants import LOCALIZATION_TASKS
from tqdm import tqdm
import torch
from argparse import ArgumentParser


def compute_miou(cutoff, pkl_paths,gt):
    """
    Caculate miou given a threshold and a list of pkl paths
    """
    ious = []

    for pkl_path in tqdm(pkl_paths):
        # get gradcam segmentation 
        info = pickle.load(open(pkl_path,'rb'))
        img_dims = info['cxr_dims']
        map_resized = F.interpolate(info['map'], size=(img_dims[1],img_dims[0]), mode='bilinear', align_corners=False)
        if torch.is_tensor(info['prob']) and info['prob'].size()[0] == 14:
            prob_idx = CHEXPERT_TASKS.index(info['task'])
            pred_prob = info['prob'][prob_idx]
        else:
            pred_prob = info['prob']
       
        if pred_prob > cutoff:
            segm = cam_to_segmentation(map_resized)
            gradcam_mask = np.array(segm)
        else:
            gradcam_mask = np.zeros((img_dims[1],img_dims[0]))

        # get gt segmentation
        path = str(pkl_path).split('/')
        task = path[-1].split('_')[-2]
        img_id = '_'.join(path[-1].split('_')[:-2])

        if img_id in gt:
            pred_item = gt[img_id][task]
            pred_mask = mask.decode(pred_item)
        else:
            pred_mask = np.zeros((img_dims[1],img_dims[0]))

        iou = calculate_iou(gradcam_mask,pred_mask)
        ious.append(iou)

    miou = np.nanmean(np.array(ious))
    return miou


def find_threshold(task,gt, cam_dir):
    """
    For a given task, find the probability threshold with max miou (on validation)
    """
    cam_pkl = sorted(list(Path(cam_dir).rglob(f"*{task}_map.pkl")))
    cutoffs = np.arange(0.1,.9,.1)
    mious = [compute_miou(cutoff,cam_pkl,gt) for cutoff in cutoffs ]
    cutoff = cutoffs[mious.index(max(mious))]
    print(f"cutoff: {cutoffs}; iou: {mious}")
    return cutoffs, mious



if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--map_dir', type=str,
                        help='directory with pickle files containing heat maps and model output')
    parser.add_argument('--gt_path', type=str,
                        help='json file where ground-truth segmentations are \
                              saved (encoded)')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='where to save the probability threshold tuned on the \
                              validation set')
    
    args = parser.parse_args()
    
    # read gt masks
    with open(args.gt_path) as f:
        gt_dict = json.load(f)
    
    # tune probability threshold
    tuning_results = pd.DataFrame(columns=['prob_threshold','mIoU','task'])
    for task in sorted(LOCALIZATION_TASKS):
        print(f"Task: {task}")
        cutoff, miou = find_threshold(task,gt_dict,args.map_dir)
        df = pd.concat([pd.DataFrame([[round(cutoff[i],1), round(miou[i],3), task]], columns=['prob_threshold','mIoU','task']) for i in range(len(cutoff))],ignore_index=True)
        tuning_results = tuning_results.append(df, ignore_index = True)
    
    tuning_results.to_csv(f'{args.save_dir}/probability_tuning_results.csv', index = False)

