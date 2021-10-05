import pickle
import glob
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from pycocotools import mask
from eval import iou_seg
from eval_constants import LOCALIZATION_TASKS


from argparse import ArgumentParser


def pkl_to_mask(pkl_path, cutoff):
    """
    # load cam pickle file, get saliency map and resize. 
    Return the binary segmentation mask 
    
    Args:
        pkl_path(str): path to the pickle file
        task(str): pathology
        cutoff: threshold for creating binary segmentation
    """
    # load pickle file, get saliency map and resize
    info = pickle.load(open(pkl_path,'rb'))
    saliency_map = info['map']
    img_dims = info['cxr_dims']
    map_reshaped = F.interpolate(saliency_map, size=(img_dims[1],img_dims[0]), mode='bilinear', align_corners=False)
    map_squeezed = map_reshaped.squeeze()
    m = map_squeezed - map_squeezed.min()
    m = m.div(m.max()).data
    m = m.cpu().numpy()
    
    segm_mask = np.array(m > cutoff,dtype = "int")
    return segm_mask

def compute_miou(cutoff, cam_pkls, gt):
    
    ious = []

    for pkl_path in tqdm(cam_pkls):

        # break down path to image name and task
        path = str(pkl_path).split('/')
        task = path[-1].split('_')[-2]
        img_id = '_'.join(path[-1].split('_')[:-2])
        
        # add image and segmentation to submission dictionary
        if img_id in gt:
            
            pred_mask = pkl_to_mask(pkl_path = pkl_path, cutoff = cutoff)
            gt_item = gt[img_id][task]
            gt_mask = mask.decode(gt_item)
            assert(pred_mask.shape == gt_mask.shape)
            iou_score = iou_seg(pred_mask, gt_mask, tp = True)
        else:
            iou_score = np.nan 
        ious.append(iou_score)
        
    miou = np.nanmean(np.array(ious))
    return miou


def find_threshold(task, gt, cam_dir):
    """
    For a given task, find the saliency map threshold with max miou
    """
    cam_pkls = sorted(list(Path(cam_dir).rglob(f"*{task}_map.pkl")))
    cutoffs = np.arange(0.2,.8,.1)
    mious = [compute_miou(cutoff,cam_pkls,gt) for cutoff in cutoffs ]
    cutoff = cutoffs[mious.index(max(mious))]
    print(f"cutoff: {cutoffs}; iou: {mious}")
    return cutoffs, mious


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--method', type=str, default = 'gradcam',
                        help='gradcam, gradcampp, ig')
    parser.add_argument('--model_type', default='ensemble',
                        help='single or ensemble')
    parser.add_argument('--model', default='densenet',
                        help='densenet, inception, resnet')
    
    args = parser.parse_args()
    
    method = args.method
    model_type = args.model_type
    model = args.model
    
    result_dir = 'threshold_tuning_results'
    all_cam_dir = "/deep/u/ashwinagrawal/results_0713"
    cam_dir = f'{all_cam_dir}/{model}_{method}_val/ensemble_results/cams/'

    # read gt masks
    print("Read gt masks")
    group_dir = '/deep/group/aihc-bootcamp-spring2020/localize'
    gt_path = f'{group_dir}/annotations/val_encoded.json'

    with open(gt_path) as f:
        gt = json.load(f)
    
    tuning_results = pd.DataFrame(columns=['threshold','mIoU','task'])
    for task in sorted(LOCALIZATION_TASKS):
        print(f"Task: {task}")
        cutoff, miou = find_threshold(task,gt,cam_dir)
        df = pd.concat([pd.DataFrame([[cutoff[i], miou[i], task]], columns=['threshold','mIoU','task']) for i in range(len(cutoff))],ignore_index=True)
        tuning_results = tuning_results.append(df, ignore_index = True)
    
    tuning_results.to_csv(f'{result_dir}/threshold_tunning_{model}_{method}.csv', index = False)