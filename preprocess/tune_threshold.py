import pickle
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from plot_helper import *
import pandas as pd
from eval_helper import *
from pycocotools import mask
from eval_constants import LOCALIZATION_TASKS
from tqdm import tqdm


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
        if info['prob'] > cutoff:
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

        iou = iou_seg(gradcam_mask,pred_mask)
        ious.append(iou)


    
    np_ious = np.array(ious)
    miou = np.mean(np_ious[np_ious>-1])
    
    return miou




def find_threshold(task,gt):
    """
    For a given task, find the probability threshold with max miou (on validation)
    """
    gradcam_ensemble_pkl = '/deep/group/aihc-bootcamp-spring2020/localize/uncertainty_handling/valid_predictions/ensemble_results/cams/'
    gradcam_pkl = sorted(list(Path(gradcam_ensemble_pkl).rglob(f"*{task}_map.pkl")))
    
    cutoffs = np.arange(0.1,.9,.1)
    mious = [compute_miou(cutoff,gradcam_pkl,gt) for cutoff in cutoffs ]
    cutoff = cutoffs[mious.index(max(mious))]
    
    print(f"cutoff: {cutoffs}; iou: {mious}")
    return cutoff, max(mious)



if __name__ == '__main__':
    
    # read gt masks
    print("Read gt masks")
    phase = 'valid'
    group_dir = '/deep/group/aihc-bootcamp-spring2020/localize'
    gt_path = f'{group_dir}/annotations/{phase}_encoded.json'

    with open(gt_path) as f:
        gt = json.load(f)
        
    tuning_results = pd.DataFrame(columns=['prob_threshold','mIoU','task'])
    for task in sorted(LOCALIZATION_TASKS):
        print(f"Task: {task}")
        cutoff, miou = find_threshold(task,gt)
        tuning_results = tuning_results.append({'prob_threshold':cutoff, 'mIoU':miou, 'task':task}, ignore_index = True)
    
    tuning_results.to_csv(f'{group_dir}/eval_results/threshold_tunning.csv')