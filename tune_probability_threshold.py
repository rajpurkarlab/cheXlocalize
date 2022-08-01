"""
When evaluating mIoU on the full dataset, we ensure that the final binary segmentation is consistent with model probability output by applying another layer of thresholding such that the segmentation mask is all zeros if the predicted probability is below a chosen level.

The probability threshold is searched on the interval of [0,0.9] with steps of 0.1. The exact value is determined per pathology by maximizing the mIoU on the validation set.
"""
from argparse import ArgumentParser
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from pycocotools import mask
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval import calculate_iou
from eval_constants import LOCALIZATION_TASKS
from heatmap_to_segmentation import cam_to_segmentation


def compute_miou(cutoff, pkl_paths,gt):
    """Caculate mIoU given a threshold and a list of pkl paths."""
    ious = []
    for pkl_path in tqdm(pkl_paths):
        # get saliency segmentation
        info = pickle.load(open(pkl_path, 'rb'))
        img_dims = info['cxr_dims']
        map_resized = F.interpolate(info['map'], size=(img_dims[1],img_dims[0]),
                                    mode='bilinear', align_corners=False)
        if torch.is_tensor(info['prob']) and info['prob'].size()[0] == 14:
            prob_idx = CHEXPERT_TASKS.index(info['task'])
            pred_prob = info['prob'][prob_idx]
        else:
            pred_prob = info['prob']

        if pred_prob > cutoff:
            segm = cam_to_segmentation(map_resized)
            pred_mask = np.array(segm)
        else:
            pred_mask = np.zeros((img_dims[1],img_dims[0]))

        # get gt segmentation
        path = str(pkl_path).split('/')
        task = path[-1].split('_')[-2]
        img_id = '_'.join(path[-1].split('_')[:-2])
        if img_id in gt:
            gt_item = gt[img_id][task]
            gt_mask = mask.decode(gt_item)
        else:
            gt_mask = np.zeros((img_dims[1],img_dims[0]))

        iou = calculate_iou(pred_mask, gt_mask, true_pos_only=False)
        ious.append(iou)

    miou = round(np.nanmean(np.array(ious)), 3)
    return miou


def find_threshold(task, gt_dict, cam_dir):
    """
    For a given task, find the probability threshold with max mIoU on val set.
    """
    cam_pkl = sorted(list(Path(cam_dir).rglob(f"*{task}_map.pkl")))
    cutoffs = np.arange(0,.9,.1)
    mious = [compute_miou(cutoff, cam_pkl, gt_dict) for cutoff in cutoffs]
    cutoff = cutoffs[mious.index(max(mious))]
    print(f"cutoff: {cutoffs}; iou: {mious}")
    return cutoffs, mious


def main(args):
    with open(args.gt_path) as f:
        gt_dict = json.load(f)

    tuning_results = pd.DataFrame(columns=['prob_threshold','mIoU','task'])
    for task in sorted(LOCALIZATION_TASKS):
        print(f"Task: {task}")
        cutoff, miou = find_threshold(task, gt_dict, args.map_dir)
        df = pd.concat([pd.DataFrame([[round(cutoff[i], 1),
                                       round(miou[i], 3),
                                       task]],
                                     columns=['prob_threshold','mIoU','task']) \
                                        for i in range(len(cutoff))],
                       ignore_index=True)
        tuning_results = tuning_results.append(df, ignore_index=True)

    tuning_results.to_csv(f'{args.save_dir}/probability_tuning_results.csv',
                          index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--map_dir', type=str,
                        help='directory with pickle files containing heatmaps \
                              and model output')
    parser.add_argument('--gt_path', type=str,
                        help='json file where ground-truth segmentations are \
                              saved (encoded)')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='where to save the probability threshold tuned on the \
                              validation set')
    args = parser.parse_args()
    main(args)
