"""
Find thresholds (used to binarize the heatmaps) that maximize mIoU on the
validation set. Pass in a list of potential thresholds [0.2, 0.3, ... , 0.8].
Save the best threshold for each pathology in a csv file.
"""
from argparse import ArgumentParser
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from pycocotools import mask
import torch.nn.functional as F
from tqdm import tqdm

from eval import calculate_iou
from eval_constants import LOCALIZATION_TASKS
from heatmap_to_segmentation import pkl_to_mask


def compute_miou(threshold, cam_pkls, gt):
    """
    Given a threshold and a list of heatmap pickle files, return the mIoU.

    Args:
        threshold (double): the threshold used to convert heatmaps to segmentations
        cam_pkls (list): a list of heatmap pickle files (for a given pathology)
        gt (dict): dictionary of ground truth segmentation masks
    """
    ious = []
    for pkl_path in tqdm(cam_pkls):
        # break down path to image name and task
        path = str(pkl_path).split('/')
        task = path[-1].split('_')[-2]
        img_id = '_'.join(path[-1].split('_')[:-2])

        # add image and segmentation to submission dictionary
        if img_id in gt:
            pred_mask = pkl_to_mask(pkl_path=pkl_path, threshold=threshold)
            gt_item = gt[img_id][task]
            gt_mask = mask.decode(gt_item)
            assert (pred_mask.shape == gt_mask.shape)
            iou_score = calculate_iou(pred_mask, gt_mask, true_pos_only=True)
        else:
            iou_score = np.nan
        ious.append(iou_score)

    miou = np.nanmean(np.array(ious))
    return miou


def tune_threshold(task, gt, cam_dir):
    """
    For a given pathology, find the threshold that maximizes mIoU.

    Args:
        task (str): localization task
        gt (dict): dictionary of the ground truth segmentation masks
        cam_dir (str): directory with pickle files containing heat maps
    """
    cam_pkls = sorted(list(Path(cam_dir).rglob(f"*{task}_map.pkl")))
    thresholds = np.arange(0.2, .8, .1)
    mious = [compute_miou(threshold, cam_pkls, gt) for threshold in thresholds]
    best_threshold = thresholds[mious.index(max(mious))]
    return best_threshold


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--map_dir', type=str,
                        help='directory with pickle files containing heat maps')
    parser.add_argument('--gt_path', type=str,
                        help='json file where ground-truth segmentations are \
                              saved (encoded)')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='where to save the best thresholds tuned on the \
                              validation set')
    args = parser.parse_args()

    with open(args.gt_path) as f:
        gt = json.load(f)

    # tune thresholds and save the best threshold for each pathology to a csv file
    tuning_results = pd.DataFrame(columns=['threshold', 'task'])
    for task in sorted(LOCALIZATION_TASKS):
        print(f"Task: {task}")
        threshold = tune_threshold(task, gt, args.map_dir)
        df = pd.DataFrame([[round(threshold, 1), task]],
                          columns=['threshold', 'task'])
        tuning_results = pd.concat([tuning_results, df], axis=0)

    tuning_results.to_csv(f'{args.save_dir}/tuning_results.csv', index=False)
