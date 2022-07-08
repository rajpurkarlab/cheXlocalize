"""
Converts saliency heat maps to binary segmentations and encodes segmentations
using RLE formats using the pycocotools Mask API. The final output is stored in
a json file.

The default thresholding used in this code is Otsu's method (an automatic global thresholding algorithm provided by cv2). 
Users can also pass in their self-defined thresholds to binarize the heatmaps through --threshold_path. 
Make sure the input is a csv file with the same format as the tuning_results.csv file we provided.
"""
from argparse import ArgumentParser
import cv2
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
import sys
import torch.nn.functional as F
from tqdm import tqdm

from eval_constants import LOCALIZATION_TASKS
from utils import encode_segmentation


def cam_to_segmentation(cam_mask, threshold=np.nan):
    """
    Threshold a saliency heat map to binary segmentation mask.

    Args:
        cam_mask (torch.Tensor): heat map in the original image size (H x W).
            Will squeeze the tensor if there are more than two dimensions.

    Returns:
        segmentation_output (np.array): binary segmentation output
    """
    if (len(cam_mask.size()) > 2):
        cam_mask = cam_mask.squeeze()

    assert len(cam_mask.size()) == 2

    # normalize heatmap
    mask = cam_mask - cam_mask.min()
    mask = mask.div(mask.max()).data
    mask = mask.cpu().detach().numpy()

    # use otsu's method if no threshold is passed in
    if np.isnan(threshold):
        mask = np.uint8(255 * mask)

        # Use Otsu's method to find threshold
        maxval = np.max(mask)
        segmentation = cv2.threshold(mask, 0, maxval, cv2.THRESH_OTSU)[1]

    else:
        segmentation = np.array(mask > threshold, dtype="int")

    return segmentation


def pkl_to_mask(pkl_path, threshold=np.nan):
    """
    Load pickle file, get saliency map and resize to original image dimension.
    Threshold the heatmap to binary segmentation.

    Args:
        pkl_path (str): path to the model output pickle file
        task (str): localization task
    """
    # load pickle file, get saliency map and resize
    info = pickle.load(open(pkl_path, 'rb'))
    saliency_map = info['map']
    img_dims = info['cxr_dims']
    map_resized = F.interpolate(saliency_map,
                                size=(img_dims[1], img_dims[0]),
                                mode='bilinear',
                                align_corners=False)

    # convert to segmentation
    segmentation = cam_to_segmentation(map_resized, threshold=threshold)

    return segmentation


def heatmap_to_mask(map_dir, output_path, threshold_path=''):
    """
    Converts all saliency maps to segmentations and stores segmentations in a
    json file.
    """
    print('Parsing saliency maps')
    all_paths = list(Path(map_dir).rglob('*_map.pkl'))

    results = {}
    for pkl_path in tqdm(all_paths):
        # break down path to image name and task
        path = str(pkl_path).split('/')
        task = path[-1].split('_')[-2]
        img_id = '_'.join(path[-1].split('_')[:-2])

        if task not in LOCALIZATION_TASKS:
            continue

        # get encoded segmentation mask
        if threshold_path:
            tuning_results = pd.read_csv(threshold_path)
            best_threshold = tuning_results[tuning_results['task'] ==
                                            'Edema']['threshold'].values[0]
        else:
            best_threshold = np.nan

        segmentation = pkl_to_mask(pkl_path, threshold=best_threshold)
        encoded_mask = encode_segmentation(segmentation)

        # add image and segmentation to results dict
        if img_id in results:
            if task in results[img_id]:
                print(f'Check for duplicates for {task} for {img_id}')
                break
            else:
                results[img_id][task] = encoded_mask
        else:
            results[img_id] = {}
            results[img_id][task] = encoded_mask

    # save to json
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f'Segmentation masks (in RLE format) saved to {output_path}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--map_dir',
        type=str,
        help='directory with pickle files containing heat maps')
    parser.add_argument(
        '--threshold_path',
        type=str,
        default='',
        help=
        'csv file that stores the threshold tuned on the validation set. Use Otsu'
        's method if no path is given.')
    parser.add_argument('--output_path',
                        type=str,
                        default='./saliency_segmentations.json',
                        help='json file path for saving encoded segmentations')

    args = parser.parse_args()

    heatmap_to_mask(args.map_dir, args.output_path, args.threshold_path)
