""" Converts saliency heatmap to segmentation for each method. Stored as encoded COCO binary mask in a json file. 
    
    Usage: python3 pred_segmentation.py --phase valid --method gradcam --model_type single
"""

from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from pycocotools import mask
from pathlib import Path
import numpy as np
import pickle
import json
from eval_constants import *
from eval_helper import cam_to_segmentation, segmentation_to_mask
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pkl_to_mask(pkl_path, task, smoothing=False, force_negative=False):
    """
    # load cam pickle file, get saliency map and resize. 
    Convert to binary segmentation mask and output encoded mask

    Args:
        pkl_path(str): path to the pickle file
        task(str): pathology
        smoothing(bool): if we use smoothing on the heatmap (for IG only)
        force_negative(bool): if we use manually chosen thresholding
    """
    # load pickle file, get saliency map and resize
    info = pickle.load(open(pkl_path, 'rb'))
    saliency_map = info['map']
    img_dims = info['cxr_dims']
    map_resized = F.interpolate(saliency_map, size=(
        img_dims[1], img_dims[0]), mode='bilinear', align_corners=False)

    # convert to segmentation
    try:
        if force_negative:
            override_negative = info['prob'] < PROB_CUTOFF[task]
            segm_map = cam_to_segmentation(
                map_resized, smoothing=smoothing, override_negative=override_negative)
        else:
            segm_map = cam_to_segmentation(map_resized, smoothing)
    except:
        print(f'Error at {img_id}, index = {idx}')
        raise

    segm_map = np.array(segm_map, dtype="int")
    encoded_map = segmentation_to_mask(segm_map)
    return encoded_map


def map_to_results(map_dir, result_name, smoothing=False, force_negative=False):
    """
    Converts saliency maps to result json format. 

    Args:
        map_dir(str): path to the directory that stores CAMs pickle files
        result_name(str): name of the prediction file
        smoothing(bool): if we use smoothing
        force_negative(bool): if we use manual thresholding
    """

    print("Parsing saliency maps to evaluation format")
    all_paths = list(Path(map_dir).rglob("*_map.pkl"))

    results = {}

    for pkl_path in tqdm(all_paths):

        # break down path to image name and task
        path = str(pkl_path).split('/')
        task = path[-1].split('_')[-2]
        img_id = '_'.join(path[-1].split('_')[:-2])

        if task not in LOCALIZATION_TASKS:
            continue

        # get encoded segmentation mask
        encoded_map = pkl_to_mask(
            pkl_path, task, smoothing, force_negative=force_negative)

        # add image and segmentation to submission dictionary
        if img_id in results:
            if task in results[img_id]:
                print(f'Check for duplicates for {task} for {img_id}')
                break
            else:
                results[img_id][task] = encoded_map
        else:
            results[img_id] = {}
            results[img_id][task] = encoded_map

    # save to json
    with open(result_name, "w") as f:
        json.dump(results, f)
    print(f'Ready for evaluation at {result_name}')


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--phase', type=str, required=True,
                        help='valid or test')
    parser.add_argument('--method', type=str, required=True,
                        help='localization method: gradcam')
    parser.add_argument('--model_type', default='ensemble',
                        help='single or ensemble')
    parser.add_argument('--if_threshold', default=False,
                        help='if using thresholding')
    parser.add_argument('--save_dir', default='/deep/group/aihc-bootcamp-spring2020/localize/eval_results',
                        help='path to save segmentations')

    args = parser.parse_args()

    method = args.method
    model_type = args.model_type
    phase = args.phase
    if_threshold = args.if_threshold
    result_dir = args.save_dir

    # get directory that stores the saliency maps
    cam_dirs = valid_cam_dirs if phase == 'valid' else test_cam_dirs

    map_dir = cam_dirs[f'{method}_{model_type}']

    # create dir
    Path(f'{result_dir}/{method}').mkdir(parents=True, exist_ok=True)

    if if_threshold:
        result_name = f'{result_dir}/{method}/{phase}_{method}_{model_type}_segmentations_encoded_prob_threshold.json'
    else:
        result_name = f'{result_dir}/{method}/{phase}_{method}_{model_type}_segmentations_encoded.json'

    if_smoothing = 'ig' in method
    map_to_results(map_dir, result_name, smoothing=if_smoothing,
                   force_negative=if_threshold)
