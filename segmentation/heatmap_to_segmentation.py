""" Converts saliency heatmaps to binary segmentations and encode segmentations RLE formats using the pycocotools Mask API. The final output is stored in a json file. 
    Input: where saliency maps are stored (saliency heatmaps are extracted from the pickle files)
            DEFAULT: ../cheXlozalize_dataset/GradCAM_maps_val_sample/
    Output: the json file that stores the encoded segmentation masks
            DEFAULT: saliency_segmentations_val.json
            
    Usage: python3 heatmap_to_segmentation.py --saliency_path ../cheXlozalize_dataset/GradCAM_maps_val_sample/ --output_file_name saliency_segmentations_val.json
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import json
import pickle
import numpy as np
from pathlib import Path
import cv2
import torch.nn.functional as F
from argparse import ArgumentParser
from eval_helper import encode_segmentation
from eval_constants import *


def cam_to_segmentation(cam_mask):
    """
    Convert a saliency heatmap to binary segmentation mask

    Args:
        cam_mask(torch.Tensor): heatmap of the original image size. dim: H x W. Will squeeze the tensor if there are more than two dimensions

    Returns:
        segmentation_output(np.array): binary segmentation output
    """
    if (len(cam_mask.size()) > 2):
        cam_mask = cam_mask.squeeze()

    assert len(cam_mask.size()) == 2

    # normalize heatmap
    mask = cam_mask - cam_mask.min()
    mask = 255 * mask.div(mask.max()).data
    mask = mask.cpu().detach().numpy()
    mask = np.uint8(mask)

    # Use Otsu's method to find threshold
    maxval = np.max(mask)
    segmentation = cv2.threshold(mask, 0, maxval, cv2.THRESH_OTSU)[1]

    return segmentation


def pkl_to_mask(pkl_path, task):
    """
    # Load a pickle file, get saliency map and resize to original image dimension. 
    Convert to binary segmentation and output the encoded mask

    Args:
        pkl_path(str): path to the model output pickle file
        task(str): localization task
    """
    # load pickle file, get saliency map and resize
    info = pickle.load(open(pkl_path, 'rb'))
    saliency_map = info['map']
    img_dims = info['cxr_dims']
    map_resized = F.interpolate(saliency_map, size=(
        img_dims[1], img_dims[0]), mode='bilinear', align_corners=False)

    # convert to segmentation
    segmentation = cam_to_segmentation(map_resized)

    # encode segmentation to
    encoded_mask = encode_segmentation(segmentation)
    return encoded_mask


def heatmap_to_mask(map_dir, output_file_name):
    """
    Converts all saliency maps to segmentations and store segmentations in a json file. 

    Args:
        map_dir(str): where the model output pickle files are saved
        output_file_name(str): json file where the encoded segmentation masks are stored
    """

    print("Parsing saliency maps")
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
        encoded_mask = pkl_to_mask(pkl_path, task)

        # add image and segmentation to submission dictionary
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
    with open(output_file_name, "w") as f:
        json.dump(results, f)
    print(f'Segmentation masks (in RLE format) saved at {output_file_name}')


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--saliency_path', type=str, default='../cheXlozalize_dataset/GradCAM_maps_val_sample/',
                        help='where saliency maps are stored')
    parser.add_argument('--output_file_name', default='saliency_segmentations_val.json',
                        help='where to save segmentations')

    args = parser.parse_args()
    map_dir = args.saliency_path
    output = args.output_file_name
    heatmap_to_mask(map_dir, output)
