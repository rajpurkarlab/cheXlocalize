"""
Converts saliency heatmaps to binary segmentations and encodes segmentations
using RLE formats using the pycocotools Mask API. The final output is stored in
a json file.

The default thresholding used in this code is Otsu's method (an automatic global
thresholding algorithm provided by cv2). Users can also pass in probability thresholds through --probability_threshold_path. If a path is given, then no segmentation is generated if a cxr's predicted probability of a pathlogy is below the given cutoff
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
import torch
import io
from PIL import Image, ImageDraw
from eval_constants import LOCALIZATION_TASKS
from utils import encode_segmentation


def cam_to_segmentation(cam_mask, threshold=np.nan, smoothing=False, k=0):
    """
    Threshold a saliency heatmap to binary segmentation mask.
    Args:
        cam_mask (torch.Tensor): heat map in the original image size (H x W).
            Will squeeze the tensor if there are more than two dimensions.
        threshold (np.float64): threshold to use

    Returns:
        segmentation (np.ndarray): binary segmentation output
    """
    if (len(cam_mask.size()) > 2):
        cam_mask = cam_mask.squeeze()

    assert len(cam_mask.size()) == 2

    # normalize heatmap
    mask = cam_mask - cam_mask.min()
    mask = mask.div(mask.max()).data
    mask = mask.cpu().detach().numpy()

    # use Otsu's method to find threshold if no threshold is passed in
    if np.isnan(threshold):
        mask = np.uint8(255 * mask)

        if smoothing:
            heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            gray_img = cv2.boxFilter(cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY),
                                     -1, (k, k))
            mask = 255 - gray_img

        maxval = np.max(mask)
        thresh = cv2.threshold(mask, 0, maxval, cv2.THRESH_OTSU)[1]

        # draw out contours
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        polygons = []
        for cnt in cnts:
            if len(cnt) > 1:
                polygons.append([list(pt[0]) for pt in cnt])

        # create segmentation based on contour
        img_dims = (mask.shape[1], mask.shape[0])
        segmentation_output = Image.new('1', img_dims)

        for polygon in polygons:
            coords = [(point[0], point[1]) for point in polygon]
            ImageDraw.Draw(segmentation_output).polygon(coords,
                                                        outline=1,
                                                        fill=1)

        segmentation = np.array(segmentation_output, dtype="int")

    else:
        segmentation = np.array(mask > threshold, dtype="int")

    return segmentation


def pkl_to_mask(pkl_path, threshold=np.nan, prob_cutoff=0, smoothing=False, k=0):
    """
    Load pickle file, get saliency map and resize to original image dimension.
    Threshold the heatmap to binary segmentation.

    Args:
        pkl_path (str): path to the model output pickle file
        threshold (np.float64): threshold to use

    Returns:
        segmentation (np.ndarray): binary segmentation output
    """

    # load pickle file, get saliency map and resize
    class CPU_Unpickler(pickle.Unpickler):

        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)

    info = CPU_Unpickler(open(pkl_path, 'rb')).load()
    saliency_map = info['map']
    img_dims = info['cxr_dims']
    map_resized = F.interpolate(saliency_map,
                                size=(img_dims[1], img_dims[0]),
                                mode='bilinear',
                                align_corners=False)

    pred_prob = info['prob']
    if pred_prob < prob_cutoff:
        segmentation = np.zeros((img_dims[1], img_dims[0]))
    else:
        # convert to segmentation
        segmentation = cam_to_segmentation(map_resized,
                                        threshold=threshold,
                                        smoothing=smoothing,
                                        k=k)

    return segmentation


def heatmap_to_mask(map_dir,
                    output_path,
                    probability_threshold_path,
                    smoothing=False,
                    k=0):
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
        if probability_threshold_path:
            prob_results = pd.read_csv(probability_threshold_path)
            max_miou = prob_results.loc[prob_results.groupby(['task'])['mIoU'].agg('idxmax')]
            prob_cutoff = max_miou[max_miou['task'] == task]['prob_threshold'].values[0]
        else:
            prob_cutoff = 0

        segmentation = pkl_to_mask(pkl_path,
                                   prob_cutoff=prob_cutoff,
                                   smoothing=smoothing,
                                   k=k)
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
    Path(os.path.dirname(output_path)).mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f'Segmentation masks (in RLE format) saved to {output_path}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--map_dir',
                        type=str,
                        help='directory with pickle files containing heatmaps')

    parser.add_argument(
        '--probability_threshold_path',
        type=str,
        help="csv file that stores pre-defined probability cutoffs. \
                    If a path is given, then no segmentation is generated if the predicted probability is below the cutoff")
    parser.add_argument('--output_path',
                        type=str,
                        default='./saliency_segmentations.json',
                        help='json file path for saving encoded segmentations')
    parser.add_argument(
        '--if_smoothing',
        type=bool,
        help="True if apply smoothing to heatmap, False if not")
    parser.add_argument(
        '--k',
        type=int,
        help="Size of the kernel used for box filter smoothing")

    args = parser.parse_args()

    heatmap_to_mask(args.map_dir, args.output_path, args.probability_threshold_path,
                    args.if_smoothing, args.k)
