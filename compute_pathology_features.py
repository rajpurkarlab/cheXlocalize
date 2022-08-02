"""
Compute four pathological features: (1) number of instances (for example, bilateral Pleural Effusion would have two instances, whereas there is only one instance for Cardiomegaly), (2) size (pathology area with respect to the area of the whole CXR), (3) elongation and (4) irrectangularity (the last two features measure the complexity of the pathology shape and were calculated by fitting a rectangle of minimum area enclosing the binary mask).

Note that we use the ground-truth annotations to extract the number of instances, and we use the ground-truth segmentation masks to calculate area, elongation and rectangularity. We chose to extract number of instances from annotations because sometimes radiologists draw two instances for a pathology that are overlapping; in this case, the number of annotations would be 2, but the number of segmentations would be 1.
"""
from argparse import ArgumentParser
import cv2
import glob
import json
import numpy as np
import pandas as pd
import pickle
from pycocotools import mask

from eval_constants import LOCALIZATION_TASKS


def get_geometric_features(segm):
    """
    Given a segmentation mask, return geometric features.

    Args:
        segm (np.array): the binary segmentation mask
    """
    # load segmentation
    rgb_img = cv2.cvtColor(255 * segm, cv2.COLOR_GRAY2RGB)

    # find contours
    contours, _ = cv2.findContours(segm.copy(), 1, 1)

    # get number of instances and area
    n_instance = len(contours)
    area_ratio = np.sum(segm) / (segm.shape[0] * segm.shape[1])

    # use the longest coutour to calculate geometric features
    max_idx = np.argmax([len(contour) for contour in contours])
    cnt = contours[max_idx]

    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), a = rect

    instance_area = cv2.contourArea(cnt)
    elongation = max(w, h) / min(w, h)
    rec_area_ratio = instance_area / (w * h)

    return n_instance, area_ratio, elongation, rec_area_ratio


def main(args):
    # load ground-truth annotations (needed to extract number of instances)
    # and ground-truth segmentations
    with open(args.gt_ann) as f:
        gt_ann = json.load(f)
    with open(args.gt_seg) as f:
        gt_seg = json.load(f)

    # extract features from all cxrs with at least one pathology
    all_instances = {}
    all_areas = {}
    all_elongations = {}
    all_rec_area_ratios = {}
    all_ids = sorted(gt_ann.keys())
    pos_ids = sorted(gt_seg.keys())
    for task in sorted(LOCALIZATION_TASKS):
        print(task)
        n_instances = []
        areas = []
        elongations = []
        rec_area_ratios = []
        for img_id in all_ids:
            n_instance = 0
            area = 0
            elongation = np.nan
            rec_area_ratio = np.nan
            # calculate features for cxr with a pathology segmentation
            if img_id in pos_ids:
                gt_item = gt_seg[img_id][task]
                gt_mask = mask.decode(gt_item)
                if np.sum(gt_mask) > 0:
                    # use annotation to get number of instances
                    n_instance = len(gt_ann[img_id][task]) \
                            if task in gt_ann[img_id] else 0
                    # use segmentation to get other features
                    n_instance_segm, area, elongation, rec_area_ratio = \
                            get_geometric_features(gt_mask)
            n_instances.append(n_instance)
            areas.append(area)
            elongations.append(elongation)
            rec_area_ratios.append(rec_area_ratio)
        all_instances[task] = n_instances
        all_areas[task] = areas
        all_elongations[task] = elongations
        all_rec_area_ratios[task] = rec_area_ratios

    instance_df = pd.DataFrame(all_instances)
    area_df = pd.DataFrame(all_areas)
    elongation_df = pd.DataFrame(all_elongations)
    rec_area_ratio_df = pd.DataFrame(all_rec_area_ratios)

    instance_df['img_id'] = all_ids
    area_df['img_id'] = all_ids
    elongation_df['img_id'] = all_ids
    rec_area_ratio_df['img_id'] = all_ids

    instance_df.to_csv(f'{args.save_dir}/num_instances.csv', index=False)
    area_df.to_csv(f'{args.save_dir}/area_ratio.csv', index=False)
    elongation_df.to_csv(f'{args.save_dir}/elongation.csv', index=False)
    rec_area_ratio_df.to_csv(f'{args.save_dir}/rec_area_ratio.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_ann', type=str,
                        help='path to json file with raw ground-truth annotations')
    parser.add_argument('--gt_seg', type=str,
                        help='path to json file with ground-truth segmentations \
                              (encoded)')
    parser.add_argument('--save_dir', default='.',
                        help='where to save feature dataframes')
    args = parser.parse_args()
    main(args)
