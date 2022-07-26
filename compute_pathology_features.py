""" 
Script that computes pathological features.
We used the original radiologist annotations to extract the number of instances, and segmentation masks to calculate areas, elongation and rectangularity (rec_area_ratio).
You can also extract number of instances from segmentation masks. The small difference is sometimes radiologists draw two pathologies that are overlapping. From radiologist annotation file,
the number of instance is 2. But using segmentation (since the pathologies overlapped), the number of instance would be 1.

 """
import glob
import cv2
import json
import pickle
import numpy as np
import pandas as pd
from pycocotools import mask
from eval_constants import LOCALIZATION_TASKS


def get_geometric_features(segm):
    """
    Given a segmentation mask, returns geometric features

    Args:
        segm (np.array): the binary segmentation mask
    """
    # load segmentation
    rgb_img = cv2.cvtColor(255 * segm, cv2.COLOR_GRAY2RGB)

    # find contours
    contours, _ = cv2.findContours(segm.copy(), 1, 1)

    # get number of instances and the area
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


if __name__ == '__main__':
    output_path = '.'
    test_labels = pd.read_csv('../cheXlocalize_data_code/dataset/test_labels.csv')
    test_labels['img_id'] = test_labels.Path.map(
        lambda x: '_'.join(x.split('/')[1:]).replace('.jpg', '')).tolist()
    all_ids = test_labels['img_id'].tolist()

    # load ground truth annotations (we use it to extract number of instances)
    gt_ann_dir = f'../cheXlocalize_data_code/dataset/annotations/ground_truth/gt_annotations_test.json'
    with open(gt_ann_dir) as f:
        gt_ann = json.load(f)

    # load ground truth segmentations
    gt_segm_dir = f'../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json'
    with open(gt_segm_dir) as f:
        gt_segm = json.load(f)

    # extract features from all cxrs with at least one pathology
    all_instances = {}
    all_areas = {}
    all_elongations = {}
    all_rec_area_ratios = {}

    pos_ids = sorted(gt_segm.keys())
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
                gt_item = gt_segm[img_id][task]
                gt_mask = mask.decode(gt_item)
                if np.sum(gt_mask) > 0:
                    n_instance = len(gt_ann[img_id][task]) if task in gt_ann[img_id] else 0  # use radiologist annotation to get number of instances
                    n_instance_segm, area, elongation, rec_area_ratio = get_geometric_features(gt_mask)  # use segmentation to get other features

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

    instance_df.to_csv(f'./num_instances_test.csv', index=False)
    area_df.to_csv(f'./area_ratio_test.csv', index=False)
    elongation_df.to_csv('elogation_test.csv', index=False)
    rec_area_ratio_df.to_csv('rec_area_ratio_test.csv', index=False)
