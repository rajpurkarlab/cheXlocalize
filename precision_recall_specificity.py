from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pycocotools import mask
from tqdm import tqdm

from eval_constants import LOCALIZATION_TASKS
from heatmap_to_segmentation import pkl_to_mask


def get_results(gt_dict, seg_path):
    """
    For each pathology, count the total number of pixels that are TP, TN, FP
    and FN. Only include CXRs that have ground-truth segmentations.
    """
    with open(seg_path) as f:
        seg_dict = json.load(f)

    results = {}
    all_ids = sorted(gt_dict.keys())
    tasks = sorted(LOCALIZATION_TASKS)

    for task in tqdm(tasks):
        for img_id in all_ids:
            gt_item = gt_dict[img_id][task]
            gt_mask = mask.decode(gt_item)

            if img_id not in seg_dict:
                seg_mask = np.zeros(gt_mask.shape)
            else:
                seg_item = seg_dict[img_id][task]
                seg_mask = mask.decode(seg_item)

            TP = np.sum(np.logical_and(seg_mask == 1, gt_mask == 1))
            TN = np.sum(np.logical_and(seg_mask == 0, gt_mask == 0))
            FP = np.sum(np.logical_and(seg_mask == 1, gt_mask == 0))
            FN = np.sum(np.logical_and(seg_mask == 0, gt_mask == 1))

            if task in results:
                results[task]['tp'] += TP
                results[task]['tn'] += TN
                results[task]['fp'] += FP
                results[task]['fn'] += FN
            else:
                results[task] = {}
                results[task]['tp'] = TP
                results[task]['tn'] = TN
                results[task]['fp'] = FP
                results[task]['fn'] = FN
    return results


def calculate_precision_recall_specificity(dict_item):
    TP = dict_item['tp']
    TN = dict_item['tn']
    FP = dict_item['fp']
    FN = dict_item['fn']
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    return precision, recall, specificity


def main(args):
    with open(args.gt_path) as f:
        gt_dict = json.load(f)

    for source in ['pred', 'hb']:
        seg_path = args.pred_seg_path if source == 'pred' else args.hb_seg_path
        results = get_results(gt_dict, seg_path)
        precisions = []
        recalls = []
        specificities = []
        for t in sorted(LOCALIZATION_TASKS):
            p, r, s = calculate_precision_recall_specificity(results[t])
            precisions.append(p)
            recalls.append(r)
            specificities.append(s)

        df = pd.DataFrame()
        df['pathology'] = sorted(LOCALIZATION_TASKS)
        df['precision'] = precisions
        df['recall/sensitivity'] = recalls
        df['specificity'] = specificities
        df.to_csv(f'{args.save_dir}/{source}_precision_recall_specificity.csv')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_path', type=str,
                        help='json file path where ground-truth segmentations \
                              are saved (encoded)')
    parser.add_argument('--pred_seg_path', type=str,
                        help='json file path where saliency method segmentations \
                              are saved (encoded)')
    parser.add_argument('--hb_seg_path', type=str,
                        help='json file path where human benchmark segmentations \
                              are saved (encoded)')
    parser.add_argument('--save_dir', default='.',
                        help='where to save precision/recall results')
    args = parser.parse_args()

    main(args)
