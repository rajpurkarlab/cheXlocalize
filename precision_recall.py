from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pycocotools import mask
from tqdm import tqdm

from eval_constants import LOCALIZATION_TASKS
from heatmap_to_segmentation import pkl_to_mask


def get_results(gt_dict, hb_path):
    """
    TODO: add comment
    """
    with open(hb_path) as f:
        hb_dict = json.load(f)

    results = {}
    all_ids = sorted(gt_dict.keys())
    tasks = sorted(LOCALIZATION_TASKS)

    for task in tqdm(tasks):
        for img_id in all_ids:
            gt_item = gt_dict[img_id][task]
            gt_mask = mask.decode(gt_item)

            if img_id not in hb_dict:
                hb_mask = np.zeros(gt_mask.shape)
            else:
                hb_item = hb_dict[img_id][task]
                hb_mask = mask.decode(hb_item)

            TP = np.sum(gt_mask == hb_mask)
            FP = np.sum(np.logical_and(hb_mask == 1, gt_mask == 0))
            FN = np.sum(np.logical_and(hb_mask == 0, gt_mask == 1))
            # append to big numpy array 
            if task in results:
                results[task]['tp'] += TP
                results[task]['fp'] += FP
                results[task]['fn'] += FN
            else:
                results[task] = {}
                results[task]['tp'] = TP 
                results[task]['fp'] = FP
                results[task]['fn'] = FN
    return results


def calculate_precision_recall(dict_item):
    TP = dict_item['tp']
    FP = dict_item['fp']
    FN = dict_item['fn']
    p = TP/(TP+FP)
    r = TP/(TP+FN)
    return p, r


def main(args):
    with open(args.gt_path) as f:
        gt_dict = json.load(f)

    for source in ['pred', 'hb']:
        seg_path = args.pred_seg_path if source == 'pred' else args.hb_seg_path
        results = get_results(gt_dict, seg_path)
        precisions = []
        recalls = []
        for t in sorted(LOCALIZATION_TASKS):
            p, r = calculate_precision_recall(results[t])
            precisions.append(p)
            recalls.append(r)

        df = pd.DataFrame(columns = ['pathology', 'precision', 'recall'])
        df['pathology'] = sorted(LOCALIZATION_TASKS)
        df['precision'] = precisions
        df['recall'] = recalls
        df.to_csv(f'{args.save_dir}/{source}_precision_recall.csv')


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
