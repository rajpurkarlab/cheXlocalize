import json
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask
from eval_constants import *
from argparse import ArgumentParser

def main(args):
    """
    Count the number of cxrs with ground-truth segmentation per pathology.
    """
    with open(args.gt_path) as f:
        gt_dict = json.load(f)
            
    cxr_ids = sorted(gt_dict.keys())
    gt_segmentation_label = {}
    for task in sorted(LOCALIZATION_TASKS):
        print(task)
        has_seg = []
        for cxr_id in cxr_ids:
            gt_item = gt_dict[cxr_id][task]
            gt_mask = mask.decode(gt_item)
            if np.sum(gt_mask) == 0:
                has_segmentation = 0
            else:
                has_segmentation = 1
            has_seg.append(has_segmentation)
        gt_segmentation_label[task] = has_seg  
    df = pd.DataFrame.from_dict(gt_segmentation_label)
    n_cxr_per_pathology = df.sum()
    print(n_cxr_per_pathology)
    n_cxr_per_pathology.to_csv(f'{args.save_dir}/n_cxr.csv')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gt_path', type=str,
                        help='directory where ground-truth segmentations are \
                              saved (encoded)')
    parser.add_argument('--save_dir', default='.',
                        help='where to save evaluation results')
    args = parser.parse_args()
    main(args)
    