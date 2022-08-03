from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from pycocotools import mask

from eval_constants import LOCALIZATION_TASKS

def main(args):
    """
    For each pathology, count the number of CXRs with at least one segmentation.
    """
    with open(args.seg_path) as f:
        seg_dict = json.load(f)

    cxr_ids = sorted(seg_dict.keys())
    segmentation_label = {}
    for task in sorted(LOCALIZATION_TASKS):
        print(task)
        has_seg = []
        for cxr_id in cxr_ids:
            seg_item = seg_dict[cxr_id][task]
            seg_mask = mask.decode(seg_item)
            if np.sum(seg_mask) == 0:
                has_segmentation = 0
            else:
                has_segmentation = 1
            has_seg.append(has_segmentation)
        segmentation_label[task] = has_seg

    df = pd.DataFrame.from_dict(segmentation_label)
    n_cxr_per_pathology = df.sum()
    print(n_cxr_per_pathology)
    n_cxr_per_pathology.to_csv(f'{args.save_dir}/n_segs.csv')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seg_path', type=str,
                        help='json file path where segmentations are saved \
                              (encoded)')
    parser.add_argument('--save_dir', default='.',
                        help='where to save results')
    args = parser.parse_args()
    main(args)
