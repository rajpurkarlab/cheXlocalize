import json
import numpy as np
import pandas as pd
import tqdm
from eval import *
from eval_constants import LOCALIZATION_TASKS

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gt_path', type=str,
                        help='directory where ground-truth segmentations are \
                              saved (encoded)')
    parser.add_argument('--pred_path', type=str,
                        help='json path where the human benchmark salient points are saved')
    parser.add_argument('--save_dir', default='.',
                        help='where to save evaluation results')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to fix')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # read human benchmark salient point
    with open(args.pred_path) as f:  
        hb_salient_pts = json.load(f) 

    # read ground truth segmentation
    with open(args.gt_path) as f:
        gt_dict = json.load(f)

    # evaluate hit
    hit_result = {}
    for task in sorted(LOCALIZATION_TASKS):
        print(f'Evaluating {task}')
        hit_result[task] = []
        for img_id in sorted(gt_dict.keys()):
            hit = np.nan
            gt_item = gt_dict[img_id][task]
            gt_mask = mask.decode(gt_item)

            if np.sum(gt_mask) !=0:
                if img_id in hb_salient_pts and task in hb_salient_pts[img_id]:
                    salient_pts = hb_salient_pts[img_id][task]
                    hit = 0
                    for pt in salient_pts:
                        if gt_mask[int(pt[1]), int(pt[0])]:
                            hit = 1
                else:
                    hit = 0

            hit_result[task].append(hit)


    pt_hb_df = pd.DataFrame.from_dict(hit_result)
    bs_df = bootstrap_metric(pt_hb_df, 1000)
    bs_df.to_csv(f'{args.save_dir}/hb_hit_bootstrap_results.csv',index = False)
    pt_hb_df['cxr_id'] = sorted(gt_dict.keys())
    pt_hb_df.to_csv(f'{args.save_dir}/hb_hit_results.csv',index = False)

    records = []
    for task in bs_df.columns:
        records.append(create_ci_record(bs_df[task], task))

    summary_df = pd.DataFrame.from_records(records)
    summary_df.to_csv(f'{args.save_dir}/hb_hit_summary_results.csv',index = False)