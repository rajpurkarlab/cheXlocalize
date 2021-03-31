""" Given model segmentation and radiologist ground truth, comapre the mIoU per task.
    
    Usage: python3 eval.py --phase valid --pred_path /path/to/segmentation json --save_dir /path/to/save/results
"""

from pycocotools import mask
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
from argparse import ArgumentParser
from eval_helper import create_mask, iou_seg
from eval_constants import LOCALIZATION_TASKS


def compute_metrics(gt_dir, pred_dir):
    """
    Take in ground truth and prediction json (both encoded) and return iou for each image under each pathology
    """
    with open(gt_dir) as f:
        gt = json.load(f)

    with open(pred_dir) as f:
        pred = json.load(f)

    ious = {}

    all_ids = sorted(pred.keys())
    tasks = sorted(LOCALIZATION_TASKS)

    for task in tasks:
        print(f'Evaluating {task}')
        ious[task] = []

        for img_id in tqdm(all_ids):

            # get predicted segmentation mask
            pred_item = pred[img_id][task]
            pred_mask = mask.decode(pred_item)

            # get ground_truth segmentation mask
            # if image not in gt, create zero mask
            if img_id not in gt:
                gt_mask = np.zeros(pred_item['size'])
            else:
                gt_item = gt[img_id][task]
                gt_mask = mask.decode(gt_item)

            assert gt_mask.shape == pred_mask.shape

            # compute metric
            iou_score = iou_seg(pred_mask, gt_mask)
            ious[task].append(iou_score)

        assert len(ious[task]) == len(pred.keys())

    return ious


def bootstrap_metric(df, num_replicates, metric='iou'):
    """
    Create dataframe of bootstrap samples 
    """
    def single_replicate_performances():
        sample_ids = np.random.choice(len(df), size=len(df), replace=True)
        replicate_performances = {}
        df_replicate = df.iloc[sample_ids]

        for task in df.columns:
            if metric == 'iou':
                performance = df_replicate[df_replicate > -1][task].mean()
            else:
                performance = df_replicate[task].mean()
            replicate_performances[task] = performance
        return replicate_performances

    all_performances = []

    for _ in range(num_replicates):
        replicate_performances = single_replicate_performances()
        all_performances.append(replicate_performances)

    df_performances = pd.DataFrame.from_records(all_performances)
    return df_performances


def compute_cis(series, confidence_level):
    """
    Compute confidence intervals given cf level
    """
    sorted_perfs = series.sort_values()
    lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
    upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
    lower = sorted_perfs.iloc[lower_index].round(3)
    upper = sorted_perfs.iloc[upper_index].round(3)
    mean = round(sorted_perfs.mean(), 3)
    return lower, mean, upper


def create_ci_record(perfs, name):
    lower, mean, upper = compute_cis(perfs, confidence_level=0.05)
    record = {"name": name,
              "lower": lower,
              "mean": mean,
              "upper": upper,
              }
    return record


def evaluate(gt_dir, pred_dir, save_dir, table_name):
    """
    Pipeline to evaluate localizations. Return miou by pathologies and their confidence intervals
    """

    # create save dir if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    metrics = compute_metrics(gt_dir, pred_dir)
    metrics_df = pd.DataFrame.from_dict(metrics)

    bs_df = bootstrap_metric(metrics_df, 1000)

    records = []
    for task in bs_df.columns:
        records.append(create_ci_record(bs_df[task], task))

    summary_df = pd.DataFrame.from_records(records)
    summary_df.to_csv(
        f'{save_dir}/{table_name}_eval_summary_ious.csv', index=False)

    print(
        f"Evaluation result saved at {save_dir}/{table_name}_eval_summary_ious.csv")


if __name__ == '__main__':

    group_dir = '/deep/group/aihc-bootcamp-spring2020/localize'
    parser = ArgumentParser()

    parser.add_argument('--phase', type=str, default='valid',
                        help='valid or test')
    parser.add_argument('--pred_path', type=str, default='',
                        help='path to the segmentation file to be evaluated')
    parser.add_argument('--save_dir', type=str, default=f'{group_dir}/eval_results',
                        help='directory to stores evaluation result')

    args = parser.parse_args()

    phase = args.phase
    pred_path = args.pred_path
    save_dir = args.save_dir

    gt_path = f'{group_dir}/annotations/{phase}_encoded.json'
    table_name = f'{phase}'
    if pred_path == '':
        pred_path = f'{group_dir}/eval_results/gradcam/{phase}_gradcam_ensemble_encoded_prob_threshold.json'

    np.random.seed(0)
    evaluate(gt_path, pred_path, save_dir, table_name)
