"""
Given predicted and ground-truth segmentations, compute mIoU and confidence
intervals for each of the 10 pathologies.

Usage: python3 eval_miou.py --phase val --save_dir /path/to/save/results
"""
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from pycocotools import mask
from tqdm import tqdm

from eval_constants import LOCALIZATION_TASKS


def calculate_iou(pred_mask, gt_mask, true_pos_only):
    """
    Calculate IoU score between two segmentation masks.

    Args:
        pred_mask (np.array): binary segmentation mask
        gt_mask (np.array): binary segmentation mask
    Returns:
        iou_score (np.float64)
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    if true_pos_only:
        if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
    else:
        if np.sum(gt_mask) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))

    return iou_score


def get_ious(gt_path, pred_path, true_pos_only):
    """
    Returns IoU scores for each combination of CXR and pathology in gt_path and pred_path.

    Args:
        gt_path (str): path to ground-truth segmentation json file (encoded)
        pred_path (str): path to predicted segmentation json file (encoded)
        true_pos_only (bool): if true, run evaluation only on the true positive
                              slice of the dataset (CXRs that contain predicted
                              and ground-truth segmentations)

    Returns:
        ious (dict): dict with 10 keys, one for each pathology (task). Values
                     are lists of all CXR IoU scores for the pathology key.
        cxr_ids (list): list of all CXR ids (e.g. 'patient64541_study1_view1_frontal').
    """
    with open(gt_path) as f:
        gt_dict = json.load(f)

    with open(pred_path) as f:
        pred_dict = json.load(f)

    ious = {}
    cxr_ids = sorted(gt_dict.keys())
    tasks = sorted(LOCALIZATION_TASKS)

    for task in tasks:
        print(f'Evaluating {task}')
        ious[task] = []

        for cxr_id in cxr_ids:
            # get ground-truth segmentation mask
            gt_item = gt_dict[cxr_id][task]
            gt_mask = mask.decode(gt_item)

            # get predicted segmentation mask
            if cxr_id not in pred_dict:
                pred_mask = np.zeros(gt_item['size'])
            else:
                pred_item = pred_dict[cxr_id][task]
                pred_mask = mask.decode(pred_item)

            assert gt_mask.shape == pred_mask.shape

            iou_score = calculate_iou(pred_mask, gt_mask, true_pos_only)
            ious[task].append(iou_score)

        assert len(ious[task]) == len(gt_dict.keys())

    return ious, cxr_ids


def bootstrap_metric(df, num_replicates):
    """Create dataframe of bootstrap samples."""
    def single_replicate_performances():
        sample_ids = np.random.choice(len(df), size=len(df), replace=True)
        replicate_performances = {}
        df_replicate = df.iloc[sample_ids]

        for task in df[LOCALIZATION_TASKS].columns:
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
    sorted_perfs = series.sort_values()
    lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
    upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
    lower = sorted_perfs.iloc[lower_index].round(3)
    upper = sorted_perfs.iloc[upper_index].round(3)
    mean = round(sorted_perfs.mean(),3)
    return lower, mean, upper


def create_ci_record(perfs, task):
    lower, mean, upper = compute_cis(perfs, confidence_level = 0.05)
    record = {"name": task,
              "lower": lower,
              "mean": mean,
              "upper": upper}
    return record


def create_map(pkl_path):
    """
    Create saliency map of original img size·
    """
    info = pickle.load(open(pkl_path,'rb'))
    saliency_map = info['map']
    img_dims = info['cxr_dims']
    map_resized = F.interpolate(saliency_map, size=(img_dims[1],img_dims[0]), mode='bilinear', align_corners=False)
    saliency_map = map_resized.squeeze().squeeze().detach().cpu().numpy()
    return saliency_map, img_dims


def get_hit_rates(gt_path, pred_path):
    """
    TODO: make this comment clearer
    TODO: also, it looks like this function expects a different path? like a
    path with a bunch of pkl files. is that right? where do these pkl files come from?
    Calculate hit rate·
    - We need to figure 
    """
    with open(gt_path) as f:
        gt_dict = json.load(f)
	
    all_paths = sorted(list(Path(pred_path).rglob("*_map.pkl")))
    results = {}
    for pkl_path in tqdm(all_paths):
        # break down path to image name and task
        path = str(pkl_path).split('/')
        task = path[-1].split('_')[-2]
        img_id = '_'.join(path[-1].split('_')[:-2])

        if task not in LOCALIZATION_TASKS:
            print(f"Invalid task {task}")
            continue

        if img_id in results:
            if task in results[img_id]:
                print(f'Check for duplicates for {task} for {img_id}')
                break
            else:
                results[img_id][task] = 0
        else:
            # get ground truth binary mask
            if img_id not in gt:
                continue
            else:
                results[img_id] = {}
                results[img_id][task] = 0

        gt_item = gt_dict[img_id][task]
        gt_mask = mask.decode(gt_item)

        # get saliency heatmap
        sal_map, img_dims = create_map(pkl_path)

        x =  np.unravel_index(np.argmax(sal_map, axis = None), sal_map.shape)[0]
        y = np.unravel_index(np.argmax(sal_map, axis = None), sal_map.shape) [1]

        assert (gt_mask.shape == sal_map.shape)
        if(gt_mask[x][y]==1):
            results[img_id][task] = 1
        elif (np.sum(gt_mask)==0):
            results[img_id][task] = np.nan

    all_ids = sorted(gt_dict.keys())
    return results, all_ids


def evaluate(gt_path, pred_path, save_dir, metric, true_pos_only):
    """
    Evaluate localization performance using mIoU or hit rate. Return miou by pathologies
    and their confidence intervals
    #TODO: format of return
    """
    # create save_dir if it does not already exist
    Path(save_dir).mkdir(exist_ok=True,parents=True)

    # TODO: factor this out?
    if metric == 'miou':
        ious, cxr_ids = get_ious(gt_path, pred_path, true_pos_only)
        metric_df = pd.DataFrame.from_dict(ious)
        metric_df['img_id'] = cxr_ids
        metric_df.to_csv(f'{save_dir}/iou_results.csv',index = False)

        bs_df = bootstrap_metric(metric_df, 1000)
        bs_df.to_csv(f'{save_dir}/bootstrap_iou_results.csv',index = False)
    elif metric == 'hitrate':
        results, cxr_ids = get_hit_rates(gt_path, pred_path)
        metrics = pd.DataFrame.from_dict(results,orient='index')
        bs_df= bootstrap_metric(metrics, 1000)
        bs_df.to_csv(f'{save_dir}/{table_name}_bs_hit.csv',index = False)
        metrics['img_id'] = all_ids
        metrics.to_csv(f'{save_dir}/{table_name}_hit.csv',index = False)
    else:
        raise ValueError('`metric` must be either `miou` or `hitrate`')

    # get confidence intervals
    records = []
    for task in bs_df.columns:
        records.append(create_ci_record(bs_df[task], task))

    summary_df = pd.DataFrame.from_records(records)
    print(summary_df)
    summary_df.to_csv(f'{save_dir}/summary_iou_results.csv',index = False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_path', type=str,
                        help='directory where ground-truth segmentations are \
                              saved (encoded)')
    parser.add_argument('--pred_path', type=str,
                        help='directory where predicted segmentations are saved \
                              saved (encoded)')
    parser.add_argument('--metric', type=str,
                        help='options are: miou or hitrate')
    parser.add_argument('--true_pos_only', default="True",
                        help='if true, run evaluation only on the true positive \
                        slice of the dataset (CXRs that contain predicted and \
                        ground-truth segmentations)')
    parser.add_argument('--save_dir', default=".",
                        help='where to save evaluation results')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to fix')
    args = parser.parse_args()

    assert args.metric == 'miou' or args.metric == 'hitrate', \
        "`metric` flag must be either `miou` or `hitrate`"

    np.random.seed(args.seed)

    evaluate(args.gt_path, args.pred_path, args.save_dir, args.metric,
             eval(args.true_pos_only))
