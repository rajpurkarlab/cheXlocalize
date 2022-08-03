from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from PIL import Image
from pycocotools import mask
import torch.nn.functional as F
from tqdm import tqdm

from eval_constants import LOCALIZATION_TASKS
from heatmap_to_segmentation import pkl_to_mask


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
        if np.sum(union) == 0:
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
                              and ground-truth segmentations); if false, also
                              include CXRs with a predicted segmentation but
                              without a ground-truth segmentation, and include
                              CXRs with a ground-truth segmentation but without
                              a predicted segmentation.

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
    tasks = sorted(LOCALIZATION_TASKS)

    for task in tasks:
        cxr_ids = sorted(gt_dict.keys())
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

        # if true_pos_only is false, include cxrs that do not have ground-truth
        # segmentations but that have predicted segmentations
        if not true_pos_only:
            for cxr_id in sorted(pred_dict.keys()):
                if cxr_id not in gt_dict:
                    pred_item = pred_dict[cxr_id][task]
                    pred_mask = mask.decode(pred_item)
                    gt_mask = np.zeros(pred_item['size'])
                    assert gt_mask.shape == pred_mask.shape
                    iou_score = calculate_iou(pred_mask, gt_mask, true_pos_only)
                    ious[task].append(iou_score)
                    cxr_ids.append(cxr_id)
        else:
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


def get_hitrates(gt_path, pred_path):
    """
	Args:
        gt_path (str): directory where ground-truth segmentations are saved (encoded)
        pred_path (str): directory with pickle file containing heat maps
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
            if img_id not in gt_dict:
                continue
            else:
                results[img_id] = {}
                results[img_id][task] = 0

        gt_item = gt_dict[img_id][task]
        gt_mask = mask.decode(gt_item)

        # get saliency heatmap
        sal_map = pkl_to_mask(pkl_path)
        x = np.unravel_index(np.argmax(sal_map, axis = None), sal_map.shape)[0]
        y = np.unravel_index(np.argmax(sal_map, axis = None), sal_map.shape)[1]

        assert (gt_mask.shape == sal_map.shape)
        if (gt_mask[x][y]==1):
            results[img_id][task] = 1
        elif (np.sum(gt_mask)==0):
            results[img_id][task] = np.nan

    all_ids = sorted(gt_dict.keys())
    results_df = pd.DataFrame.from_dict(results, orient='index')
    return results_df, all_ids


def get_hb_hitrates(gt_path, pred_path):
    """
	Args:
        gt_path (str): directory where ground-truth segmentations are saved (encoded)
        pred_path (str): json file with human annotations for most representative point
    """
    with open(pred_path) as f:
        hb_salient_pts = json.load(f)
    with open(gt_path) as f:
        gt_dict = json.load(f)

    # evaluate hit
    results = {}
    all_ids = sorted(gt_dict.keys())
    for task in sorted(LOCALIZATION_TASKS):
        print(f'Evaluating {task}')
        results[task] = []
        for img_id in all_ids:
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

            results[task].append(hit)

    results['cxr_id'] = all_ids
    results_df = pd.DataFrame.from_dict(results)
    results_df = results_df.set_index('cxr_id')
    return results_df, all_ids


def evaluate(gt_path, pred_path, save_dir, metric, true_pos_only, if_human_benchmark):
    """
	Generates and saves three csv files:
	-- `{iou/hitmiss}_results.csv`: IoU or hit/miss results for each CXR and
                                     each pathology.
	-- `{iou/hitmiss}_bootstrap_results.csv`: 1000 bootstrap samples of IoU
                                               or hit/miss for each pathology.
	-- `{miou/hitrate}_summary_results.csv`: mIoU or hit rate 95% bootstrap
                                             confidence intervals for each pathology.
    """
    # create save_dir if it does not already exist
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    if metric == 'iou':
        ious, cxr_ids = get_ious(gt_path, pred_path, true_pos_only)
        metric_df = pd.DataFrame.from_dict(ious)
    elif metric == 'hitmiss' and if_human_benchmark == False:
        metric_df, cxr_ids = get_hitrates(gt_path, pred_path)
    elif metric == 'hitmiss' and if_human_benchmark == True:
        metric_df, cxr_ids = get_hb_hitrates(gt_path, pred_path)
    else:
        raise ValueError('`metric` must be either `iou` or `hitmiss`')

    hb = 'humanbenchmark_' if if_human_benchmark else ''

    metric_df['img_id'] = cxr_ids
    metric_df = metric_df.sort_values(by='img_id')
    metric_df.to_csv(f'{save_dir}/{metric}_{hb}results_per_cxr.csv', index=False)

    bs_df = bootstrap_metric(metric_df, 1000)
    bs_df.to_csv(f'{save_dir}/{metric}_{hb}bootstrap_results_per_cxr.csv', index=False)

    # get confidence intervals
    records = []
    for task in bs_df.columns:
        records.append(create_ci_record(bs_df[task], task))

    summary_df = pd.DataFrame.from_records(records).sort_values(by='name')
    print(summary_df)
    summary_df.to_csv(f'{save_dir}/{metric}_summary_results.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--metric', type=str,
                        help='options are: iou or hitmiss')
    parser.add_argument('--gt_path', type=str,
                        help='directory where ground-truth segmentations are \
                              saved (encoded)')
    parser.add_argument('--pred_path', type=str,
                        help='json path where predicted segmentations are saved \
                              (if metric = iou) or directory with pickle files \
							  containing heat maps (if metric = hitmiss and \
                              if_human_benchmark = false) or json path with \
                              human annotations for most representative points \
                              (if metric = hitmiss and if_human_benchmark = \
                              true)')
    parser.add_argument('--true_pos_only', type=str, default='True',
                        help='if true, run evaluation only on the true positive \
                        slice of the dataset (CXRs that contain predicted and \
                        ground-truth segmentations); if false, also include cxrs \
                        with a predicted segmentation but without a ground-truth \
                        segmentation, and include cxrs with a ground-truth\
                        segmentation but without a predicted segmentation.')
    parser.add_argument('--save_dir', default='.',
                        help='where to save evaluation results')
    parser.add_argument('--if_human_benchmark', type=str, default='False',
                        help='if true, scripts expects human benchmark inputs')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to fix')
    args = parser.parse_args()

    assert args.metric in ['iou', 'hitmiss'], \
        "`metric` flag must be either `iou` or `hitmiss`"
    assert args.if_human_benchmark in ['True', 'False'], \
        "`if_human_benchmark` flag must be either `True` or `False`"

    np.random.seed(args.seed)

    evaluate(args.gt_path, args.pred_path, args.save_dir, args.metric,
             eval(args.true_pos_only), eval(args.if_human_benchmark))
