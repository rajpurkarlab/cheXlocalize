""" Given segmentation and radiologist ground truth, compute the mIoU and the confidence interval per task.
    
    Usage: python3 eval_miou.py --phase val --save_dir /path/to/save/results
"""
import numpy as np
from PIL import Image
from pycocotools import mask
import pandas as pd

import json
from pathlib import Path
from argparse import ArgumentParser
from eval_constants import LOCALIZATION_TASKS

def iou_seg(pred_mask,gt_mask, tp = False):
    """
    Calculate iou scores of two segmentation masks
    
    Args: 
        pred_mask (np.array): binary segmentation mask
        gt_mask (np.array): binary segmentation mask
    Returns:
        iou score (a scalar)
    """
    intersection = np.logical_and(pred_mask,gt_mask)
    union = np.logical_or(pred_mask,gt_mask)
    
    # tp only\
    if tp:
        if np.sum(pred_mask) ==0 or np.sum(gt_mask)==0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
    else:
        if np.sum(gt_mask)==0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
    
    return iou_score


def compute_metrics(gt_dir, pred_dir, tp = False):
    """
    Take in ground truth and prediction json (both encoded) and return iou for each image under each pathology
    """
    with open(gt_dir) as f:
        gt = json.load(f)
        
    with open(pred_dir) as f:
        pred = json.load(f)
        
    ious = {}
    all_ids = sorted(gt.keys())
    tasks = sorted(LOCALIZATION_TASKS)

    for task in tasks:
        print(f'Evaluating {task}')
        ious[task] = []
   
        for img_id in all_ids:
             
            
            # get ground_truth segmentation mask
            gt_item = gt[img_id][task]
            gt_mask = mask.decode(gt_item)
            
            # get predicted segmentation mask
            if img_id not in pred:
                pred_mask = np.zeros(gt_item['size'])
            else:
                pred_item = pred[img_id][task]
                pred_mask = mask.decode(pred_item)
            
            assert gt_mask.shape == pred_mask.shape
            
            iou_score = iou_seg(pred_mask, gt_mask, tp = tp)
            ious[task].append(iou_score)
        
    
        assert len(ious[task]) == len(gt.keys())
            
    return ious, all_ids


def bootstrap_metric(df, num_replicates, metric = 'iou'):
    """
    Create dataframe of bootstrap samples 
    """
    def single_replicate_performances():
        sample_ids = np.random.choice(len(df), size=len(df), replace=True)
        replicate_performances = {}
        df_replicate = df.iloc[sample_ids]

        for task in df.columns:
            if metric == 'iou':
                performance = df_replicate[task].mean()
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
    mean = round(sorted_perfs.mean(),3)
    return lower, mean, upper

def create_ci_record(perfs, name):
    lower, mean, upper = compute_cis(perfs, confidence_level = 0.05)
    record = {"name": name,
              "lower": lower,
              "mean": mean,
              "upper": upper,
                  }
    return record     


def evaluate(gt_dir, pred_dir,save_dir,table_name, tp = False):
    """
    Pipeline to evaluate localizations. Return miou by pathologies and their confidence intervals
    """
    
    # create save dir if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True,parents=True)

    metrics, all_ids = compute_metrics(gt_dir, pred_dir, tp = tp)
    metrics_df = pd.DataFrame.from_dict(metrics)
    
    bs_df = bootstrap_metric(metrics_df,1000)
    bs_df.to_csv(f'{save_dir}/{table_name}_bs_iou.csv',index = False)
    metrics_df['img_id'] = all_ids
    metrics_df.to_csv(f'{save_dir}/{table_name}_iou.csv',index = False)
    
    records = []
    for task in bs_df.columns:
        records.append(create_ci_record(bs_df[task], task))
   
    summary_df = pd.DataFrame.from_records(records)
    print(summary_df)
    summary_df.to_csv(f'{save_dir}/{table_name}_summary.csv',index = False)
    
    print(f"Evaluation result saved at {save_dir}/{table_name}_summary.csv")


if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument('--method', type=str, required=True,
                        help='gradcam, ig, gradcam++, human')
    parser.add_argument('--phase', type=str, required=True,
                        help='enter val or test')
    parser.add_argument('--model_type', type=str, default = 'ensemble',
                        help='single, ensemble or baseline')
    parser.add_argument('--model', type=str, default = 'densenet',
                        help='densenet, inception or resnet')
    parser.add_argument('--tp', default="False",
                    help='if tp only')
    parser.add_argument('--save_dir', default=".",
                help='directory where the evaluation result will be saved')
    
    args = parser.parse_args()
    
    method = args.method
    phase = args.phase
    model_type = args.model_type
    model = args.model
    tp = eval(args.tp)
    save_dir = args.save_dir
    
    group_dir = '/deep/group/aihc-bootcamp-spring2020/localize'
    gt_path = f'{group_dir}/annotations/{phase}_encoded.json'
    
    if method == 'human':
        assert phase == 'test'
        pred_path = f'{group_dir}/annotations/vietnam_encoded.json'
        
        if tp:
            table_name = f'{phase}_{method}_tp'
        else:
            table_name = f'{phase}_{method}'
    else:
        if tp:
            table_name = f'{phase}_{method}_{model}_{model_type}_tp'
        else:
            table_name = f'{phase}_{method}_{model}_{model_type}'
        
    pred_path = f'{group_dir}/eval_results/{method}/{phase}_{method}_{model}_{model_type}_encoded.json'
    np.random.seed(0)
    print(f"Save results at {save_dir}/{table_name}")
    evaluate(gt_path, pred_path, save_dir,table_name, tp = tp)
