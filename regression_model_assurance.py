"""
Run a simple linear regression for each pathology using the model’s probability output as the single independent variable and using the predicted evaluation metric (IoU or hit/miss) as the dependent variable. The script also runs a simple regression that uses the same approach as above, but that includes all 10 pathologies.
"""
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import torch

from eval_constants import CHEXPERT_TASKS, LOCALIZATION_TASKS
from utils import format_ci, run_linear_regression


def get_model_probability(map_dir):
    """
    Extract the predicted probability per cxr and per pathology given saliency model outputs
    """
    prob_dict = {}
    cxr_ids = []
    for task in sorted(LOCALIZATION_TASKS):
        print(f'Extracting model probability on {task}')
        probs = []
        pkl_paths = sorted(list(Path(map_dir).rglob(f"*{task}_map.pkl")))
        for pkl_path in pkl_paths:
            # get cxr id
            path = str(pkl_path).split('/')
            task = path[-1].split('_')[-2]
            cxr_id = '_'.join(path[-1].split('_')[:-2])
            # get model probability
            info = pickle.load(open(pkl_path,'rb'))
            if torch.is_tensor(info['prob']) and info['prob'].size()[0] == 14:
                prob_idx = CHEXPERT_TASKS.index(info['task'])
                pred_prob = info['prob'][prob_idx]
            else:
                pred_prob = info['prob']
            
            probs.append(pred_prob)
            if cxr_id not in cxr_ids:
                cxr_ids.append(cxr_id)
        prob_dict[task] = probs
    
    prob_df = pd.DataFrame.from_dict(prob_dict)
    prob_df['img_id'] = sorted(cxr_ids)
    return prob_df


def run_model_assurance_regression(args):
    """Run regression using model probability as the independent variable."""
    pred_results = pd.read_csv(args.pred_results)
    model_probs_df = get_model_probability(args.map_dir)
    y = args.metric

    coef_summary = pd.DataFrame(columns = ["lower", "mean", "upper",
                        "coef_pval","corr_lower", "corr","corr_upper",
                        "corr_pval", "feature", "task"])
    overall_regression = pd.DataFrame()
    for task in sorted(LOCALIZATION_TASKS):
        df = pd.DataFrame()
        # align localization perf metrics and probabilities
        ids = pred_results['img_id'].tolist()
        prob_results = model_probs_df[model_probs_df['img_id'].isin(ids)]
        # create regression data frame
        data = {y: pred_results[task].values,
                'prob': prob_results[task].tolist()}
        regression_df = pd.DataFrame(data)
        overall_regression = pd.concat([overall_regression, regression_df])

        # run regression
        results = run_linear_regression(regression_df, task, y, 'prob')
        coef_summary = pd.concat([coef_summary, results])

    # add overall regression
    results = run_linear_regression(overall_regression, 'Overall', y, 'prob')
    coef_summary = pd.concat([coef_summary, results])
    coef_summary = coef_summary.apply(format_ci,
                                      bonferroni_correction=1,
                                      axis = 1)\
                    [['task', 'n',
                      'Linear regression coefficients',
                      'Spearman correlations']]
    coef_summary.to_csv(f'{args.save_dir}/regression_modelprob_{y}.csv',
                        index = False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--metric', type=str,
                        help='options are: iou or hitmiss')
    parser.add_argument('--map_dir', type=str,
                        help='directory with pickle files containing heatmaps')
    parser.add_argument('--pred_results', type=str,
                        help='path to csv file with saliency method IoU or \
                              hit/miss results for each CXR and each pathology.')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='where to save regression results')
    args = parser.parse_args()
    assert args.metric in ['iou', 'hitmiss'], \
        "`metric` flag must be either `iou` or `hitmiss`"

    run_model_assurance_regression(args)
