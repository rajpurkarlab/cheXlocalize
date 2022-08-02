"""
TODO
"""
from argparse import ArgumentParser
import json
import pickle
import numpy as np
from pathlib import Path
import torch
from eval_constants import LOCALIZATION_TASKS
from utils import run_linear_regression

def get_model_probability(map_dir):
    """
    Extract the predicted probability per cxr and per pathology given saliency model outputs
    """
    prob_dict = {}
    for task in sorted(LOCALIZATION_TASKS):
        print(f'Extracting model probability on {task}')
        probs = []
        pkl_paths = sorted(list(Path(map_dir).rglob(f"*{task}_map.pkl")))
        for pkl_path in pkl_paths:
            # get model probability
            info = pickle.load(open(pkl_path,'rb'))
            if torch.is_tensor(info['prob']) and info['prob'].size()[0] == 14:
                prob_idx = CHEXPERT_TASKS.index(info['task'])
                pred_prob = info['prob'][prob_idx]
            else:
                pred_prob = info['prob']

            probs.append(pred_prob)
        prob_dict[task] = probs
    
    prob_df = pd.DataFrame.from_dict(prob_dict)
    return prob_df

def run_model_assurance_regression(args):
    """Run regression using model probability as the independent variable."""
    gradcam_iou = pd.read_csv(f'{input_path}/test_gradcam_densenet_ensemble_full_iou.csv') # this is the output of mIoU evaluation on the full dataset. We can add a flag such as --pred_miou_results
    gradcam_pt = pd.read_csv(f'{input_path}/test_gradcam_densenet_ensemble_ptgame.csv') 
    probs_df = get_model_probability(args.map_dir)
    # probs_df = pd.read_csv(f'{input_path}/prob.csv')

    for y in ['iou', 'pt']:
        coef_summary = pd.DataFrame(columns = ["lower","mean","upper","coef_pval","corr_lower", "corr","corr_upper","corr_pval", "feature","task"])
        overall_regression = pd.DataFrame()
        for task in sorted(LOCALIZATION_TASKS):
            df = pd.DataFrame()

            data = {'iou':gradcam_iou[task].values,
                    'pt': gradcam_pt[task].values,
                    'prob': probs_df[task].tolist()
                }
            regression_df = pd.DataFrame(data) 
            overall_regression = pd.concat([overall_regression,regression_df])

            # run regression
            results = run_linear_regression(regression_df, task, y, 'prob')
            coef_summary = pd.concat([coef_summary,results])

        # add overall regression
        results = run_linear_regression(overall_regression, 'Overall', y, 'prob')
        coef_summary = pd.concat([coef_summary,results])
        coef_summary = coef_summary.apply(format_ci,axis = 1).iloc[:,-4:]
        coef_summary.to_csv(f'{output_path}/regression_prob_{y}.csv', index = False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--map_dir', type=str,
                        help='directory with pickle files containing heatmaps')
    parser.add_argument('--threshold_path', type=str,
                        help="csv file that stores pre-defined threshold values. \
                        If no path is given, script uses Otsu's.")
    parser.add_argument('--output_path', type=str,
                        default='./saliency_segmentations.json',
                        help='json file path for saving encoded segmentations')
    args = parser.parse_args()

    run_model_assurance_regression(args)
