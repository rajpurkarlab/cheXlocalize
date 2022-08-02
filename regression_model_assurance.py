"""
TODO
"""
from argparse import ArgumentParser

from eval_constants import LOCALIZATION_TASKS
from utils import run_linear_regression


def run_model_assurance_regression(args):
    """Run regression using model probability as the independent variable."""
    gradcam_iou = pd.read_csv(f'{input_path}/test_gradcam_densenet_ensemble_full_iou.csv')
    gradcam_pt = pd.read_csv(f'{input_path}/test_gradcam_densenet_ensemble_ptgame.csv') 
    probs_df = pd.read_csv(f'{input_path}/prob.csv')

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
            regression_df = regression_df[regression_df.iou>-1] # exclude true negatives for miou calculation
            overall_regression = pd.concat([overall_regression,regression_df])
            # overall_regression = overall_regression.append(regression_df)

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
