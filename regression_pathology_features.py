"""
Run regression using a pathology characteristic as the independent variable and
evaluation metric as the dependent variable.
"""
from argparse import ArgumentParser
import pandas as pd

from eval_constants import LOCALIZATION_TASKS
from utils import format_ci, run_linear_regression


def normalize(column):
    if column.min() == column.max():
        return column
    else:
        return (column-column.min())/(column.max()-column.min())


def run_features_regression(args):
    evaluate_hb = eval(args.evaluate_hb)

    # read localization performance
    pred_iou_results = pd.read_csv(args.pred_iou_results)
    pred_hitmiss_results = pd.read_csv(args.pred_hitmiss_results)
    if evaluate_hb:
        hb_iou_results = pd.read_csv(args.hb_iou_results)
        hb_hitmiss_results = pd.read_csv(args.hb_hitmiss_results)

    # read geometric features
    instance_df =       pd.read_csv(f'{args.features_dir}/num_instances.csv').\
                           drop(['img_id'], axis=1)
    area_df =           pd.read_csv(f'{args.features_dir}/area_ratio.csv').\
                           drop(['img_id'], axis=1)
    elongation_df =     pd.read_csv(f'{args.features_dir}/elongation.csv').\
                           drop(['img_id'], axis=1)
    rec_area_ratio_df = pd.read_csv(f'{args.features_dir}/rec_area_ratio.csv').\
                           drop(['img_id'], axis=1)
    irrectangularity_df = 1-rec_area_ratio_df # irrectangularity = 1-(area_ratio)

    # get cxrs with at least one ground-truth segmentation for any pathology
    neg_idx = instance_df[instance_df.eq(0).all(1)].index.values
    pos_idx = [i for i in instance_df.index.values if i not in neg_idx]
    instance_df_pos = instance_df.iloc[pos_idx]
    area_df_pos = area_df.iloc[pos_idx]
    elongation_df_pos = elongation_df.iloc[pos_idx]
    irrectangularity_df_pos = irrectangularity_df.iloc[pos_idx]

    # create regression dataframe 
    if evaluate_hb:
        regression_cols = ['pred_iou', 'iou_diff', 'pred_hitmiss', 'hitmiss_diff']
    else:
        regression_cols = ['pred_iou', 'pred_hitmiss']

    for y in regression_cols:
        overall_regression = pd.DataFrame()
        coef_summary = pd.DataFrame(columns = ['lower', 'upper', 'mean',
            'coef_pval', 'corr', 'corr_pval', 'corr_lower', 'corr_upper', 'feature'])
        for task in sorted(LOCALIZATION_TASKS):
            df = pd.DataFrame()
            data = {'pred_iou': pred_iou_results[task].values,
                    'pred_hitmiss': pred_hitmiss_results[task].values,
                    'n_instance': instance_df_pos[task].values,
                    'area_ratio':area_df_pos[task].values,
                    'elongation': elongation_df_pos[task].values,
                    'irrectangularity': irrectangularity_df_pos[task].values,
                    'img_id': pred_iou_results['img_id']}
            if evaluate_hb:
                data['iou_diff'] =     hb_iou_results[task].values \
                                            - pred_iou_results[task].values
                data['hitmiss_diff'] =  hb_hitmiss_results[task].values \
                                            - pred_hitmiss_results[task].values
            df = pd.DataFrame(data)
            # get cxrs with ground-truth segmentation for this task
            df = df[df.n_instance>0]
            overall_regression = pd.concat([overall_regression, df])

        features = ['n_instance', 'area_ratio', 'elongation', 'irrectangularity']
        for feature in features:
            overall_regression[feature] = normalize(overall_regression[feature])
            results = run_linear_regression(overall_regression, 'Overall', y, feature)
            coef_summary = pd.concat([coef_summary, results])

        coef_summary = coef_summary.apply(format_ci,
                                          bonferroni_correction=4,
                                          axis=1)\
                        [['task', 'feature', 'Linear regression coefficients']]
        coef_summary.to_csv(f'{args.save_dir}/regression_features_{y}.csv',
                            index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--features_dir', type=str,
                        help='directory with four csv files: area_ratio.csv, \
                              elongation.csv, num_instances.csv`, and \
                              rec_area_ratio.csv.')
    parser.add_argument('--pred_iou_results', type=str,
                        help='path to csv file with saliency method IoU results \
                              for each CXR and each pathology.')
    parser.add_argument('--pred_hitmiss_results', type=str,
                        help='path to csv file with saliency method hit/miss \
                              results for each CXR and each pathology.')
    parser.add_argument('--evaluate_hb', type=str, default='False',
                        help='if true, evaluate human benchmark in addition to \
                              saliency method.')
    parser.add_argument('--hb_iou_results', type=str,
                        help='path to csv file with human benchmark IoU results \
                              for each CXR and each pathology.')
    parser.add_argument('--hb_hitmiss_results', type=str,
                        help='path to csv file with human benchmark hit/miss \
                              results for each CXR and each pathology.')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='where to save regression results')
    args = parser.parse_args()
    assert args.evaluate_hb in ['True', 'False'], \
        "`evaluate_hb` flag must be either `True` or `False`"

    run_features_regression(args)
