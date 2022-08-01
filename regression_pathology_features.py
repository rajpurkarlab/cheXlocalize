"""
Run regression using a pathology characteristic as the independent variable and
evaluation metric as the dependent variable.
"""
from argparse import ArgumentParser
import pandas as pd

from eval_constants import LOCALIZATION_TASKS
from utils import run_linear_regression


def normalize(column):
    if column.min() == column.max():
        return column
    else:
        return (column-column.min())/(column.max()-column.min())


def format_ci(row):
    """Format confidence interval."""
    def format_stats_sig(p_val):
        """Output *, **, *** based on p-value."""
        stats_sig_level = ''
        if p_val < 0.001:
            stats_sig_level = '***'
        elif p_val < 0.01:
            stats_sig_level = '**'
        elif p_val < 0.05:
            stats_sig_level = '*'
        return stats_sig_level

    # CI on linear regression coefficients
    p_val = row['coef_pval']
    p_val *= 4 # to get Bonferroni corrected p-values
    stats_sig_level = format_stats_sig(p_val)
    mean = row['mean']
    upper = row['upper']
    lower = row['lower']
    row['Linear regression coefficients'] =  f'{mean}, ({lower}, {upper}){stats_sig_level}'

    # CI on spearman correlations
    p_val = row['corr_pval']
    p_val *= 4 # to get Bonferroni corrected p-values
    stats_sig_level = format_stats_sig(p_val)
    mean = row['corr']
    upper = row['corr_upper']
    lower = row['corr_lower']
    row['Spearman correlations'] =  f'{mean}, ({lower}, {upper}){stats_sig_level}'

    return row


def run_features_regression(args):
    # read localization performance
    pred_miou_results = pd.read_csv(args.pred_miou_results)
    pred_hitrate_results = pd.read_csv(args.pred_hitrate_results)
    if args.evaluate_hb:
        hb_miou_results = pd.read_csv(args.hb_miou_results)
        hb_hitrate_results = pd.read_csv(args.hb_hitrate_results)

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
    if args.evaluate_hb:
        regression_cols = ['pred_miou', 'miou_diff', 'pred_hitrate', 'hitrate_diff']
    else:
        regression_cols = ['pred_miou', 'pred_hitrate']

    for y in regression_cols:
        overall_regression = pd.DataFrame()
        coef_summary = pd.DataFrame(columns = ['lower', 'upper', 'mean',
            'coef_pval', 'corr', 'corr_pval', 'corr_lower', 'corr_upper', 'feature'])
        for task in sorted(LOCALIZATION_TASKS):
            df = pd.DataFrame()
            data = {'pred_miou': pred_miou_results[task].values,
                    'pred_hitrate': pred_hitrate_results[task].values,
                    'n_instance': instance_df_pos[task].values,
                    'area_ratio':area_df_pos[task].values,
                    'elongation': elongation_df_pos[task].values,
                    'irrectangularity': irrectangularity_df_pos[task].values,
                    'img_id': hb_miou_results['img_id']}
            if args.evaluate_hb:
                data['miou_diff'] =     hb_miou_results[task].values \
                                            - pred_miou_results[task].values
                data['hitrate_diff'] =  hb_hitrate_results[task].values \
                                            - pred_hitrate_results[task].values
            df = pd.DataFrame(data)
            # get cxrs with ground-truth segmentation for this task
            df = df[df.n_instance>0]
            overall_regression = pd.concat([overall_regression, df])

        features = ['n_instance', 'area_ratio', 'elongation', 'irrectangularity']
        for feature in features:
            overall_regression[feature] = normalize(overall_regression[feature])
            results = run_linear_regression(overall_regression, 'Overall', y, feature)
            coef_summary = pd.concat([coef_summary, results])

        coef_summary = coef_summary.apply(format_ci,axis = 1)\
                        [['task', 'Linear regression coefficients']]
        coef_summary.to_csv(f'{args.save_dir}/regression_{y}.csv', index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--features_dir', type=str,
                        help='directory with four csv files: area_ratio.csv, \
                              elongation.csv, num_instances.csv`, and \
                              rec_area_ratio.csv.')
    parser.add_argument('--pred_miou_results', type=str,
                        help='path to csv file with saliency method IoU results \
                              for each CXR and each pathology.')
    parser.add_argument('--pred_hitrate_results', type=str,
                        help='path to csv file with saliency method hit/miss \
                              results for each CXR and each pathology.')
    parser.add_argument('--evaluate_hb', type=bool, default=False,
                        help='if true, evaluate human benchmark in addition to \
                              saliency method.')
    parser.add_argument('--hb_miou_results', type=str,
                        help='path to csv file with human benchmark IoU results \
                              for each CXR and each pathology.')
    parser.add_argument('--hb_hitrate_results', type=str,
                        help='path to csv file with human benchmark hit/miss \
                              results for each CXR and each pathology.')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='where to save regression results')
    args = parser.parse_args()

    run_features_regression(args)
