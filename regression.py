"""
Run regression on pathology characteristics and model probability. All the inputs are in the cheXlocalize_data_code folder. 
"""
import glob
import json
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from eval_constants import LOCALIZATION_TASKS
from scipy import stats
import math

def normalize(column):
    if column.min() == column.max():
        return column
    
    return (column-column.min())/(column.max()-column.min())

def standardize(column):
    if column.min() == column.max():
        return column
    
    return (column-column.mean())/(column.std())

def run_linear_regression(regression_df, task, y, x):
    """
    Run linear regression model given a regression data frame of a single pathology.
    
    Args:
        task (string): localization task
        y (string): the dependent variable
        x (string): the independent variable
    """
     # run regression
    est = smf.ols(f"{y} ~ {x}", data = regression_df)
    est2 = est.fit()
    ci = est2.conf_int(alpha=0.05, cols=None)  # get ci 
    lower,upper = ci.loc[x]
    mean = est2.params.loc[x]
    pval = est2.pvalues.loc[x]
    corr, corr_pval = stats.spearmanr(regression_df[y].values,regression_df[x].values,nan_policy = 'omit')
    n = len(regression_df)
    stderr = 1.0 / math.sqrt(n - 3)
    delta = 1.96 * stderr
    lower_r = math.tanh(math.atanh(corr) - delta)
    upper_r = math.tanh(math.atanh(corr) + delta)

    # results
    results = {'lower': round(lower,3),
                'upper': round(upper,3),
                'mean': round(mean,3),
                'coef_pval': pval,
                'corr_lower': round(lower_r,3),
                'corr_upper': round(upper_r,3),
                'corr': round(corr,3),
                'corr_pval': corr_pval,
                'n': int(len(regression_df)),
                'feature': x,
                'task': task}
    return pd.DataFrame([results])

def format_ci(row):
    """
    Format confidence interval 
    """
    def format_stats_sig(p_val):
        """
        Output *, **, *** based on p-value
        """
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
    if row['feature'] != 'prob':
        p_val *= 4
    stats_sig_level = format_stats_sig(p_val)
    mean = row['mean']
    upper = row['upper']
    lower = row['lower']
    row['Linear regression coefficients'] =  f'{mean}, ({lower}, {upper}){stats_sig_level}'
    
    # CI on spearman correlations
    p_val = row['corr_pval']
    if row['feature'] != 'prob':
        p_val *= 4
    stats_sig_level = format_stats_sig(p_val)
    mean = row['corr']
    upper = row['corr_upper']
    lower = row['corr_lower']
    row['Spearman correlations'] =  f'{mean}, ({lower}, {upper}){stats_sig_level}'
    
    return row    

def regression_prob(input_path, output_path):
    """
    Run regression on probability and store results in csv
    """
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


def regression_geometric_features(input_path, output_path):
    # read localization performance
    hb_iou_tp = pd.read_csv((f'{input_path}/test_hb_tp_iou.csv'))
    gradcam_iou_tp = pd.read_csv(f'{input_path}/test_gradcam_densenet_ensemble_tp_iou.csv')


    hb_pt = pd.read_csv((f'{input_path}/test_hb_hit.csv'),index_col = False)
    gradcam_pt = pd.read_csv(f'{input_path}/test_gradcam_densenet_ensemble_hit.csv')

    # read geometric features
    instance_df = pd.read_csv(f'{feature_dir}/num_instances_test.csv').drop(['img_id'],axis=1)
    areas_df = pd.read_csv(f'{feature_dir}/area_ratio_test.csv').drop(['img_id'],axis=1)
    elongation_df = pd.read_csv(f'{feature_dir}/elogation_test.csv').drop(['img_id'],axis=1)
    rec_area_ratio_df = pd.read_csv(f'{feature_dir}/rec_area_ratio_test.csv').drop(['img_id'],axis=1)
    rec_area_ratio_df = 1-rec_area_ratio_df

    # Create pos only feature df
    neg_idx = instance_df[instance_df.eq(0).all(1)].index.values
    pos_idx = [i for i in instance_df.index.values if i not in neg_idx]

    instance_df_pos = instance_df.iloc[pos_idx]
    areas_df_pos = areas_df.iloc[pos_idx]
    elongation_df_pos = elongation_df.iloc[pos_idx]
    rec_area_ratio_df_pos = rec_area_ratio_df.iloc[pos_idx]

    # create regression dataframe 
    for y in ['gradcam_iou', 'iou_diff', 'gradcam_pt', 'pt_diff']:
        overall_regression = pd.DataFrame()
        coef_summary = pd.DataFrame(columns = ["lower","upper","mean","coef_pval","corr","corr_pval","corr_lower","corr_upper","feature"])
        for task in sorted(LOCALIZATION_TASKS):

            df = pd.DataFrame()

            data = {'gradcam_iou':gradcam_iou_tp[task].values,
                    'iou_diff': hb_iou_tp[task].values - gradcam_iou_tp[task].values,
                    'gradcam_pt': gradcam_pt[task].values,
                    'pt_diff':  hb_pt[task].values - gradcam_pt[task].values ,
                    'n_instance': instance_df_pos[task].values,
                    'area_ratio':areas_df_pos[task].values,
                    'elongation': elongation_df_pos[task].values,
                    'rec_area_ratio': rec_area_ratio_df_pos[task].values,
                    'img_id': hb_iou_tp['img_id']
                }
            df = pd.DataFrame(data) 
            df = df[df.n_instance>0]   # Positive only
            overall_regression = pd.concat([overall_regression,df])

        features = ['n_instance','area_ratio','elongation','rec_area_ratio']

        for feature in features:
            overall_regression[feature] = normalize(overall_regression[feature])
            results = run_linear_regression(overall_regression, 'Overall', y, feature)
            coef_summary = pd.concat([coef_summary,results])
        
        coef_summary = coef_summary.apply(format_ci,axis = 1).iloc[:,-3:-1]
        coef_summary.to_csv(f'{output_path}/regression_{y}.csv', index = False)

if __name__ == "__main__":
    # change input path if data is stored elsewhere
    input_path = '../cheXlocalize_data_code/paper_results/localization_performance'
    feature_dir = '../cheXlocalize_data_code/dataset/pathology_characteristics'
    output_path = '../cheXlocalize_data_code/paper_results/regression'
    regression_geometric_features(input_path, output_path)
    regression_prob(input_path, output_path)
