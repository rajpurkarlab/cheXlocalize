import math
import numpy as np
import pandas as pd
from pycocotools import mask
from scipy import stats
import statsmodels.formula.api as smf


def encode_segmentation(segmentation_arr):
    """
    Encode a binary segmentation (np.array) to RLE format using the pycocotools Mask API.
    Args:
        segmentation_arr (np.array): [h x w] binary segmentation
    Returns:
		Rs (dict): the encoded mask in RLE format
    """
    segmentation = np.asfortranarray(segmentation_arr.astype('uint8'))
    Rs = mask.encode(segmentation)
    Rs['counts'] = Rs['counts'].decode()
    return Rs


def run_linear_regression(regression_df, task, y, x):
    """
    Run linear regression model given a regression dataframe of a single pathology.

    Args:
        task (str): localization task
        y (str): the dependent variable
        x (str): the independent variable
    """
    est = smf.ols(f"{y} ~ {x}", data = regression_df)
    est2 = est.fit()
    ci = est2.conf_int(alpha=0.05, cols=None)
    lower, upper = ci.loc[x]
    mean = est2.params.loc[x]
    pval = est2.pvalues.loc[x]
    corr, corr_pval = stats.spearmanr(regression_df[y].values,regression_df[x].values,nan_policy = 'omit')
    n = len(regression_df)
    stderr = 1.0 / math.sqrt(n - 3)
    delta = 1.96 * stderr
    lower_r = math.tanh(math.atanh(corr) - delta)
    upper_r = math.tanh(math.atanh(corr) + delta)

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
