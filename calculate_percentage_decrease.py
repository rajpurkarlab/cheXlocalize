"""
Calculate the percentage decrease from human benchmark localization
performance to saliency method pipeline localization performance, along
with the 95% CIs.
"""
from argparse import ArgumentParser
import numpy as np
import pandas as pd

from eval import compute_cis, create_ci_record


def create_pct_diff_df(metric, pred_bootstrap_results, hb_bootstrap_results,
                       save_dir):
    """
    Calculate percentage decrease per pathology from human benchmark
    localization metric to saliency method pipeline localization metric,
    and obtain 95% CI on the percentage decreases.
    """
    # get 1000 bootstrap samples of IoU or hit/miss
    pred_bs = pd.read_csv(pred_bootstrap_results)
    hb_bs = pd.read_csv(hb_bootstrap_results)

    # use the percentage difference as the statistic;
    # get the CI (2.5th and 97.5th percentile) on the percentage difference
    pct_diff_bs = (hb_bs - pred_bs)/hb_bs
    records = []
    for task in pct_diff_bs.columns:
        records.append(create_ci_record(pct_diff_bs[task], task))
    pct_diff_ci = pd.DataFrame.from_records(records)

    # create results df
    pct_diff_df = pd.DataFrame()
    pct_diff_df['hb'] = hb_bs.mean()
    pct_diff_df['pred'] = pred_bs.mean()
    pct_diff_df['pct_diff'] = round(
            (pct_diff_df['hb']-pct_diff_df['pred'])/pct_diff_df['hb']*100,
            3)
    pct_diff_df['pct_diff_lower'] = round(pct_diff_ci['lower'] * 100, 3).\
                                        tolist()
    pct_diff_df['pct_diff_upper'] = round(pct_diff_ci['upper'] * 100, 3).\
                                        tolist()
    pct_diff_df = pct_diff_df.sort_values(['pct_diff'], ascending=False)

    # calculate avg human benchmark and saliency method localization metric
    avg_pred = round(pct_diff_df['pred'].mean(), 3)
    avg_hb = round(pct_diff_df['hb'].mean(), 3)
    avg_pct_diff = round((avg_hb-avg_pred)/avg_hb * 100, 3)

    # find the 95% CI of the percentage difference between average saliency
    # method and average human benchmark
    #   - get bootstrap sample of average saliency method localization metric
    avg_pred_bs = pred_bs.mean(axis=1)
    #   - get bootstrap sample of average human benchmark localization metric
    avg_hb_bs = hb_bs.mean(axis=1)
    #   - use pct diff between avg saliency and avg human benchmark as the
    #     statistic, and get the 2.5th and 97.5th percentile of the bootstrap
    #     distribution to create the CI
    avg_bs_df = 100 * (avg_hb_bs - avg_pred_bs)/avg_hb_bs
    lower, mean, upper = compute_cis(avg_bs_df, confidence_level = 0.05)

    pct_diff_df.loc['Average'] = {'pred': avg_pred, 'hb': avg_hb,
                                  'pct_diff': avg_pct_diff,
                                  'pct_diff_lower': round(lower, 3),
                                  'pct_diff_upper': round(upper, 3)}
    print(pct_diff_df)
    pct_diff_df.to_csv(f'{save_dir}/{metric}_pct_decrease.csv')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--metric', type=str,
                        help='options are: miou or hitrate')
    parser.add_argument('--hb_bootstrap_results', type=str,
                        help='path to csv file with 1000 bootstrap samples of \
                              human benchmark IoU or hit/miss for each \
                              pathology')
    parser.add_argument('--pred_bootstrap_results', type=str,
                        help='path to csv file with 1000 bootstrap samples of \
                              saliency method IoU or hit/miss for each \
                              pathology')
    parser.add_argument('--save_dir', default='.',
                        help='where to save results')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to fix')
    args = parser.parse_args()

    assert args.metric in ['miou', 'hitrate'], \
        "`metric` flag must be either `miou` or `hitrate`"

    np.random.seed(args.seed)

    create_pct_diff_df(args.metric, args.pred_bootstrap_results,
                       args.hb_bootstrap_results, args.save_dir)
