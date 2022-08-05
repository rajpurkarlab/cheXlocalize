"""
Plot the distribution of pathology features.
"""
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from eval_constants import LOCALIZATION_TASKS


def plot(args):
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

    overall_df = pd.DataFrame()
    for task in sorted(LOCALIZATION_TASKS):
        df = pd.DataFrame()
        data = {'n_instance': instance_df_pos[task].values,
                'area_ratio':area_df_pos[task].values,
                'elongation': elongation_df_pos[task].values,
                'irrectangularity': irrectangularity_df_pos[task].values,
                'task': task}
        df = pd.DataFrame(data)
        # get cxrs with ground-truth segmentation for this task
        df = df[df.n_instance>0]
        overall_df = pd.concat([overall_df, df])

    sns.set_style("whitegrid")
    features = ['n_instance', 'area_ratio', 'elongation', 'irrectangularity']
    features_labels = ['Number of Instances', 'Area Ratio', 'Elongation',
                       'Irrectangularity']
    for feature, feature_label in zip(features, features_labels):
        task_labels = sorted(LOCALIZATION_TASKS)
        task_labels[5] = 'E. Cardiom.'
        plt.figure(figsize=(12,6))
        g1 = sns.boxenplot(x='task', y=feature, data=overall_df,
                           palette=sns.color_palette("husl",10))
        g1.set_xticklabels(task_labels, fontsize=14, rotation=60, ha="right",
                           rotation_mode="anchor")
        plt.xlabel('')
        plt.ylabel(feature_label,fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{args.save_dir}/{feature}_dist.png',dpi=300)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--features_dir', type=str,
                        help='directory with four csv files: area_ratio.csv, \
                              elongation.csv, num_instances.csv`, and \
                              rec_area_ratio.csv.')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='directory where plots will be saved')
    args = parser.parse_args()

    plot(args)
