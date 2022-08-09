# CheXlocalize

This repository contains the code used to generate segmentations from saliency method heatmaps and human annotations, and to evaluate the localization performance of those segmentations, as described in the paper [_Benchmarking saliency methods for chest X-ray interpretation_](https://www.medrxiv.org/content/10.1101/2021.02.28.21252634v4.full.pdf).

You may run the scripts in this repo using your own heatmaps/annotations/segmentations, or you may run them on the [CheXlocalize dataset](https://stanfordaimi.azurewebsites.net/datasets/abfb76e5-70d5-4315-badc-c94dd82e3d6d).

### Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Download data](#download)
- [Exploratory data analysis](#eda)
- [Generate segmentations from saliency method heatmaps](#heatmap_to_segm)
	- [Fine-tune segmentation thresholds](#seg_threshold)
	- [Fine-tune probability thresholds](#prob_threshold)
- [Generate segmentations from human annotations](#ann_to_segm)
- [Evaluate localization performance](#eval)
	- [Calculate percentage decrease](#pct_dec)
- [Compute pathology features](#path_features)
	- [Plot distribution of pathology features](#dist_path_features)
- [Run regressions on pathology features](#regression_pathology)
- [Run regressions on model assurance](#regression_model_assurance)
- [Get precision, recall/sensitivity, and specificity values](#precision_recall)
- [Bugs or questions?](#bugs)
- [Citation](#citation)


<a name="overview"></a>
## Overview

While deep learning has enabled automated medical imaging interpretation at a level shown to surpass that of practicing experts, the "black box" nature of neural networks represents a major barrier to clinical trust and adoption. Therefore, to encourage the development and validation of more "interpretable" models for chest X-ray interpretation, we present a new radiologist-annotated segmentation dataset.

[CheXlocalize](https://stanfordaimi.azurewebsites.net/datasets/abfb76e5-70d5-4315-badc-c94dd82e3d6d) is a radiologist-annotated segmentation dataset on chest X-rays. The dataset consists of two types of radiologist annotations for the localization of 10 pathologies: pixel-level segmentations and most-representative points. Annotations were drawn on images from the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) validation and test sets. The dataset also consists of two separate sets of radiologist annotations: (1) ground-truth pixel-level segmentations on the validation and test sets, drawn by two board-certified radiologists, and (2) benchmark pixel-level segmentations and most-representative points on the test set, drawn by a separate group of three board-certified radiologists.

![overview](/img/overview.png)

The validation and test sets consist of 234 chest X-rays from 200 patients and 668 chest X-rays from 500 patients, respectively. The 10 pathologies of interest are Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Lung Lesion, Lung Opacity, Pleural Effusion, Pneumothorax, and Support Devices.

For more details, please see our paper, [_Benchmarking saliency methods for chest X-ray interpretation_](https://www.medrxiv.org/content/10.1101/2021.02.28.21252634v4).

<a name="setup"></a>
## Setup

The code should be run using Python 3.8.3. If using conda, run:
```
> conda create -n chexlocalize python=3.8.3
> conda activate chexlocalize
(chexlocalize) >
```

Install all dependency packages using the following command:
```
(chexlocalize) > pip install -r requirements.txt
```

<a name="download"></a>
## Download data

You may run the scripts in this repo using your own heatmaps/annotations/segmentations, or you may run them on the CheXlocalize dataset.

Download the CheXlocalize dataset [here](https://stanfordaimi.azurewebsites.net/datasets/abfb76e5-70d5-4315-badc-c94dd82e3d6d). You'll find:

* `CheXpert-v1.0/valid/`: validation set CXR images
* `CheXpert-v1.0/valid.csv`: validation set ground-truth labels
* `gradcam_maps_val/`: validation set DenseNet121 + Grad-CAM heatmaps
* `gradcam_segmentations_val.json`: validation set DenseNet121 + Grad-CAM pixel-level segmentations
* `gt_annotations_val.json`: validation set ground-truth raw radiologist annotations
* `gt_segmentations_val.json`: validation set ground-truth pixel-level segmentations

We have also included a small sample of the above data in this repo in [`./sample`](https://github.com/rajpurkarlab/cheXlocalize/tree/master/sample).

If you'd like to use your own heatmaps/annotations/segmentations, see the relevant sections below for the expected data formatting.

<a name="eda"></a>
## Exploratory data analysis
Given a json file with segmentations, for each pathology, to count the number of CXRs with at least one segmentation, run:

```
(chexlocalize) > python count_segs.py [FLAGS]
```

**Required flags**
* `--seg_path`: the json file path where segmentations are saved (encoded). [_Evaluate localization performance_](#eval) describes how the segmentation json should be formated.

**Optional flags**
* `--save_dir`: the directory to save the csv file `n_segs.csv` with results. Default is current directory.

<a name="heatmap_to_segm"></a>
## Generate segmentations from saliency method heatmaps

To generate binary segmentations from saliency method heatmaps, run:

```
(chexlocalize) > python heatmap_to_segmentation.py [FLAGS]
```

**Required flags**
* `--map_dir`: the directory with pickle files containing the heatmaps. The script extracts the heatmaps from the pickle files.

If you downloaded the CheXlocalize dataset, then these pickle files are in `/cheXlocalize_dataset/gradcam_maps_val/`. Each CXR has a pickle file associated with each of the ten pathologies, so that each pickle file contains information for a single CXR and pathology in the following format:

```
{
# DenseNet121 + Grad-CAM heatmap <torch.Tensor> of shape (1, 1, h, w)
'map': tensor([[[[1.4711e-06, 1.4711e-06, 1.4711e-06,  ..., 5.7636e-06, 5.7636e-06, 5.7636e-06],
           	 [1.4711e-06, 1.4711e-06, 1.4711e-06,  ..., 5.7636e-06, 5.7636e-06, 5.7636e-06],
           	 ...,
    		 [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 7.9709e-05, 7.9709e-05, 7.9709e-05],
           	 [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 7.9709e-05, 7.9709e-05, 7.9709e-05]]]]),

# model probability (float)
'prob': 0.02029409697279334,

# one of the ten possible pathologies (string)
'task': Consolidation,

# 0 if ground-truth label for 'task' is negative, 1 if positive (int)
'gt': 0,

# original cxr image
'cxr_img': tensor([[[0.7490, 0.7412, 0.7490,  ..., 0.8196, 0.8196, 0.8118],
  		    [0.6627, 0.6627, 0.6706,  ..., 0.7373, 0.7137, 0.6941],
          	    [0.5137, 0.5176, 0.5294,  ..., 0.6000, 0.5686, 0.5255],
          	    ...,
          	    [0.7294, 0.7725, 0.7804,  ..., 0.2941, 0.2549, 0.2078],
          	    [0.7804, 0.8157, 0.8157,  ..., 0.3216, 0.2824, 0.2510],
          	    [0.8353, 0.8431, 0.8549,  ..., 0.3725, 0.3412, 0.3137]],
          	    ...
         	   [[0.7490, 0.7412, 0.7490,  ..., 0.8196, 0.8196, 0.8118],
          	    [0.6627, 0.6627, 0.6706,  ..., 0.7373, 0.7137, 0.6941],
          	    [0.5137, 0.5176, 0.5294,  ..., 0.6000, 0.5686, 0.5255],
          	    ...,
          	    [0.7294, 0.7725, 0.7804,  ..., 0.2941, 0.2549, 0.2078],
          	    [0.7804, 0.8157, 0.8157,  ..., 0.3216, 0.2824, 0.2510],
          	    [0.8353, 0.8431, 0.8549,  ..., 0.3725, 0.3412, 0.3137]]]),

# dimensions of original cxr (w, h)
'cxr_dims': (2022, 1751)
}
```

If using your own saliency maps, please be sure to save them as pickle files using the above formatting.

**Optional flags**
* `--threshold_path`: an optional csv file path that you can pass in to use your own thresholds to binarize the heatmaps. As an example, we provide [`./sample/tuning_results.csv`](https://github.com/rajpurkarlab/cheXlocalize/blob/master/sample/tuning_results.csv), which contains the threshold for each pathology that maximizes mIoU on the validation set. When passing in your own csv file, make sure to follow the same formatting as this example csv. By default, no threshold path is passed in, in which case we will apply Otsu's method (an automatic global thresholding algorithm provided by the cv2 package).
* `--probability_threshold_path`: an optional csv file path that you can pass in to reduce false positive segmentation prediction using probability cutoffs. If the predicted model probability is below the cutoff, we generate a segmentation mask of all zeros regardless of thresholding scheme. As an example, we provide [`./sample/probability_tuning_results.csv`](https://github.com/rajpurkarlab/cheXlocalize/blob/master/sample/probability_tuning_results.csv). When passing in your own csv file, make sure to follow the same formatting as this example csv. See [_Fine-tune probability thresholds_](#prob_threshold) for more context.
* `--output_path`: the json file path used for saving the encoded segmentation masks. The json file is formatted such that it can be used as input to `eval.py` (see [_Evaluate localization performance_](#eval) for formatting details). Default is `./saliency_segmentations.json`.
* `--if_smoothing`: Set this flag to `True` to smooth the pixelated heatmaps using box filtering. Default is `False`.
* `--k`: the kernel size used for box filtering (`int`). Default is 0. The user-defined `k` must be >= 0. If you set `k` as any number > 0, make sure to set `if_smoothing` to `True`, otherwise no smoothing would be performed.

To store the binary segmentations efficiently, we use RLE format, and the encoding is implemented using [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools). If an image has no saliency segmentations, we store a mask of all zeros.

Running this script on the validation set heatmaps from the CheXlocalize dataset should take about 10 minutes.

<a name="seg_threshold"></a>
### Fine-tune segmentation thresholds
To find the thresholds that maximize mIoU for each pathology on the validation set, run:

```
(chexlocalize) > python tune_heatmap_threshold.py [FLAGS]
```

**Required flags**
* `--map_dir`: the directory with pickle files containing the heatmaps.
* `--gt_path`: the json file where ground-truth segmentations are saved (encoded).

**Optional flags**
* `--save_dir`: the directory to save the csv file that stores the tuned thresholds. Default is current directory.

This script will replicate `./sample/tuning_results.csv` when you use the CheXlocalize validation set DenseNet121 + Grad-CAM heatmaps in `/cheXlocalize_dataset/gradcam_maps_val/` as `<map_dir>` and the validation set ground-truth pixel-level segmentations in `/cheXlocalize_dataset/gt_segmentations_val.json` as `<gt_path>`. Running this script should take about one hour.

<a name="prob_threshold"></a>
### Fine-tune probability thresholds
We noticed that for many CXRs false positive saliency segmentations were generated even though model probability was low. To ensure that the saliency segmentation is consistent with model probability output, we apply a logic such that the segmentation mask is all zeros if the predicted probability was below a chosen cutoff. To choose these cutoffs for each pathology, we maximize the mIoU on the validation set.

To create the csv file with these cutoffs (which should be used with the flag `--probability_threshold_path` for `heatmap_to_segmentation.py`), run:

```
(chexlocalize) > python tune_probability_threshold.py [FLAGS]
```

**Required flags**
* `--map_dir`: the directory with pickle files containing the heatmaps.
* `--gt_path`: the json file where ground-truth segmentations are saved (encoded).

**Optional flags**
* `--save_dir`: the directory to save the csv file that stores the tuned thresholds. Default is current directory.

In [our paper](https://www.medrxiv.org/content/10.1101/2021.02.28.21252634v4), we use these cutoffs to report results in Table 3 and Extended Data Fig. 4 because when evaluating localization performance using the full dataset, we found that many false postive saliency segmentations were generated even though model probability was low. When evaluation only uses the true positive slice of the dataset, we recommend using `<threshold_path>` to find the best thresholds since we're not accounting for false positives.

This script will replicate `./sample/probability_tuning_results.csv` when you use the CheXlocalize validation set DenseNet121 + Grad-CAM heatmaps in `/cheXlocalize_dataset/gradcam_maps_val/` as `<map_dir>` and the validation set ground-truth pixel-level segmentations in `/cheXlocalize_dataset/gt_segmentations_val.json` as `<gt_path>`. Running this script should take about one hour.

<a name="ann_to_segm"></a>
## Generate segmentations from human annotations

To generate binary segmentations from raw human annotations, run:

```
(chexlocalize) > python annotation_to_segmentation.py [FLAGS]
```

**Required flags**
* `--ann_path`: the json file path with raw human annotations. If you downloaded the CheXlocalize dataset, then this is the json file `/cheXlocalize_dataset/gt_annotations_val.json`. Each key of the json file is a single CXR id with its data formatted as follows:

```
{
    'patient64622_study1_view1_frontal': {
        'img_size': [2320, 2828], # (h, w)
	'Support Devices': [[[1310.68749, 194.47059],
   		    	     [1300.45214, 194.47059],
   			     [1290.21691, 201.29412],
			     ...
			     [1310.68749, 191.05883],
			     [1300.45214, 197.88236],
			     [1293.62865, 211.52943]]],
 	'Cardiomegaly': [[[1031.58047, 951.35314],
   			  [1023.92373, 957.09569],
   			  [1012.43856, 964.75249],
			  ...
			  [1818.31313, 960.92406],
   			  [1804.91384, 955.1815],
   			  [1789.60024, 951.35314]]],
	...
    },
    'patient64542_study1_view2_lateral': {
        ...
    }
}
```

Each pathology key (e.g. `json_dict['patient64622_study1_view1_frontal']['Support Devices']`) is associated with a nested list of contours and coordinates: `[[coordinates for contour 1], [coordinates for contour 2]]`. The number of contours corresponds to the number of segmentations on a CXR for a given pathology. For example, the below CXR has two segmentations (and therefore would have two contours) for Atelectasis.

<img src="img/example_two_segmentations.png" alt="example CXR with two segmentations" width="350"/>

Each contour holds a list of [X,Y] coordinates that contour the shape of the pathology.

This input json should include only those CXRs with at least one positive ground-truth label, and each CXR in the json should include only those pathologies for which its ground-truth label is positive. (So `/cheXlocalize_dataset/gt_annotations_val.json` will have keys for 187 subjects with 643 total annotations.)

If using your own human annotations, please be sure to save them in a json using the above formatting.

**Optional flags**
* `--output_path`: the json file path used for saving the encoded segmentation masks. The json file is formatted such that it can be used as input to `eval.py` (see [_Evaluate localization performance_](#eval) for formatting details). Default is `./human_segmentations.json`.

Running this script on the validation set heatmaps from the CheXlocalize dataset should take about 5 minutes.

<a name="eval"></a>
## Evaluate localization performance

We use two evaluation metrics to compare segmentations:
- **mIoU**: mean Intersection over Union is a stricter metric that measures how much, on average, the predicted segmentations overlap with the ground-truth segmentations.
- **hit rate**: hit rate is a less strict metric that does not require the localization method to locate the full extent of a pathology. Hit rate is based on the pointing game setup, in which credit is given if the most representative point identified by the localization method lies within the ground-truth segmentation. A "hit" indicates that the correct region of the CXR was located regardless of the exact bounds of the binary segmentations. Localization performance is then calculated as the hit rate across the dataset.

![metrics](/img/metrics.png)
> Left: CXR with ground-truth and saliency method annotations for Pleural Effusion. The segmentations have a low overlap (IoU is 0.078), but pointing game is a "hit" since the saliency method's most representative point is inside of the ground-truth segmentation. Right, CXR with ground-truth and human benchmark annotations for Enlarged Cardiomediastinum. The segmentations have a high overlap (IoU is 0.682), but pointing game is a "miss" since saliency method's most representative point is outside of the ground-truth segmentation.

For more details on mIoU and hit rate, please see our paper, [_Benchmarking saliency methods for chest X-ray interpretation_](https://www.medrxiv.org/content/10.1101/2021.02.28.21252634v4).

To run evaluation, use the following command:

```
(chexlocalize) > python eval.py [FLAGS]
```

**Required flags**
* `--metric`: options are `iou` or `hitmiss`
* `--gt_path`: Path to file where ground-truth segmentations are saved (encoded). This could be the json output of `annotation_to_segmentation.py`. Or, if you downloaded the CheXlocalize dataset, then this is the json file `/cheXlocalize_dataset/gt_segmentations_val.json`.
* `--pred_path`:
	* If `metric = iou`: this should be the path to file where predicted segmentations are saved (encoded). This could be the json output of `heatmap_to_segmentation.py`, or `annotation_to_segmentation.py`, or, if you downloaded the CheXlocalize dataset, then this could be the json file `/cheXlocalize_dataset/gradcam_segmentations_val.json`.
	* If `metric = hitmiss`: when evaluating saliency methods, then this should be directory with pickle files containing heatmaps (the script extracts the most representative point from the pickle files). If you downloaded the CheXlocalize dataset, then these pickle files are in `/cheXlocalize_dataset/gradcam_maps_val/`. However, when evaluating human benchmark annotations, then this should be a json file that holds the human annotations for most representative points.

**Optional flags**
* `--true_pos_only`: Default is `True`. If `True`, run evaluation only on the true positive slice of the dataset (CXRs that contain both predicted and ground-truth segmentations). If `False`, also include CXRs with a predicted segmentation but without a ground-truth segmentation, and include CXRs with a ground-truth segmentation but without a predicted segmentation.
* `--save_dir`: Where to save evaluation results. Default is current directory.
* `--if_human_benchmark`: If `True`, run evaluation on human benchmark performance. Default is set to `False`. This is especially important when evaluating hit rate performance, since the most representative point input for saliency method is formatted differently than the most representative point input for human benchmark.
* `--seed`: Default is `0`. Random seed to fix for bootstrapping.

If `metric = iou`, both `pred_path` and `gt_path` must be json files where each key is a single CXR id with its data formatted as follows:

```
{
    'patient64622_study1_view1_frontal': {
	    'Enlarged Cardiomediastinum': {
		'size': [2320, 2828], # (h, w)
		'counts': '`Vej1Y2iU2c0B?F9G7I6J5K6J6J6J6J6H8G9G9J6L4L4L4L4L3M3M3M3L4L4...'},
	    ....
	    'Support Devices': {
		'size': [2320, 2828], # (h, w)
		'counts': 'Xid[1R1ZW29G8H9G9H9F:G9G9G7I7H8I7I6K4L5K4L5K4L4L5K4L5J5L5K...'}
    },
    ...
    'patient64652_study1_view1_frontal': {
	...
    }
}
```

Both `pred_path` (if `metric = iou`) and `gt_path` json files must contain a key for all CXR ids (regardless of whether it has any positive ground-truth labels), and each CXR id key must have values for all ten pathologies (regardless of ground-truth label). In other words, all CXRs and images are indexed. If a CXR has no segmentations, we store a segmentation mask of all zeros. If using your own `pred_path` and `gt_path` json files as input to this script, be sure that they are formatted per the above, with segmentation masks encoded using RLE using [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools).

This evaluation script generates three csv files:
* `{iou/hitmiss}_results_per_cxr.csv`: IoU or hit/miss results for each CXR and each pathology.
* `{iou/hitmiss}_bootstrap_results.csv`: 1000 bootstrap samples of IoU or hit/miss for each pathology.
* `{iou/hitmiss}_summary_results.csv`: mIoU or hit rate 95% bootstrap confidence intervals for each pathology.

<a name="pct_dec"></a>
### Calculate percentage decrease

To calculate the percentage decrease from the human benchmark localization metric to the saliency method pipeline localization metric, run:

```
(chexlocalize) > python calculate_percentage_decrease.py [FLAGS]
```

**Required flags**
* `--metric`: options are `miou` or `hitrate`
* `--hb_bootstrap_results`: Path to csv file with 1000 bootstrap samples of human benchmark IoU or hit/miss for each pathology. This is the output of `eval.py` called `{iou/hitmiss}_humanbenchmark_bootstrap_results_per_cxr.csv`.
* `--pred_bootstrap_results`: Path to csv file with 1000 bootstrap samples of saliency method IoU or hit/miss for each pathology. This is the output of `eval.py` called `{iou/hitmiss}_bootstrap_results_per_cxr.csv`.

**Optional flags**
* `--save_dir`: Where to save results as csv files. Default is current directory.

<a name="path_features"></a>
## Compute pathology features
We define four pathological features: (1) number of instances (for example, bilateral Pleural Effusion would have two instances, whereas there is only one instance for Cardiomegaly), (2) size (pathology area with respect to the area of the whole CXR), (3) elongation and (4) irrectangularity (the last two features measure the complexity of the pathology shape and were calculated by fitting a rectangle of minimum area enclosing the binary mask).

To compute the four pathology features, run:

```
(chexlocalize) > python compute_pathology_features.py [FLAGS]
```

**Required flags**
* `--gt_ann`: Path to json file with raw ground-truth annotations. If you downloaded the CheXlocalize dataset, then this is the json file `/cheXlocalize_dataset/gt_annotations_val.json`.
* `--gt_seg`: Path to json file with ground-truth segmentations (encoded). This could be the json output of `annotation_to_segmentation.py`. Or, if you downloaded the CheXlocalize dataset, then this is the json file `/cheXlocalize_dataset/gt_segmentations_val.json`.

**Optional flags**
* `--save_dir`: Where to save four pathology feature dataframes as csv files. Default is current directory.

Note that we use the ground-truth annotations to extract the number of instances, and we use the ground-truth segmentation masks to calculate area, elongation and rectangularity. We chose to extract number of instances from annotations because sometimes radiologists draw two instances for a pathology that are overlapping; in this case, the number of annotations would be 2, but the number of segmentations would be 1.

Running this script on the validation set annotations and segmentations from the CheXlocalize dataset should take about 5 minutes.

<a name="dist_path_features"></a>
### Plot distribution of pathology features

To plot the distribution of the four pathology features across all 10 pathologies, run:

```
(chexlocalize) > python plot_pathology_features.py [FLAGS]
```

**Required flags**
* `--features_dir`: Path to directory that holds four csv files: `area_ratio.csv`, `elongation.csv`, `num_instances.csv`, and `rec_area_ratio.csv`. These four files are the output of `compute_pathology_features.py`.

**Optional flags**
* `--save_dir`: Where to save plots. Default is current directory.

<a name="regression_pathology"></a>
## Run regressions on pathology features
We provide a script to run a simple linear regression with the evaluation metric (IoU or hit/miss) as the dependent variable (to understand the relationship between the geometric features of a pathology and saliency method localization performance). Each regression uses one of the above four geometric features as a single independent variable.

```
(chexlocalize) > python regression_pathology_features.py [FLAGS]
```

**Required flags**
* `--features_dir`: Path to directory that holds four csv files: `area_ratio.csv`, `elongation.csv`, `num_instances.csv`, and `rec_area_ratio.csv`. These four files are the output of `compute_pathology_features.py`.
* `--pred_iou_results`: path to csv file with saliency method IoU results for each CXR and each pathology. This is the output of `eval.py` called `iou_results_per_cxr.csv`.
* `--pred_hitmiss_results`: path to csv file with saliency method hit/miss results for each CXR and each pathology. This is the output of `eval.py` called `hitmiss_results_per_cxr.csv`.

**Optional flags**
* `--evalute_hb`: Default is `False`. If true, evaluate human benchmark in addition to saliency method. If `True`, the flags `hb_iou_results` and `hb_hitmiss_results` (below) are also required. If `True`, additional regressions will be run using the difference between the evaluation metrics of the saliency method pipeline and the human benchmark as the dependent variable (to understand the relationship between the geometric features of a pathology and the gap in localization performance between the saliency method pipeline and the human benchmark).
* `--hb_iou_results`: Path to csv file with human benchmark IoU results for each CXR and each pathology. This is the output of `eval.py` called `miou_humanbenchmark_results_per_cxr.csv`.
* `--hb_hitmiss_results`: Path to csv file with human benchmark hit/miss results for each CXR and each pathology. This is the output of `eval.py` called `hitmiss_humanbenchmark_results_per_cxr.csv`.
* `--save_dir`: Where to save regression results. Default is current directory. If `evaluate_hb` is `True`, four files will be saved: `regression_features_pred_iou.csv`, `regression_features_pred_hitmiss.csv`, `regression_features_iou_diff.csv`, `regression_features_hitmiss_diff.csv`. If `evaluate_hb` is `False`, only two files will be saved: `regression_features_pred_iou.csv`, `regression_features_pred_hitmiss.csv`.

In [our paper](https://www.medrxiv.org/content/10.1101/2021.02.28.21252634v4), only the true positive slice was included in each regression (see Table 2). Each feature is normalized using min-max normalization and the regression coefficient can be interpreted as the effect of that geometric feature on the evaluation metric at hand. The regression results report the 95% confidence interval and the Bonferroni corrected p-values. For confidence intervals and p-values, we use the standard calculation for linear models.

<a name="regression_model_assurance"></a>
## Run regressions on model assurance
We provide a script to run a simple linear regression for each pathology using the model’s probability output as the single independent variable and using the predicted evaluation metric (IoU or hit/miss) as the dependent variable. The script also runs a simple regression that uses the same approach as above, but that includes all 10 pathologies.

```
(chexlocalize) > python regression_model_assurance.py [FLAGS]
```

**Required flags**
* `--map_dir`: the directory with pickle files containing the heatmaps. The script extracts the heatmaps from the pickle files.
* `--pred_iou_results`: path to csv file with saliency method IoU results for each CXR and each pathology. This is the output of `eval.py` called `iou_results_per_cxr.csv`.
* `--pred_hitmiss_results`: path to csv file with saliency method hit/miss results for each CXR and each pathology. This is the output of `eval.py` called `hitmiss_results_per_cxr.csv`.

**Optional flags**
* `--save_dir`: Where to save regression results. Default is current directory.

Note that in [our paper](https://www.medrxiv.org/content/10.1101/2021.02.28.21252634v4), for each of the 11 regressions, we use the _full_ dataset since the analysis of false positives and false negatives was also of interest (see Table 3). In addition to the linear regression coefficients, the regression results also report the Spearman correlation coefficients to capture any potential non-linear associations.

<a name="precision_recall"></a>
## Get precision, recall/sensitivity, and specificity values

To get the precision, recall/sensitivity, and specificity values of the saliency method pipeline and the human benchmark segmentations, run:

```
(chexlocalize) > python precision_recall_specificity.py [FLAGS]
```

**Required flags**
* `--gt_path`: the json file path where ground-truth segmentations are saved (encoded).
* `--pred_seg_path`: the json file path where saliency method segmentations are saved (encoded). This could be the json output of `heatmap_to_segmentation.py`, or, if you downloaded the CheXlocalize dataset, then this could be the json file `/cheXlocalize_dataset/gradcam_segmentations_val.json`.
* `--hb_seg_path`: the json file path where human benchmark segmentations are saved (encoded). This could be the json output of `annotation_to_segmentation.py`.

**Optional flags**
* `--save_dir`: the directory to save the csv file that stores the precision, recall/sensitivity, and specificity values. Default is current directory.

We treat each pixel in the saliency method pipeline and the human benchmark segmentations as a classification, use each pixel in the ground-truth segmentation as the ground-truth label, and calculate precision, recall/sensitivity, and specificity over all CXRs for each pathology.

<a name="bugs"></a>
## Bugs or questions?
Feel free to email Adriel Saporta (adriel@nyu.edu), or to open an issue in this repo, with any questions related to the code or our paper.

<a name="citation"></a>
## Citation

If you are using the CheXlocalize dataset, or are using our code in your research, please cite our paper:

```
@article {Saporta2021.02.28.21252634,
	author = {Saporta, Adriel and Gui, Xiaotong and Agrawal, Ashwin and Pareek, Anuj and Truong, Steven QH and Nguyen, Chanh DT and Ngo, Van-Doan and Seekins, Jayne and Blankenberg, Francis G. and Ng, Andrew Y. and Lungren, Matthew P. and Rajpurkar, Pranav},
	title = {Benchmarking saliency methods for chest X-ray interpretation},
	elocation-id = {2021.02.28.21252634},
	year = {2021},
	doi = {10.1101/2021.02.28.21252634},
	URL = {https://doi.org/10.1101/2021.02.28.21252634}
}
```
