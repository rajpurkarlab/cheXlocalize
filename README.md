# CheXlocalize

This repository contains the code used to generate segmentations from saliency method heat maps and human annotations, and to evaluate the localization performance of those segmentations, as described in the paper _Benchmarking saliency methods for chest X-ray interpretation_. [TODO: add link]

You may run the scripts in this repo using your own heat maps/annotations/segmentations, or you may run them on the CheXlocalize dataset. TODO: ADD LINK.

### Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Download data](#download)
- [Generate segmentations from saliency method heatmaps](#heatmap_to_segm)
- [Generate segmentations from human annotations](#ann_to_segm)
- [Evaluate localization performance](#eval)
- [Citation](#citation)

<a name="overview"></a>
## Overview

While deep learning has enabled automated medical imaging interpretation at a level shown to surpass that of practicing experts, the "black box" nature of neural networks represents a barrier to physicians’ trust and model adoption in the clinical setting. Therefore, to encourage the development and validation of more "interpretable" models for chest X-ray interpretation, we present a new radiologist-annotated segmentation dataset.

CheXlocalize (TODO: ADD LINK) is a radiologist-annotated segmentation dataset on chest X-rays. The dataset consists of two types of radiologist annotations for the localization of 10 pathologies: pixel-level segmentations and most-representative points. Annotations were drawn on images from the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) validation and test sets. The dataset also consists of two separate sets of radiologist annotations: (1) ground-truth pixel-level segmentations on the validation and test sets, drawn by two board-certified radiologists, and (2) benchmark pixel-level segmentations and most-representative points on the test set, drawn by a separate group of three board-certified radiologists.

![overview](/img/overview.png)

The validation and test sets consist of 234 chest X-rays from 200 patients and 668 chest X-rays from 500 patients, respectively. The 10 pathologies of interest were Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Lung Lesion, Lung Opacity, Pleural Effusion, Pneumothorax, and Support Devices.

For more details, please see our paper, [_Benchmarking saliency methods for chest X-ray interpretation_](https://www.medrxiv.org/content/10.1101/2021.02.28.21252634v3).

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

You may run the scripts in this repo using your own predicted and ground-truth segmentations, or you may run them on the CheXlocalize dataset.

If you'd like to use the CheXlocalize dataset, download (1) the validation set Grad-CAM heat maps, (2) the validation set ground-truth raw radiologist annotations, (3) the validation set ground-truth pixel-level segmentations here: (TODO: ADD LINK).

If you'd like to use your own heatmaps, annotations, and/or segmentations, see the relevant sections below for the expected data formatting.

<a name="heatmap_to_segm"></a>
## Generate segmentations from saliency method heatmaps

To generate binary segmentations from saliency method heat maps, run:

```
(chexlocalize) > python heatmap_to_segmentation.py --map_dir <map_dir> --output_path <output_path>
```

`<map_dir>` is the directory with pickle files containing the heat maps. The script extracts the heat maps from the pickle files.

If you downloaded the CheXlocalize dataset, then these pickle files are in `/chexlocalize_dataset/gradcam_heatmaps_val/`. Each CXR has a pickle file associated with each of the ten pathologies, so that each pickle file contains information for a single CXR and pathology in the following format:

```
{
# DenseNet121 + Grad-CAM heat map <torch.Tensor> of shape (1, 1, h, w) #TODO: is this right?
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


`<output_path>` is the json file path used for saving the encoded segmentation masks. The json file is formatted such that it can be used as input to `eval.py` (see [_Evaluate localization performance_](#eval) for formatting details).

To store the binary segmentations efficiently, we use RLE format, and the encoding is implemented using [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools). If an image has no saliency segmentations, we store a mask of all zeros.

Running this script on the validation set heat maps from the CheXlocalize dataset should take about 10 minutes.

<a name="ann_to_segm"></a>
## Generate segmentations from human annotations

To generate binary segmentations from raw human annotations, run:

```
(chexlocalize) > python annotation_to_segmentation.py --ann_path <ann_path> --output_path <output_path>
```

`<ann_path>` is the json file path with raw human annotations.

If you downloaded the CheXlocalize dataset, then this is the json file `/chexlocalize_dataset/gt_annotations_val.json`. Each key of the json file is a single CXR id with its data formatted as follows:

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

This input json should include only those CXRs with at least one positive ground-truth label, and each CXR in the json should include only those pathologies for which its ground-truth label is positive.

If using your own human annotations, please be sure to save them in a json using the above formatting.

`<output_path>` is the json file path used for saving the encoded segmentation masks. The json file is formatted such that it can be used as input to `eval.py` (see [_Evaluate localization performance_](#eval) for formatting details).

Running this script on the validation set heat maps from the CheXlocalize dataset should take about 5 minutes.

<a name="eval"></a>
## Evaluate localization performance

We use two evaluation metrics to compare segmentations:
- **mIoU**: mean Intersection over Union is a stricter metric that measures how much, on average, the predicted segmentations overlap with the ground-truth segmentations.
- **hit rate**: hit rate is a less strict metric that does not require the localization method to locate the full extent of a pathology. Hit rate is based on the pointing game setup, in which credit is given if the most representative point identified by the localization method lies within the ground-truth segmentation. A "hit" indicates that the correct region of the CXR was located regardless of the exact bounds of the binary segmentations. Localization performance is then calculated as the hit rate across the dataset.

![metrics](/img/metrics.png)
> Left: CXR with ground-truth and saliency method annotations for Pleural Effusion. The segmentations have a low overlap (IoU is 0.078), but pointing game is a "hit" since the saliency method's most representative point is inside of the ground-truth segmentation. Right, CXR with ground-truth and human benchmark annotations for Enlarged Cardiomediastinum. The segmentations have a high overlap (IoU is 0.682), but pointing game is a "miss" since saliency method's most representative point is outside of the ground-truth segmentation.

For more details on mIoU and hit rate, please see our paper, [_Benchmarking saliency methods for chest X-ray interpretation_](https://www.medrxiv.org/content/10.1101/2021.02.28.21252634v3).

To run evaluation, use the following command:

```
(chexlocalize) > python eval.py [FLAGS]
```

**Required flags**:
`--metric`: options are 'miou' or 'hitrate'
`--gt_path`: Directory where ground-truth segmentations are saved (encoded). This could be the json output of `annotation_to_segmentation.py`. Or, if you downloaded the CheXlocalize dataset, then this is the json file `/chexlocalize_dataset/gt_segmentations_val.json`.
`--pred_path`: If `metric = miou`, then this should be the directory where predicted segmentations are saved (encoded). This could be the json output of `heatmap_to_segmentation.py`, or, if you downloaded the CheXlocalize dataset, then this could be the json file TODO. If `metric = hitrate`, then this should be directory with pickle files containing heat maps (the script extracts the most representative point from the pickle files). If you downloaded the CheXlocalize dataset, then these pickle files are in `/chexlocalize_dataset/gradcam_heatmaps_val/`.

**Optional flags:**
`--true_pos_only`: Default is `True`. If `True`, run evaluation only on the true positive slice of the dataset (CXRs that contain both predicted and ground-truth segmentations).
`--save_dir`: Default is `./`. Where to save evaluation results.
`--seed`: Default is `0`. Random seed to fix for bootstrapping.

Both `gt_path` and `pred_path` must be json files where each key is a single CXR id with its data formatted as follows:

```
{
    'patient64622_study1_view1_frontal': {
	    'Enlarged Cardiomediastinum': {
		'size': [2320, 2828], # (h, w)
		'counts': '`Vej1Y2iU2c0B?F9G7I6J5K6J6J6J6J6H8G9G9J6L4L4L4L4L3M3M3M3L4L4L4L4K6K4L4L4L4L4M3M3L4M3M3M3L4M3M3L4M3M3M3M3M3M2N3M3M3M3M3M3M3M3M3M3L4M3L3N3M2M4M3L3N3M2N3N1N3N2M2O2M3N1O2M3N1N3N2M3N1N3N2N2N1O2N2N1O2N2N2N1O2N2N2N1N3N2N1O2M3N1O2N1O2M3N1O2N1N3N1O2N2M2O2N1O2N1O2N2N1O2N1O2N1O2N1O2N1O2N1O1O2N1O2N1O2N1O2N1O2N1O2O0O2N1O2N1O1O2N1O2N1O2N101N1O2N1O2N1XNi_OV[NY`0ad1Q@\\[NP`0\\d1[@`[Ng?Vd1d@h[N\\?Rd1m@j[NU?Rd1QAj[NP'},
	    ....
	    'Support Devices': {
		'size': [2320, 2828], # (h, w)
		'counts': 'Xid[1R1ZW29G8H9G9H9F:G9G9G7I7H8I7I6K4L5K4L5K4L4L5K4L5J5L5K4L4L4L4L4L4L3M4M3L4M3M3L3N3M3L4M3M2M4M3M3L4M3M2N3L4M3M3L3N3M3L4M3M3L3N2N2M3N2'}
    },
    ...
    'patient64652_study1_view1_frontal': {
	...
    }
}
```

Both `pred_path` (if `metric = miou`) and `gt_path` json files must contain a key for all CXR ids (regardless of whether it has any positive ground-truth labels), and each CXR id key must have values for all ten pathologies (regardless of ground-truth label). In other words, all CXRs and images are indexed. If a CXR has no segmentations, we store a segmentation mask of all zeros. If using your own `pred_path` and `gt_path` json files as input to this script, be sure that they are formatted per the above, with segmentation masks encoded using RLE using [pycocotools](https://github.com/
cocodataset/cocoapi/tree/master/PythonAPI/pycocotools).

This evaluation script generates three csv files:
`{miou/hitrate}_results.csv`: IoU or hit/miss results for each CXR and each pathology.
`{miou/hitrate}_bootstrap_results.csv`: 1000 bootstrap samples of mIoU or hit rate for each pathology.
`{miou/hitrate}_summary_results.csv`: mIoU or hit rate 95% bootstrap confidence intervals for each pathology.

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
