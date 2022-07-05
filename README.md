![LOGO](/img/CheXplanation.svg)

TODO: UPDATE ABOVE LOGO
TODO: UPDATE THIS SECTION BELOW

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

For more details, please see our paper, [Benchmarking saliency methods for chest X-ray interpretation](https://www.medrxiv.org/content/10.1101/2021.02.28.21252634v3).

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
'map': tensor([[[[1.4711e-06, 1.4711e-06, 1.4711e-06,  ..., 5.7636e-06, 5.7636e-06, 5.7636e-06],
           	 [1.4711e-06, 1.4711e-06, 1.4711e-06,  ..., 5.7636e-06, 5.7636e-06, 5.7636e-06],
           	 [1.4711e-06, 1.4711e-06, 1.4711e-06,  ..., 5.7636e-06, 5.7636e-06, 5.7636e-06],
           	 ...,
           	 [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 7.9709e-05, 7.9709e-05, 7.9709e-05],
    		 [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 7.9709e-05, 7.9709e-05, 7.9709e-05],
           	 [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 7.9709e-05, 7.9709e-05, 7.9709e-05]]]]), # DenseNet121 + Grad-CAM heat map <torch.Tensor> of shape (1, 1, h, w) #TODO: is this right?
'prob': 0.02029409697279334, # model probability (float)
'task': Consolidation, # one of the ten possible pathologies (string)
'gt': 0, # 0 if ground-truth label for 'task' is negative, 1 if positive
'cxr_img': tensor([[[0.7490, 0.7412, 0.7490,  ..., 0.8196, 0.8196, 0.8118],
  		    [0.6627, 0.6627, 0.6706,  ..., 0.7373, 0.7137, 0.6941],
          	    [0.5137, 0.5176, 0.5294,  ..., 0.6000, 0.5686, 0.5255],
          	    ...,
          	    [0.7294, 0.7725, 0.7804,  ..., 0.2941, 0.2549, 0.2078],
          	    [0.7804, 0.8157, 0.8157,  ..., 0.3216, 0.2824, 0.2510],
          	    [0.8353, 0.8431, 0.8549,  ..., 0.3725, 0.3412, 0.3137]],

         	   [[0.7490, 0.7412, 0.7490,  ..., 0.8196, 0.8196, 0.8118],
          	    [0.6627, 0.6627, 0.6706,  ..., 0.7373, 0.7137, 0.6941],
          	    [0.5137, 0.5176, 0.5294,  ..., 0.6000, 0.5686, 0.5255],
          	    ...,
          	    [0.7294, 0.7725, 0.7804,  ..., 0.2941, 0.2549, 0.2078],
          	    [0.7804, 0.8157, 0.8157,  ..., 0.3216, 0.2824, 0.2510],
          	    [0.8353, 0.8431, 0.8549,  ..., 0.3725, 0.3412, 0.3137]],

         	   [[0.7490, 0.7412, 0.7490,  ..., 0.8196, 0.8196, 0.8118],
          	    [0.6627, 0.6627, 0.6706,  ..., 0.7373, 0.7137, 0.6941],
          	    [0.5137, 0.5176, 0.5294,  ..., 0.6000, 0.5686, 0.5255],
          	    ...,
          	    [0.7294, 0.7725, 0.7804,  ..., 0.2941, 0.2549, 0.2078],
          	    [0.7804, 0.8157, 0.8157,  ..., 0.3216, 0.2824, 0.2510],
          	    [0.8353, 0.8431, 0.8549,  ..., 0.3725, 0.3412, 0.3137]]]), # original cxr image
'cxr_dims': (2022, 1751) # dimensions of original cxr (h, w)
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
        'img_size': [2320, 2828], # dimensions of original CXR (h, w)
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

![example CXR with two segmentations](/img/example_two_segmentations.png)

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
*Left: CXR with ground-truth and saliency method annotations for Pleural Effusion. The segmentations have a low overlap (IoU is 0.078), but pointing game is a "hit" since the saliency method's most representative point is inside of the ground-truth segmentation. Right, CXR with ground-truth and human benchmark annotations for Enlarged Cardiomediastinum. The segmentations have a high overlap (IoU is 0.682), but pointing game is a "miss" since saliency method's most representative point is outside of the ground-truth segmentation.*

To evaluate localization performance using your own predicted and ground-truth segmentations

To run evaluation using mIoU, use the following command:

y is reported on the true positive slice of the
174 dataset (CXRs that contain both saliency method and human benchmark segmentations
175 when the ground-truth label of the pathology is positive).

In the output segmentation json file, we index all images and all pathologies. If an image has no saliency segmentations, we store a segmentation mask of all zeros.

The json file is formatted such that all images and pathologies are indexed.
To run evaluation using hit rate, use the following command:


To evaluate localization performance using your own predicted and ground-truth segmentations, must be in certain format? TODO: ask what encoded json is?
Mention that they should be encode, the masks should be encoded binary masks using RLE using pycocotools https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py


Our evaluation script generates the two summary metrics (mIoU and hit rate) for each localization task, as well as the corresponding 95% bootstrap confidence interval (n_boostrap_sample = 1000). 

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
