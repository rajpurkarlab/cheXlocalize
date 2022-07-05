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

If you'd like to use the CheXlocalize dataset, download (1) the validation set Grad-CAM heat maps, (2) the validation set ground-truth raw radiologist annotations, (3) the validation set ground-truth pixel-level segmentations here: (TODO: ADD LINK).If you'd like to use your own predicted and ground-truth segmentations, they will need to be in the following format (TODO: ADD FORMAT INSTRUCTIONS).

If you'd like to use your own heatmaps, annotations, and/or segmentations, see the relevant sections below for the expected data formatting.

<a name="heatmap_to_segm"></a>
## Generate segmentations from saliency method heatmaps

To generate binary segmentations from saliency method heat maps, run:

```
(chexlocalize) > python heatmap_to_segmentation.py --map_dir <map_dir> --output_path <output_path>
```

`<map_dir>` is the directory with pickle files containing the heat maps. TODO: info on the format needed here.

TODO: EXAMPLE OF JSON FILE FORMAT AND GRADCAN EXAMPKE
saliency heatmaps are extracted from the pickle files)

We also made public the Grad-CAM heatmaps that was run on the DenseNet121 Ensemble model for the validation set (Read our paper for model details). The Grad-CAM heatmaps are stored in the .pkl files under ./GradCAM_maps_val/. We also included a sample of the heatmaps in ./GradCAM_maps_val_sample for fast demo (we included three images). Each image id in the validation set has a pickel (.pkl) file associated with each of the ten pathologies. The .pkl files store model probability, saliency map and original image dimensions. 

`<output_path>` is the json file path used for saving the encoded segmentation masks. To store the binary segmentations efficiently, we use RLE format, and the encoding is implemented using [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools). The json file is formatted such that all images and pathologies are indexed.

TODO: INCLUDE EXAMPLE OF JSON FILE FORMAT

If an image has no saliency segmentations, we stored a mask of all zeros.

<a name="ann_to_segm"></a>
## Generate segmentations from human annotations

To generate binary segmentations from raw human annotations, run:

```
(chexlocalize) > python annotation_to_segmentation.py --ann_path <ann_path> --output_path <output_path>
```

`<ann_path>` is the json file path with raw human annotations. TODO INFO ON FORMATTING FOR CHEXLOCALIZE AND OTHERWISE.

represented as a list of coordiantes)

We also released the code to generate segmentation masks from annotations (a list of X-Y coordinates that contours the shape of the pathology.)

The input annotation json file only includes images and pathologies that have a ground truth annotation. In the output segmentation json file, we index all images and all pathologies. If an image has no saliency segmentations, we store a segmentation mask of all zeros.

<a name="eval"></a>
## Evaluate localization performance

We use two evaluation metrics to compare segmentations (TODO: add Fig. 1b):
- **mIoU**: mean Intersection over Union is a stricter metric that measures how much, on average, the predicted segmentations overlap with the ground-truth segmentations.
- **hit rate**: hit rate is a less strict metric that does not require the localization method to locate the full extent of a pathology. Hit rate is based on the pointing game setup, in which credit is given if the most representative point identified by the localization method lies within the ground-truth segmentation. A "hit" indicates that the correct region of the CXR was located regardless of the exact bounds of the binary segmentations. Localization performance is then calculated as the hit rate across the dataset.

To evaluate localization performance using your own predicted and ground-truth segmentations

To run evaluation using mIoU, use the following command:

y is reported on the true positive slice of the
174 dataset (CXRs that contain both saliency method and human benchmark segmentations
175 when the ground-truth label of the pathology is positive).

To run evaluation using hit rate, use the following command:


To evaluate localization performance using your own predicted and ground-truth segmentations, must be in certain format? TODO: ask what encoded json is?
Mention that they should be encode, the masks should be encoded binary masks using RLE using pycocotools https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py


Our evaluation script generates the two summary metrics (mIoU and hit rate) for each localization task, as well as the corresponding 95% bootstrap confidence interval (n_boostrap_sample = 1000). 

<a name="citation"></a>

## Citation
TODO: UPDATE THIS

If you are using the CheXlocalize dataset, or are using our code in your research, please cite our paper:

```
@article {Saporta2021.02.28.21252634,
	author = {Saporta, Adriel and Gui, Xiaotong and Agrawal, Ashwin and Pareek, Anuj and Truong, Steven QH and Nguyen, Chanh DT and Ngo, Van-Doan and Seekins, Jayne and Blankenberg, Francis G. and Ng, Andrew Y. and Lungren, Matthew P. and Rajpurkar, Pranav},
	title = {Benchmarking saliency methods for chest X-ray interpretation},
	elocation-id = {2021.02.28.21252634},
	year = {2021},
	doi = {10.1101/2021.02.28.21252634},
	publisher = {Cold Spring Harbor Laboratory Press},
	abstract = {Saliency methods, which {\textquotedblleft}explain{\textquotedblright} deep neural networks by producing heat maps that highlight the areas of the medical image that influence model prediction, are often presented to clinicians as an aid in diagnostic decision-making. Although many saliency methods have been proposed for medical imaging interpretation, rigorous investigation of the accuracy and reliability of these strategies is necessary before they are integrated into the clinical setting. In this work, we quantitatively evaluate three saliency methods (Grad-CAM, Grad-CAM++, and Integrated Gradients) across multiple neural network architectures using two evaluation metrics. We establish the first human benchmark for chest X-ray interpretation in a multilabel classification set up, and examine under what clinical conditions saliency maps might be more prone to failure in localizing important pathologies compared to a human expert benchmark. We find that (i) while Grad-CAM generally localized pathologies better than the two other saliency methods, all three performed significantly worse compared with the human benchmark; (ii) the gap in localization performance between Grad-CAM and the human benchmark was largest for pathologies that had multiple instances, were smaller in size, and had shapes that were more complex; (iii) model confidence was positively correlated with Grad-CAM localization performance. Our work demonstrates that several important limitations of saliency methods must be addressed before we can rely on them for deep learning explainability in medical imaging.Competing Interest StatementThe authors have declared no competing interest.Funding StatementN/AAuthor DeclarationsI confirm all relevant ethical guidelines have been followed, and any necessary IRB and/or ethics committee approvals have been obtained.YesThe details of the IRB/oversight body that provided approval or exemption for the research described are given below:The project did not involve human subjects researchI confirm that all necessary patient/participant consent has been obtained and the appropriate institutional forms have been archived, and that any patient/participant/sample identifiers included were not known to anyone (e.g., hospital staff, patients or participants themselves) outside the research group so cannot be used to identify individuals.YesI understand that all clinical trials and any other prospective interventional studies must be registered with an ICMJE-approved registry, such as ClinicalTrials.gov. I confirm that any such study reported in the manuscript has been registered and the trial registration ID is provided (note: if posting a prospective study registered retrospectively, please provide a statement in the trial ID field explaining why the study was not registered in advance).YesI have followed all appropriate research reporting guidelines and uploaded the relevant EQUATOR Network research reporting checklist(s) and other pertinent material as supplementary files, if applicable.YesCheXpert data is available at https://stanfordmlgroup.github.io/competitions/chexpert/. The validation set and corresponding benchmark radiologist annotations will be available online for the purpose of extending the study. https://stanfordmlgroup.github.io/competitions/chexpert/},
	URL = {https://www.medrxiv.org/content/early/2021/10/08/2021.02.28.21252634},
	eprint = {https://www.medrxiv.org/content/early/2021/10/08/2021.02.28.21252634.full.pdf},
	journal = {medRxiv}
}
```
