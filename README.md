![LOGO](/img/CheXplanation.svg)

TODO: update logo above, and make sure this section makes sense after you go through the code

This repository contains the code used to generate segmentations from saliency method heatmaps and to evaluate the localization performance of those segmentations on the CheXlocalize dataset, as described in the paper _Benchmarking saliency methods for chest X-ray interpretation_. [TODO: add link]

To download the validation dataset or view and submit to the leaderboard, visit the [CheXplanation website](https://stanfordmlgroup.github.io/competitions/chexplanation/).


Typical install time: ~5 minutes
Expected run time for demo: ~20 minutes

### Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Download segmentations](#download)
- [Evaluate segmentations](#eval)
- [Generate segmentations from saliency method heatmaps](#heatmap_to_segm)
- [Generate segmentations from human annotations](#ann_to_segm)
- [Evaluation of localization performance](#eval)
- [Citing](#citing)

---

<a name="overview"></a>
## Overview
TODO: ADD

<a name="setup"></a>
## Setup

The code should be run using Python 3.8.3 If using conda, run:
```
> conda create -n chexlocalize python=3.8.3
> conda activate
(chexlocalize) >
```

Install all dependency packages using the following command:
```
(chexlocalize) > pip install -r requirements.txt
```

<a name="download"></a>
## Download segmentations

You may run the below scripts using your own predicted and ground-truth segmentations, or you may run them on the CheXlocalize dataset.

If you'd like to use the CheXlocalize dataset, download the validation set ground-truth pixel-level segmentations here (TODO: ADD LINK).

If you'd like to use your own predicted and ground-truth segmentations, they will need to be in the following format (TODO: ADD FORMAT INSTRUCTIONS).

## Generate Segmentations from Saliency Heatmaps
We provided the code to generate binary segmentations from saliency heatmaps (read about the thresholding scheme in the manuscript). To store the binary segmentations efficiently, we used RLE format and the encoding is implemented using the toolbox provided in COCO detection challenge, [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools).

We also made public the Grad-CAM heatmaps that was run on the DenseNet121 Ensemble model for the validation set (Read our paper for model details). The Grad-CAM heatmaps are stored in the .pkl files under ./GradCAM_maps_val/. We also included a sample of the heatmaps in ./GradCAM_maps_val_sample for fast demo (we included three images). Each image id in the validation set has a pickel (.pkl) file associated with each of the ten pathologies. The .pkl files store model probability, saliency map and original image dimensions. 

In the output segmentation json file, we format it such that all images and pathologies are indexed. If an image has no saliency segmentations, we stored a mask of all zeros.

```
python segmentation/heatmap_to_segmentation.py [OPTIONS]

Options:
--saliency_path 	Where saliency maps are stored (saliency heatmaps are extracted from the pickle files)
--output_file_name  Name and path of the output json file that stores the encoded segmentation masks
```

<a name="ann_to_segm"></a>

## Generate Segmentations from Annotations
We also released the code to generate segmentation masks from annotations (a list of X-Y coordinates that contours the shape of the pathology.)

The input annotation json file only includes images and pathologies that have a ground truth annotation. In the output segmentation json file, we index all images and all pathologies. If an image has no saliency segmentations, we store a segmentation mask of all zeros.

```
python segmentation/annotation_to_segmentation.py [OPTIONS]

Options:
--ann_path  Where the annotation json file is stored (represented as a list of coordiantes)
--output_file_name  Name and path of the output json file that stores the encoded segmentation masks
```
y is reported on the true positive slice of the
174 dataset (CXRs that contain both saliency method and human benchmark segmentations
175 when the ground-truth label of the pathology is positive).

To run evaluation using hit rate, use the following command:


To evaluate localization performance using your own predicted and ground-truth segmentations, must be in certain format? TODO: ask what encoded json is?
Mention that they should be encode, the masks should be encoded binary masks using RLE using pycocotools https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py


Our evaluation script generates the two summary metrics (mIoU and hit rate) for each localization task, as well as the corresponding 95% bootstrap confidence interval (n_boostrap_sample = 1000). 

### Usage

```
Usage: python eval_miou.py [OPTIONS]

Options:
    --phase      	Use validation or test data.
    --save_dir 		Path to which the summary csv is stored.
```

```
Usage: python eval_ptgame.py [OPTIONS]

Options:
    --phase      	Use validation or test data.
    --save_dir 		Path to which the summary csv is stored.
```

<a name="segm"></a>

## Generate binary segmentations from saliency method heatmaps
As a reference, we provide the code used generate binary segmentations from saliency method heat maps using a thresholding scheme. The technical details can be found in the Method section of our paper manuscript. To save the binary segmentations efficiently, we used RLE format for storage and the encoding is implemented using the toolbox provided in COCO detection challenge, [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools).


```
python segmentation/pred_segmentation.py [OPTIONS]

Options:
--phase			Use validation or test data
--method   		The saliency methods used
--model     		Single or ensemble
--if_threshold 		Whether the thresholding scheme is adopted.
--save_dir 		Path where the RLE-formatted segmentations are stored.
```

<a name="eval"></a>
## Evaluate segmentations

We use two evaluation metrics to compare segmentations (TODO: add Fig. 1b):
- **mIoU**: mean Intersection over Union is a stricter metric that measures how much, on average, the predicted segmentations overlap with the ground-truth segmentations.
- **hit rate**: hit rate is a less strict metric that does not require the localization method to locate the full extent of a pathology. Hit rate is based on the pointing game setup, in which credit is given if the most representative point identified by the localization method lies within the ground-truth segmentation. A "hit" indicates that the correct region of the CXR was located regardless of the exact bounds of the binary segmentations. Localization performance is then calculated as the hit rate across the dataset.

To evaluate localization performance using your own predicted and ground-truth segmentations

To run evaluation using mIoU, use the following command:

<a name="heatmap_to_segm"></a>



<a name="citing"></a>

## Citing
TODO: UPDATE THIS
If you are using the CheXphoto dataset, please cite this paper:

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
