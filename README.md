![LOGO](/img/CheXplanation.svg)

This the repo referenced in the paper, "Deep learning saliency maps do not accurately highlight diagnostically relevant regions for medical image interpretation". We provided the source code used for initial data preprocessing, generating segmentations from saliency maps and evaluating localization. To download the validation dataset or view and submit to the leaderboard, visit the [CheXplanation website](https://stanfordmlgroup.github.io/competitions/chexplanation/) (coming up soon). 

### Table of Contents

- [Prerequisites](#prereqs)
- [Generate Segmentations from Saliency Heatmap](#segm)
- [Evaluation of Localization](#eval)
- [License](#license)
- [Citing](#citing)

---

<a name="prereqs"></a>

## Prerequisites

The code should be run using Python 3.7.6.

Before starting, please install the repo Python requirements using the following command:
```
pip install -r requirements.txt
```

<a name="segm"></a>

## Generate Segmentations from Saliency Heatmap
We provided the code to generate binary segmentations from saliency heatmaps using a thresholding scheme. The technical details can be found in the Method section of our paper manuscript. To save the binary segmentations efficiently, we used RLE format for storage and the encoding is implemented using the toolbox provided in COCO detection challenge, [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools).

### Usage 

We explored with multiple saliency methods such as Grad-CAM and Intergrated Gradients. Although our paper only highlighted Grad-CAM, the code works on multiple methods

```
python segmentation/pred_segmentation.py [OPTIONS]

Options:
--phase			Use validation or test data
--method   		The saliency methods used
--model     		Single or ensemble
--if_threshold 		Whether the thresholding scheme is adopted.
--save_dir 		Path where the RLE-formatted segmentations are stored.
```

<a name="synthetic"></a>

## Evaluating Localization using Semantic Segmentation Scheme.

Our evaluation script generated the summary metrics (mIoU) per localization task and the 95% bootstrap confidence interval.

### Usage

```
Usage: python eval.py [OPTIONS]

Options:
    --phase      	Use validation or test data.
    --pred_path 	Path to which the segmentation json file is stored.
    --save_dir 		Path to which the summary csv is stored.
```


<a name="license"></a>

## License

This repository is made publicly available under the MIT License.

<a name="citing"></a>

## Citing

If you are using the CheXphoto dataset, please cite this paper:

```
@article {Saporta2021.02.28.21252634,
	author = {Saporta, Adriel and Gui, Xiaotong and Agrawal, Ashwin and Pareek, Anuj and Truong, Steven QH and Nguyen, Chanh DT and Ngo, Van-Doan and Seekins, Jayne and Blankenberg, Francis G. and Ng, Andrew Y. and Lungren, Matthew P. and Rajpurkar, Pranav},
	title = {Deep learning saliency maps do not accurately highlight diagnostically relevant regions for medical image interpretation},
	elocation-id = {2021.02.28.21252634},
	year = {2021},
	doi = {10.1101/2021.02.28.21252634},
	publisher = {Cold Spring Harbor Laboratory Press},
	abstract = {Deep learning has enabled automated medical image interpretation at a level often surpassing that of practicing medical experts. However, many clinical practices have cited a lack of model interpretability as reason to delay the use of {\textquotedblleft}black-box{\textquotedblright} deep neural networks in clinical workflows. Saliency maps, which {\textquotedblleft}explain{\textquotedblright} a model{\textquoteright}s decision by producing heat maps that highlight the areas of the medical image that influence model prediction, are often presented to clinicians as an aid in diagnostic decision-making. In this work, we demonstrate that the most commonly used saliency map generating method, Grad-CAM, results in low performance for 10 pathologies on chest X-rays. We examined under what clinical conditions saliency maps might be more dangerous to use compared to human experts, and found that Grad-CAM performs worse for pathologies that had multiple instances, were smaller in size, and had shapes that were more complex. Moreover, we showed that model confidence was positively correlated with Grad-CAM localization performance, suggesting that saliency maps were safer for clinicians to use as a decision aid when the model had made a positive prediction with high confidence. Our work demonstrates that several important limitations of interpretability techniques for medical imaging must be addressed before use in clinical workflows.Competing Interest StatementThe authors have declared no competing interest.Funding StatementN/AAuthor DeclarationsI confirm all relevant ethical guidelines have been followed, and any necessary IRB and/or ethics committee approvals have been obtained.YesThe details of the IRB/oversight body that provided approval or exemption for the research described are given below:The project did not involve human subjects researchAll necessary patient/participant consent has been obtained and the appropriate institutional forms have been archived.YesI understand that all clinical trials and any other prospective interventional studies must be registered with an ICMJE-approved registry, such as ClinicalTrials.gov. I confirm that any such study reported in the manuscript has been registered and the trial registration ID is provided (note: if posting a prospective study registered retrospectively, please provide a statement in the trial ID field explaining why the study was not registered in advance).YesI have followed all appropriate research reporting guidelines and uploaded the relevant EQUATOR Network research reporting checklist(s) and other pertinent material as supplementary files, if applicable.YesCheXpert data is available at https://stanfordmlgroup.github.io/competitions/chexpert/. The validation set and corresponding benchmark radiologist annotations will be available online for the purpose of extending the study. https://stanfordmlgroup.github.io/competitions/chexpert/},
	URL = {https://www.medrxiv.org/content/early/2021/03/02/2021.02.28.21252634},
	eprint = {https://www.medrxiv.org/content/early/2021/03/02/2021.02.28.21252634.full.pdf},
	journal = {medRxiv}
}
```

 
