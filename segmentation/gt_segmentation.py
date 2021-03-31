"""" Create segmentation for ground truth annotations 
    Input: {valid/test/vietnam}_annotations_merged.json
    Output: 1. pngs of the segmentation 2. encoded mask in json format
    
    Usage: python3 gt_segmentation.py --dataset test --save_dir target_dir
"""
from pycocotools import mask
from eval_constants import LOCALIZATION_TASKS
from PIL import Image
import argparse
from eval_helper import create_mask, segmentation_to_mask
import json
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gt_to_mask(gt_file, output_file):
    """
    Create segmentation based on radiologist-labeled contour polygons 
    """
    print(f"Read ground truth labels from {gt_file}")
    with open(gt_file) as f:
        gt = json.load(f)

    print(f"Create segmentation and encode")
    results = {}
    for img_id in tqdm(gt.keys()):

        if img_id not in results:
            results[img_id] = {}
        for task in LOCALIZATION_TASKS:
            # create segmentation
            polygons = gt[img_id][task] if task in gt[img_id] else []
            img_dims = gt[img_id]['img_size']
            segm_map = create_mask(polygons, img_dims)  # np array

            # encode to coco mask
            encoded_map = segmentation_to_mask(segm_map)
            results[img_id][task] = encoded_map

    assert len(results.keys()) == len(gt.keys())

    print(f"Write results to json at {output_file}")
    # write results to json file
    with open(output_file, "w") as outfile:
        json.dump(results, outfile)


def gt_to_png(gt_path, png_path):
    """
    Create segmentation based on radiologist ground truth and save them to pngs

    Args: 
        gt_path: path to the encoded json file
    """

    with open(gt_path) as f:
        gt = json.load(f)

    for idx, img_id in tqdm(enumerate(sorted(gt.keys()))):
        assert len(gt[img_id].keys()) == 10

        if idx % 100 == 0:
            print(f'Processing the {idx}th x-ray image')

        for task in gt[img_id].keys():
            gt_item = gt[img_id][task]
            gt_mask = mask.decode(gt_item)
            im = Image.fromarray((gt_mask*255).astype('uint8'))
            im.save(f"{png_path}/{img_id}_{task}.png")


if __name__ == "__main__":
    path_group = '/deep/group/aihc-bootcamp-spring2020/localize'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='valid',
                        help="valid, test or vietnam")
    parser.add_argument(
        '--save_dir', default=f'{path_group}/annotations', help="path to save segmentation masks")
    args = parser.parse_args()

    dataset = args.dataset
    save_dir = args.save_dir
    gt_encoded_file = f'{save_dir}/{dataset}_gt_segmentations_encoded.json'
    gt_file = f'{path_group}/annotations/{dataset}_annotations_merged.json'
    gt_to_mask(gt_file, gt_encoded_file)  # save as RLEs
