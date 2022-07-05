"""" Create segmentation masks from annotations and encode segmentations in RLE formats using the pycocotools Mask API. The final output is stored in a json file. 
    Input: annotations of the pathologies (represented as a list of coordiantes) in json format 
            (DEFAULT) ../cheXlozalize_dataset/gt_annotations_val.json
    Output: encoded segmentation mask in json format
            (DEFAULT) gt_segmentations_val.json
    
    Usage: python3 segmentation/annotation_to_segmentation.py --ann_path ../cheXlozalize_dataset/gt_annotations_val.json --segm_path gt_segmentations_val.json
"""
import os
import sys
# add parent directory to path so we can import eval_constants directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_constants import LOCALIZATION_TASKS
from eval_helper import encode_segmentation
from pycocotools import mask
from PIL import Image, ImageDraw
import numpy as np
import argparse
import json
from tqdm import tqdm


def create_mask(polygons, img_dims):
    """
    Creates a binary mask (of the original matrix   size) given a list of polygons (annotations format)

    Args:
        poly_coords (list): [[[x11,y11],[x12,y12],...[x1n,y1n]],...]

    Returns:
        mask (np.array): binary mask, 1 where the pixel is predicted to be the pathology 0 otherwise
    """
    poly = Image.new('1', (img_dims[1], img_dims[0]))
    for polygon in polygons:
        coords = [(point[0], point[1]) for point in polygon]
        ImageDraw.Draw(poly).polygon(coords,  outline=1, fill=1)

    binary_mask = np.array(poly, dtype="int")
    return binary_mask


def ann_to_mask(input_path, output_path):
    """
    Create binary segmentation from annotations (polygons that's a list of [X,Y] coordinates) and store the segmentations in RLE format in a json file
    Args:
        input_path (string): annotation file 
        output_path (string): file where the encoded segmentation masks are stored 
    """
    print(f"Read annotations from {input_path}")
    with open(input_path) as f:
        ann = json.load(f)

    print(f"Create segmentation and encode")
    results = {}
    for img_id in tqdm(ann.keys()):

        if img_id not in results:
            results[img_id] = {}
        for task in LOCALIZATION_TASKS:
            # create segmentation
            polygons = ann[img_id][task] if task in ann[img_id] else []
            img_dims = ann[img_id]['img_size']
            segm_map = create_mask(polygons, img_dims)  # np array

            # encode to coco mask
            encoded_map = encode_segmentation(segm_map)
            results[img_id][task] = encoded_map

    assert len(results.keys()) == len(ann.keys())

    # save segmentations to json file
    print(f"Segmentation masks saved at {output_path}")
    with open(output_path, "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', default='../cheXlozalize_dataset/gt_annotations_val.json',
                        help="Path to the annotations")
    parser.add_argument(
        '--segm_path', default=f'gt_segmentations_val.json', help="path to save segmentation masks")
    args = parser.parse_args()

    annotation_path = args.ann_path
    segmentation_path = args.segm_path
    ann_to_mask(annotation_path, segmentation_path)
