""" Script that computes geometric features """
import glob
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask
import statsmodels.api as sm
import statsmodels.formula.api as smf
from eval_helper import iou_seg

group_dir = '/deep/group/aihc-bootcamp-spring2020/localize'
output_path = f'{group_dir}/eval_results/regression'

test_labels = pd.read_csv('/deep/group/anujpare/CheXpert_original_test500/test.csv')
test_labels['img_id'] = test_labels.Path.map(lambda x: '_'.join(x.split('/')[2:]).replace('.jpg','')).tolist()
all_ids = test_labels['img_id'].tolist()

# function that parses the ground truth file and extracts image feature and stores them in a dictionary\
phase = 'test'
og_gt_dir = f'{group_dir}/annotations/{phase}_annotations_merged.json'
with open(og_gt_dir) as f:
    og_gt = json.load(f)
    
gt_encoded_dir = f'{group_dir}/annotations/{phase}_encoded.json'
with open(gt_encoded_dir) as f:
    gt_segm = json.load(f)
    
# number of instances per image
all_instances = {}
pos_patients = sorted(gt_segm.keys())
for task in sorted(tasks):
    n_instance = []
    for img_id in pos_patients:
        n = len(og_gt[img_id][task]) if task in og_gt[img_id] else 0
        n_instance.append(n)
    all_instances[task] = n_instance
    
instance_df = pd.DataFrame(all_instances)
instance_df['img_id'] = pos_patients


# area ratio per image 
all_areas = {}
for task in sorted(tasks):
    areas = []
    for img_id in pos_patients:
        gt_item = gt_segm[img_id][task]
        gt_mask = mask.decode(gt_item)
        area_ratio = np.sum(gt_mask)/(gt_mask.shape[0]*gt_mask.shape[1])
        areas.append(area_ratio)
    all_areas[task] = areas
    
areas_df = pd.DataFrame(all_areas)
areas_df['img_id'] = pos_patients

# append the negative images
for img_id in all_ids:
    if img_id not in pos_patients:
        row = {}
        for task in tasks:
            row[task] = 0
        row['img_id'] = img_id
        instance_df = instance_df.append(row, ignore_index = True)
        areas_df = areas_df.append(row, ignore_index = True)
        
        
instance_df = instance_df.sort_values(by = ['img_id'])
areas_df = areas_df.sort_values(by = ['img_id'])

instance_df.to_csv(f'{output_path}/num_instances_test.csv',index = False)
areas_df.to_csv(f'{output_path}/area_ratio_test.csv',index = False)



# get vietnam iou
phase = 'test'
group_dir = '/deep/group/aihc-bootcamp-spring2020/localize'
gt_dir = f'{group_dir}/annotations/{phase}_encoded.json'
pred_dir = f'{group_dir}/annotations/vietnam_encoded.json'

ious = {}
with open(gt_dir) as f:
    gt = json.load(f)
        
with open(pred_dir) as f:
    pred = json.load(f)
    
for task in tasks:
    print(f'Evaluating {task}')
    ious[task] = []
   
    for img_id in all_ids:
        
        if img_id not in pred and img_id not in gt:
            iou_score = -1
       
        elif img_id not in pred and img_id in gt:
            iou_score = 0
    
        # get predicted segmentation mask
        elif img_id in pred and img_id not in gt:   
            if task in pred[img_id]:
                pred_item = pred[img_id][task]
                pred_mask = mask.decode(pred_item)
                gt_mask = np.zeros(pred_item['size'])
                assert gt_mask.shape == pred_mask.shape 
                iou_score = iou_seg(pred_mask, gt_mask)
            else:
                iou_score = 0
        else:
            pred_item = pred[img_id][task]
            pred_mask = mask.decode(pred_item)
            
            gt_item = gt[img_id][task]
            gt_mask = mask.decode(gt_item)
            
            assert gt_mask.shape == pred_mask.shape 
            iou_score = iou_seg(pred_mask, gt_mask)
            
        ious[task].append(iou_score)
        
    if np.all(ious[task]==-1):
        print(f'{task} has all true negatives')

vietnam_df = pd.DataFrame.from_dict(ious)
vietnam_df.to_csv((f'{vietnam_dir}/test_human_ioua_all.csv'))