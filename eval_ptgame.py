""" Given saliency map output and radiologist ground truth, compute the hit rate with confidence interval per task.
    
    Usage: python3 eval_ptgame.py --phase val --save_dir /path/to/save/results
"""
import json
import pickle
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw
from eval_constants import *
from eval_miou import *
from tqdm import tqdm

def create_map(pkl_path):
    """
    Create saliency map of original img size 
    """
    info = pickle.load(open(pkl_path,'rb'))
    saliency_map = info['map']
    img_dims = info['cxr_dims']
    map_resized = F.interpolate(saliency_map, size=(img_dims[1],img_dims[0]), mode='bilinear', align_corners=False)
    saliency_map = map_resized.squeeze().squeeze().detach().cpu().numpy()
    return saliency_map, img_dims


def calculate_hit(gt_path, map_dir):
    """
    Calculate hit rate 
    """
    with open(gt_path) as f:
        gt = json.load(f)

    all_paths = sorted(list(Path(map_dir).rglob("*_map.pkl")))

    results = {}

    for pkl_path in tqdm(all_paths):

        # break down path to image name and task
        path = str(pkl_path).split('/')
        task = path[-1].split('_')[-2]
        img_id = '_'.join(path[-1].split('_')[:-2])

        if task not in LOCALIZATION_TASKS:
            print(f"Invalid task {task}")
            continue

        if img_id in results:
            if task in results[img_id]:
                print(f'Check for duplicates for {task} for {img_id}')
                break
            else:
                results[img_id][task] = 0
        else:
            # get ground truth binary mask
            if img_id not in gt:
                continue
            else:
                results[img_id] = {}
                results[img_id][task] = 0


        gt_item = gt[img_id][task]
        gt_mask = mask.decode(gt_item)

        # get saliency heatmap
        sal_map, img_dims = create_map(pkl_path)

        x =  np.unravel_index(np.argmax(sal_map, axis = None), sal_map.shape)[0]
        y = np.unravel_index(np.argmax(sal_map, axis = None), sal_map.shape) [1]
        
        assert (gt_mask.shape == sal_map.shape)
        if(gt_mask[x][y]==1):
            results[img_id][task] = 1

        elif (np.sum(gt_mask)==0):
            results[img_id][task] = np.nan
    
    all_ids = sorted(gt.keys())
    return all_ids, results

def evaluate(gt_dir, pred_dir,save_dir,table_name):
    all_ids, results = calculate_hit(gt_dir, pred_dir)
    metrics = pd.DataFrame.from_dict(results,orient='index') 
    bs_df = bootstrap_metric(metrics, 1000, metric = "pt_game")
    bs_df.to_csv(f'{save_dir}/{table_name}_bs_hit.csv',index = False)
    metrics['img_id'] = all_ids
    metrics.to_csv(f'{save_dir}/{table_name}_hit.csv',index = False)
    records = []
    for task in bs_df.columns:
        records.append(create_ci_record(bs_df[task], task))
    summary_df = pd.DataFrame.from_records(records)
    print(summary_df)
    summary_df.to_csv(f'{save_dir}/{table_name}_summary_ptgame.csv',index = False)
    print(f"Evaluation result saved at {save_dir}/{table_name}_summary_ptgame.csv")
    


if __name__ == '__main__':
    
    parser = ArgumentParser()
    
    parser.add_argument('--phase', type=str, default="test",
                        help='valid(val) or test')
    parser.add_argument('--method', type=str, required=True,
                        help='localization method: gradcam')
    parser.add_argument('--model', default='densenet',
                        help='densenet, inception, resnet')
    parser.add_argument('--save_dir', default=".",
                        help='directory where the evaluation result will be saved')
    
    args = parser.parse_args()
    
    method = args.method
    phase = args.phase
    model = args.model
    save_dir = args.save_dir
    
    pred_path = f'{update_results_dir}/{model}_{method}_{phase}/ensemble_results/cams'
    gt_path = f'{group_dir}/annotations/{phase}_encoded.json'
    np.random.seed(0)
    table_name = f'{phase}_{method}_{model}_ensemble'
    evaluate(gt_path, pred_path, save_dir,table_name)
    
    