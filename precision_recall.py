from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

# from eval import *
# from pred_segmentation import *
from eval_constants import LOCALIZATION_TASKS
from heatmap_to_segmentation import pkl_to_mask


"""
def pkl_to_map(pkl_path, task):
    """
    # load cam pickle file, get saliency map and resize. 
    Convert to np array
    
    Args:
        pkl_path(str): path to the pickle file
        task(str): pathology
    """
    info = pickle.load(open(pkl_path,'rb'))
    saliency_map = info['map']
    img_dims = info['cxr_dims']
    map_resized = F.interpolate(saliency_map, size=(img_dims[1],img_dims[0]), mode='bilinear', align_corners=False)
    segm_map = cam_to_segmentation(map_resized)
    segm_map = np.array(segm_map,dtype = "int")
    return segm_map
"""
def get_pred_results(args):
    """
    TODO: add comment
    """
    with open(args.gt_path) as f:
        gt_dict = json.load(f)
    pred_paths = sorted(list(Path(map_dir).rglob('*_map.pkl')))

    results = {}
    for pkl_path in tqdm(pred_paths):
        task, img_id = parse_pkl_filename(pkl_path)
        if task not in LOCALIZATION_TASKS:
            continue
        if img_id not in gt_dict:
            continue

        # get encoded segmentation mask
        segm = pkl_to_mask(pkl_path)
        gt_item = gt_dict[img_id][task]
        gt_mask = mask.decode(gt_item)

        TP = np.sum(gt_mask == segm)
        FP = np.sum(np.logical_and(segm == 1, gt_mask == 0))
        FN = np.sum(np.logical_and(segm == 0, gt_mask == 1))

        # append to big numpy array
        if task in results:
            results[task]['tp'] += TP
            results[task]['fp'] += FP
            results[task]['fn'] += FN
        else:
            results[task] = {}
            results[task]['tp'] = TP
            results[task]['fp'] = FP
            results[task]['fn'] = FN

    return results


def get_hb():
    """
    input ref or hb
    """
    group_dir = '/deep/group/aihc-bootcamp-spring2020/localize'
    gt_dir = f'{group_dir}/annotations/test_encoded.json'
    hb_dir = f'{group_dir}/annotations/vietnam_encoded.json'

    with open(gt_dir) as f:
        gt = json.load(f)
    with open(hb_dir) as f:
        hb = json.load(f)


    results = {}

    all_ids = sorted(gt.keys())
    tasks = sorted(LOCALIZATION_TASKS)

    for task in tqdm(tasks):
        for img_id in all_ids:     
            # get ground_truth segmentation mask
            if img_id not in gt:
                continue
            gt_item = gt[img_id][task]
            gt_mask = mask.decode(gt_item)
            
            if img_id not in hb:
                hb_mask = np.zeros(gt_mask.shape)
            else:
                hb_item = hb[img_id][task]
                hb_mask = mask.decode(hb_item)

            TP = np.sum(gt_mask == hb_mask)
            FP = np.sum(np.logical_and(hb_mask == 1, gt_mask == 0))
            FN = np.sum(np.logical_and(hb_mask == 0, gt_mask == 1))
            # append to big numpy array 
            if task in results:
                results[task]['tp'] += TP
                results[task]['fp'] += FP
                results[task]['fn'] += FN
            else:
                results[task] = {}
                results[task]['tp'] = TP 
                results[task]['fp'] = FP
                results[task]['fn'] = FN
    return results


def calculate_precision_recall(dict_item):
    TP = dict_item['tp']
    FP = dict_item['fp']
    FN = dict_item['fn']
    p = TP/(TP+FP)
    r = TP/(TP+FN)
    return p, r


def main(args):
    # calculate precision/recall for saliency method
    saliency_results = get_pred_results(args)
    precisions = []
    recalls = []
    for t in sorted(LOCALIZATION_TASKS):
        p, r = calculate_precision_recall(saliency_results[t])
        precisions.append(p)
        recalls.append(r)

    df = pd.DataFrame(columns = ['pathology', 'precision', 'recall'])
    df['pathology'] = sorted(LOCALIZATION_TASKS)
    df['precision'] = precisions
    df['recall'] = recalls
    df.to_csv(f'{args.save_dir}/pred_precision_recall.csv')

    # calculate precision/recall for saliency method
#     hb_results = get_hb()
#     # save human pr
#     precisions = []
#     recalls = []
#     for t in sorted(LOCALIZATION_TASKS):
#         p,r = calc_precision_recall(hb_results[t])
#         precisions.append(p)
#         recalls.append(r)

#     df = pd.DataFrame(columns = ['Pathology', 'Precision', 'Recall'])
#     df['Pathology'] = sorted(LOCALIZATION_TASKS)
#     df['Precision'] = precisions
#     df['Recall'] = recalls
#     df.to_csv(f'pr_auc_results/hb_precision_recall.csv')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_path', type=str,
                        help='directory where ground-truth segmentations are \
                              saved (encoded)')
    parser.add_argument('--map_dir', type=str,
                        help='directory with pickle files containing heatmaps')
    parser.add_argument('--save_dir', default='.',
                        help='where to save precision/recall results')
    args = parser.parse_args()

    main(args)
