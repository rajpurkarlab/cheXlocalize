# Table 1
METHODS="gradcam gradcampp eigencam occlusion"
for method in $METHODS
do
    python3 heatmap_to_segmentation.py --map_dir ../models/densenet_${method}_test --output_path ./saliency_segmentations/test_${method}_densenet_ensemble_segmentation.json 
    python3 eval.py --gt_path ../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json --pred_path ./saliency_segmentations/test_${method}_densenet_ensemble_segmentation.json --metric miou --save_dir ./results/test_${method}_densenet_ensemble
done

# ig
python3 heatmap_to_segmentation.py --map_dir ../models/densenet_ig_test --output_path ./saliency_segmentations/test_ig_densenet_ensemble_segmentation.json --if_smoothing True --k 100
python3 eval.py --gt_path ../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json --pred_path ./saliency_segmentations/test_ig_densenet_ensemble_segmentation.json --metric miou --save_dir ./results/test_ig_densenet_ensemble

# deeplift
python3 heatmap_to_segmentation.py --map_dir ../models/densenet_deeplift_test --output_path ./saliency_segmentations/test_deeplift_densenet_ensemble_segmentation.json --if_smoothing True --k 50
python3 eval.py --gt_path ../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json --pred_path ./saliency_segmentations/test_deeplift_densenet_ensemble_segmentation.json --metric miou --save_dir ./results/test_deeplift_densenet_ensemble

# lrp
python3 heatmap_to_segmentation.py --map_dir ../models/densenet_lrp_test --output_path ./saliency_segmentations/test_lrp_densenet_ensemble_segmentation.json --if_smoothing True --k 80
python3 eval.py --gt_path ../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json --pred_path ./saliency_segmentations/test_lrp_densenet_ensemble_segmentation.json --metric miou --save_dir ./results/test_lrp_densenet_ensemble


# ED Fig 3
MODELS="resnet inception"
METHODS="gradcam gradcampp"
for model in $MODELS
do
    for method in $METHODS
    do
        python3 heatmap_to_segmentation.py --map_dir ../models/${model}_${method}_test --output_path ./saliency_segmentations/test_${method}_${model}_ensemble_segmentation.json 
        python3 eval.py --gt_path ../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json --pred_path ./saliency_segmentations/test_${method}_${model}_ensemble_segmentation.json --metric miou --save_dir ./results/test_${method}_${model}_ensemble
    done
done

for model in $MODELS
do
    python3 heatmap_to_segmentation.py --map_dir ../models/${model}_ig_test --output_path ./saliency_segmentations/test_ig_${model}_ensemble_segmentation.json --if_smoothing True --k 50 # this is the parameter used in the paper
    python3 eval.py --gt_path ../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json --pred_path ./saliency_segmentations/test_ig_${model}_ensemble_segmentation.json --metric miou --save_dir ./results/test_ig_${model}_ensemble
done

# ED Fig 7
python3 heatmap_to_segmentation.py --map_dir ../models/densenet_gradcam_test_single --output_path ./saliency_segmentations/test_gradcam_densenet_single_segmentation.json
python3 eval.py --gt_path ../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json --pred_path ./saliency_segmentations/test_gradcam_densenet_single_segmentation.json --metric miou --save_dir ./results/test_gradcam_densenet_single
