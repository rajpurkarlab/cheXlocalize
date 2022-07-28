# Table 1
METHODS="gradcam gradcampp ig deeplift eigencam lrp occlusion"
for method in $METHODS
do
python3 eval.py --gt_path ../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json --pred_path ../models/densenet_${method}_test --metric hitrate --save_dir ./results/test_${method}_densenet_ensemble
done

# ED Fig 3
METHODS="gradcam gradcampp ig"
MODELS = "resnet inception"
for method in $METHODS
do
    for model in $MODELS
    do    
        python3 eval.py --gt_path ../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json --pred_path ../models/${model}_${method}_test --metric hitrate --save_dir ./results/test_${method}_${model}_ensemble
    done
done

# ED Fig 7
python3 eval.py --gt_path ../cheXlocalize_data_code/dataset/segmentations/ground_truth/gt_segmentations_test.json --pred_path ../models/densenet_gradcam_test_single --metric hitrate --save_dir ./results/test_gradcam_densenet_single