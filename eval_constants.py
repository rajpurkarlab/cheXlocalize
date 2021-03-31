# constants for localization evaluation
group_dir = '/deep/group/aihc-bootcamp-spring2020/localize'
valid_cam_dirs = {'gradcam_single':'/deep/group/aihc-bootcamp-spring2020/localize/densenet_single/best_densenet_single_ckpt_epoch=0-chexpert_competition_AUROC=0.88.ckpt_valid/cams/',
                     'gradcam_ensemble': '/deep/group/aihc-bootcamp-spring2020/localize/uncertainty_handling/valid_predictions/ensemble_results/cams/',
                     'ig_ensemble': f'{group_dir}/ig_results/ig_ensemble_valid/cams/'}
    
test_cam_dirs = {'gradcam_single':f'{group_dir}/densenet_single/best_densenet_single_ckpt_epoch=0-chexpert_competition_AUROC=0.88.ckpt_test/cams/',
                     'gradcam_ensemble':f'{group_dir}/uncertainty_handling/test_predictions/ensemble_results/cams/',
                     'ig_ensemble': f'{group_dir}/ig_results/ig_ensemble_test/cams/',
                     'ignt_ensemble': f'{group_dir}/ig_results/ignt_ensemble_test/cams/',
                     'gradcamnt_ensemble': f'{group_dir}/gradcam_nt/cams/'}


LOCALIZATION_TASKS =  ["Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Support Devices"
                  ]

PROB_CUTOFF = {'Airspace Opacity': 0.4,
 'Atelectasis': 0.3,
 'Cardiomegaly': 0.1,
 'Consolidation': 0.1,
 'Edema': 0.5,
 'Enlarged Cardiomediastinum': 0,
 'Lung Lesion': 0.1,
 'Pleural Effusion': 0.6,
 'Pneumothorax': 0.6,
 'Support Devices': 0.4}