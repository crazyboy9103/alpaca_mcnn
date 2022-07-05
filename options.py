import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    defaults = {
        "--base_path": "/home/aiot/alpaca/alpaca_crowd/", 
        "--train_folder": "train_images", 
        "--val_folder": "val_images", 
        "--proc_train_folder": "preprocessed_train", 
        "--proc_val_folder": "preprocessed_valid", 
        "--target_size": 336, 
        "--train_ant_file": "annotation_train.odgt", 
        "--valid_ant_file": "annotation_val.odgt",
        "--train_pkl_file": "train_gt_density_maps_84x84_cleaned.pkl", 
        "--valid_pkl_file": "valid_gt_density_maps_84x84_cleaned.pkl",
        "--gt_target_size": 84, 
        "--model_path": "../tf/alpaca/20220608-135816",
        "--multi_gpu": 0,
    }

    for opt, default in defaults.items():
        parser.add_argument(opt, default=default, type=type(default))
    
    args = parser.parse_args()
    return args