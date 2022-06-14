import argparse

def get_args():
    parser = argparse.ArgumentParser()
    opts = [
        "--base_path", # path that contains below
        "--train_folder", # folder that contains original images
        "--val_folder", 
        "--proc_train_folder", # folder that resized images will go into 
        "--proc_val_folder", 
        "--target_size",    # image will be resized to (target_size, target_size)
        "--train_ant_file", # .odgt file 
        "--valid_ant_file", 
        "--train_pkl_file", # gt saved as pkl file
        "--valid_pkl_file", 
        "--gt_target_size", # gt of size (gt_target_size,gt_target_size) will be created
        "--model_path" , # saved tf model path
        "--multi_gpu", # 1: use multigpu, 0: use single or cpu
    ]
    defaults = [
        "/home/aiot/alpaca/alpaca_crowd/", 
        "train_images", 
        "val_images", 
        "preprocessed_train", 
        "preprocessed_valid", 
        336, 
        "annotation_train.odgt", 
        "annotation_val.odgt",
        "train_gt_density_maps_84x84_cleaned.pkl", 
        "valid_gt_density_maps_84x84_cleaned.pkl",
        84, 
        "../tf/alpaca/20220608-135816",
        0,
    ]

    for opt, default in zip(opts, defaults):
        parser.add_argument(opt, default=default, type=type(default))
    
    args = parser.parse_args()
    return args