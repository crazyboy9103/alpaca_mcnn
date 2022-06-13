import os
from utils import *
from options import get_args
if __name__ == "__main__":
    args = get_args()
    base_path = args.base_path
    train_folder = os.path.join(base_path, args.train_folder) # 원본 이미지 폴더
    val_folder = os.path.join(base_path, args.val_folder)
    
    ant_train_path = os.path.join(base_path, args.train_ant_file)
    ant_val_path =  os.path.join(base_path, args.valid_ant_file)

    train_density_maps_pkl = os.path.join(base_path, args.train_pkl_file) # gt density map 생성 피클 파일명
    valid_density_maps_pkl = os.path.join(base_path, args.valid_pkl_file)

    GT_DENSITY_SIZE = (args.gt_target_size, args.gt_target_size)
    ratio_threshold = 0 # no thresholding
    if not os.path.exists(train_density_maps_pkl):
        save_gt_density_maps(train_folder, ant_train_path, GT_DENSITY_SIZE, train_density_maps_pkl, ratio_threshold)
    if not os.path.exists(valid_density_maps_pkl):
        save_gt_density_maps(val_folder, ant_val_path, GT_DENSITY_SIZE, valid_density_maps_pkl, ratio_threshold)
