import os
from utils_gray import *
from options import get_args
if __name__ == "__main__":
    args = get_args()
    base_path = args.base_path
    train_folder = os.path.join(base_path, args.train_folder) # 원본 이미지 폴더
    val_folder = os.path.join(base_path, args.val_folder)
    
    processed_train_folder = os.path.join(base_path, args.proc_train_folder) # Resize 된 이미지 폴더
    processed_valid_folder = os.path.join(base_path, args.proc_val_folder)
    INPUT_IMAGE_SIZE = (args.target_size, args.target_size)
    
    preprocess_images(train_folder, processed_train_folder, image_size = INPUT_IMAGE_SIZE)
    preprocess_images(val_folder, processed_valid_folder, image_size = INPUT_IMAGE_SIZE)
