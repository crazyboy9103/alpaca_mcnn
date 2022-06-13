import os
from utils_gray import *
if __name__ == "__main__":
    base_path = "/home/aiot/alpaca/alpaca_crowd/"
    train_folder = os.path.join(base_path, "train_images") # 원본 이미지 폴더
    val_folder = os.path.join(base_path, "val_images")
    
    processed_train_folder = os.path.join(base_path, "preprocessed_train_gray") # Resize 된 이미지 폴더
    processed_valid_folder = os.path.join(base_path, "preprocessed_valid_gray")
    INPUT_IMAGE_SIZE = (336, 336)
    
    preprocess_images(train_folder, processed_train_folder, image_size = INPUT_IMAGE_SIZE)
    preprocess_images(val_folder, processed_valid_folder, image_size = INPUT_IMAGE_SIZE)
