from scipy import ndimage
from tqdm import tqdm
import pickle
import cv2 
import os 
import numpy as np
from PIL import Image

def get_filtered_list(filename):
    with open(filename, "r") as f:
        list = f.read().splitlines()
    return [path.strip(".jpg") for path in list]

preproc_train, preproc_valid = "train.txt", "valid.txt"
train_list, valid_list = get_filtered_list(preproc_train), get_filtered_list(preproc_valid)

def get_gt_density_maps(h_box_centers, density_map_size):
    sigma = 2.5    
    density_maps = {}
    for image_id, centers in tqdm(h_box_centers.items(), desc="processing density maps"):
        density_map = np.zeros(density_map_size)

        for center_x, center_y in centers:
            base_map = np.zeros(density_map_size)
            base_map[center_y, center_x] += 1
            
            density_map += ndimage.filters.gaussian_filter(base_map, sigma = sigma, mode='constant')
        
        density_maps[image_id] = density_map
    return density_maps

def save_gt_density_maps(orig_image_folder, ant_path, density_map_size, pkl_name, ratio_threshold):
    h_box_centers = get_center_hboxes(ant_path, orig_image_folder, density_map_size, ratio_threshold)
    gt_density_maps = get_gt_density_maps(h_box_centers, density_map_size)
    with open(pkl_name, "wb") as f:
        pickle.dump(gt_density_maps, f)
    

def get_center_hboxes(ant_path, input_folder, density_map_size, ratio_threshold):
    ant_file = open(ant_path, "r")
    lines = ant_file.readlines()
    ant_file.close()
    target_width, target_height = density_map_size
    
    h_box_centers = {}
    ratios = []
    for line in tqdm(lines, desc="processing hboxes"):
        json = eval(line)
        image_id=json["ID"]

        if image_id not in train_list and image_id not in valid_list:
            continue
            
        gt_boxes = json["gtboxes"]
        
        img=Image.open(input_folder+"/"+image_id +".jpg")
        width, height = img.size
        orig_img_size = width * height
        
        processed_hbox_centers = []
        
        for gt_box in gt_boxes: 
            tag = gt_box["tag"]
            if "ignore" in gt_box["head_attr"]:
                if gt_box["head_attr"]["ignore"] == 1:
                    continue
            if "unsure" in gt_box["head_attr"]:
                if gt_box["head_attr"]["unsure"] == 1:
                    continue
            if "ignore" in gt_box["extra"]:
                if gt_box["extra"]["ignore"] == 1:
                    continue
            
            if tag != "person":
                continue
            x, y, w, h = gt_box["hbox"]
            hbox_size = w * h
            
            hbox_ratio = hbox_size / orig_img_size
            if hbox_ratio > ratio_threshold:

                cx, cy = x + (w // 2), y + (h // 2)
                px, py = cx / width, cy / height
                if px >= 1:
                    px = 1
                if py >= 1:
                    py = 1

                target_x, target_y = int(px * target_width) - 1, int(py * target_height) - 1
                processed_hbox_centers.append([target_x, target_y])

        if processed_hbox_centers:
            h_box_centers[image_id] = processed_hbox_centers 

    return h_box_centers

def load_gt_density_maps(pkl_name):
    f=open(pkl_name, "rb")
    maps=pickle.load(f)
    f.close()
    return maps

def preprocess_images(orig_image_folder, target_folder, image_size = (224, 224)):
    # resize images in orig_image_folder and save the resized images in target_folder 
    images = os.listdir(orig_image_folder)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
        
    for filename in tqdm(images, desc="resizing images"):
        src_img = orig_image_folder + "/" + filename
        target_img = target_folder + "/" + filename
        
        src_image = cv2.imread(src_img)
        src_image = cv2.resize(src_image, dsize=image_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(target_img, src_image)
