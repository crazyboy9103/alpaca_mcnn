import tensorflow as tf
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from options import get_args
from utils import *
from train_utils import *
#
# Compute 
#
def compute_stats(loaded_model, valid_generator):
    mae = 0.0
    mape = 0.0
    adj_mae = 0.0
    adj_mape = 0.0

    mapes = {}
    adj_mapes = {}
    gt_nums = {}
    pred_nums = {}
    adj_pred_nums = {}
    for i in tqdm(range(len(valid_generator)), desc="evaluating"):
        image, gt_density = valid_generator[i]
        image_path = valid_generator.image_paths[i]
        gt_num = np.sum(gt_density)
        pred_density = loaded_model(image)
        
        # 실제 예측 값
        pred_num = np.sum(pred_density)
        curr_mae = abs(pred_num - gt_num)
        curr_mape = abs((pred_num - gt_num)/gt_num)
        mae += curr_mae
        mape += curr_mape
        
        # 평균 이상인 것들만 더한 adjusted 예측 값
        adj_pred_num = tf.reduce_sum(pred_density[pred_density > np.mean(pred_density)]).numpy()
        adj_curr_mae = abs(adj_pred_num - gt_num)
        adj_curr_mape = abs((adj_pred_num - gt_num)/gt_num)
        adj_mae += adj_curr_mae
        adj_mape += adj_curr_mape
        
        mapes[image_path] = curr_mape
        adj_mapes[image_path] = adj_curr_mape
        gt_nums[image_path] = gt_num
        pred_nums[image_path] = pred_num
        adj_pred_nums[image_path] = adj_pred_num
        
    mae /= len(valid_generator)
    mape /= len(valid_generator)
    adj_mae /= len(valid_generator)
    adj_mape /= len(valid_generator)
    print(f"""
        mae      : {mae}
        mape     : {mape}
        adj_mae  : {adj_mae}
        adj_mape : {adj_mape}
    """)
    
    # data_to_plot = [
    #     [],
    #     [],
    #     []
    # ]
    # for image_path, gt_num in gt_nums.items():
    #     data_to_plot[0].append(gt_num)
    #     data_to_plot[1].append(mapes[image_path])
    #     data_to_plot[2].append(adj_mapes[image_path])
    
    # plt.figure(1)
    # plt.scatter(data_to_plot[0], data_to_plot[1], label="mape")
    # plt.scatter(data_to_plot[0], data_to_plot[2], label="adj_mape")
    # plt.legend()
    # plt.xlabel("gt_num")
    # plt.ylabel("mape (fraction)")
    # plt.savefig("eval.png", dpi=300)
    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = get_args()
    base_path = args.base_path
    processed_valid_folder = os.path.join(base_path, args.proc_val_folder)
    valid_density_maps_pkl = os.path.join(base_path, args.valid_pkl_file)
    
    valid_generator = CrowdHumanDataGenerator(processed_valid_folder, load_gt_density_maps(valid_density_maps_pkl), 1)
    model_path = args.model_path
    
    print(f"Evaluating {model_path}")
    loaded_model = tf.keras.models.load_model(model_path)
    compute_stats(loaded_model, valid_generator)
