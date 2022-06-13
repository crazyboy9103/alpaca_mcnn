import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import *
from train_utils import *
from options import get_args
    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = get_args()

    base_path = args.base_path
    processed_valid_folder = os.path.join(base_path, args.proc_val_folder)
    valid_density_maps_pkl = os.path.join(base_path, args.valid_pkl_file)

    valid_generator = CrowdHumanDataGenerator(processed_valid_folder, load_gt_density_maps(valid_density_maps_pkl), 1)
    model_path = args.model_path
    # log_path = "./alpaca"
    # paths = [path for path in os.listdir(log_path) if "-" in path]
    # paths = sorted(paths)

    # log_path = os.path.join(log_path, paths[-1])
    print(f"Evaluating {model_path}")
    loaded_model = tf.keras.models.load_model(model_path)

    rand_idx = np.random.randint(0, len(valid_generator))
    image, label = valid_generator[rand_idx]

    pred = loaded_model(image)

    fig = plt.figure()
  
    ax1 = fig.add_subplot(131)
    ax1.imshow(np.squeeze(image))
    ax1.set_title("Image")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(132)
    ax2.imshow(np.squeeze(label))
    ax2.set_title(f"Gt={np.sum(label):.2f}")
    ax2.axis("off")
    
    ax3 = fig.add_subplot(133)
    ax3.imshow(np.squeeze(pred))
    ax3.set_title(f"Prediction={np.sum(pred):.2f}")  
    ax3.axis("off")

    fig.savefig("render.png", dpi=600)
