import tensorflow as tf
from train_utils import *
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_gpu", default=0, type=int)
    args = parser.parse_args()

    base_path = "/home/aiot/alpaca/alpaca_crowd/"
    processed_train_folder = os.path.join(base_path, "preprocessed_train_gray") # Resize 된 이미지 폴더
    processed_valid_folder = os.path.join(base_path, "preprocessed_valid_gray")

    train_density_maps_pkl = "train_gt_density_maps_84x84_cleaned.pkl" # gt density map 생성 피클 파일명
    valid_density_maps_pkl = "valid_gt_density_maps_84x84_cleaned.pkl"

    if args.multi_gpu:
        strat = tf.distribute.MirroredStrategy()
        with strat.scope():
            trainer = CrowdHumanTrainer(processed_train_folder, processed_valid_folder, train_density_maps_pkl, valid_density_maps_pkl)
            trainer.train(epochs=100, batch_size=32)
            trainer.evaluate()
    else:
        trainer = CrowdHumanTrainer(processed_train_folder, processed_valid_folder, train_density_maps_pkl, valid_density_maps_pkl)
        trainer.train(epochs=100, batch_size=32)
        trainer.evaluate()
