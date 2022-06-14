import tensorflow as tf
from train_utils import *
from options import get_args
if __name__ == "__main__":
    args = get_args()

    base_path = args.base_path
    processed_train_folder = os.path.join(base_path, "preprocessed_train_gray") # Resize 된 이미지 폴더
    processed_valid_folder = os.path.join(base_path, "preprocessed_valid_gray")

    train_density_maps_pkl = os.path.join(base_path, args.train_pkl_file) # gt density map 생성 피클 파일명
    valid_density_maps_pkl = os.path.join(base_path, args.valid_pkl_file)

    if args.multi_gpu:
        strat = tf.distribute.MirroredStrategy()
        with strat.scope():
            trainer = CrowdHumanTrainer(processed_train_folder, processed_valid_folder, train_density_maps_pkl, valid_density_maps_pkl)
            trainer.train(epochs=100, batch_size=32)
            trainer.evaluate()
    else:
        with tf.device("/GPU:0")
        trainer = CrowdHumanTrainer(processed_train_folder, processed_valid_folder, train_density_maps_pkl, valid_density_maps_pkl)
        trainer.train(epochs=100, batch_size=32)
        trainer.evaluate()
